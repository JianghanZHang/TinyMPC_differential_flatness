"""Circle-in-Square Constrained Tracking Experiment

Track a circular trajectory in the xy-plane with square position bounds.
The circle radius R slightly exceeds the square half-width L, so the
constrained controller must deviate from the reference to stay inside.

Compares 4 variants along the complexity cascade:
  1. Online Relinearization + ADMM  -- O(n^3) + K*O(n^2*N)  (classical NMPC)
  2. Flat Template + ADMM           -- K*O(n^2*N)            (cached DARE)
  3. Barrier Template (zero-iter)   -- O(n*m) ~260 ops       (constraint-aware)
  4. Analytic Template (zero-iter)  -- O(n*m) ~260 ops       (clip only)

"Wai yuan nei fang" (外圆内方) -- circle outside, square inside.

Usage:
    python examples/circle_square/circle_square.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.quadrotor import QuadrotorDynamics
from src.flat_linearization import (
    get_quad_params, flat_equilibrium, linearise,
    quat_to_euler_state,
)
from src.flat_template import FlatTemplateMPC
from src.barrier_template import BarrierTemplate
from src.analytic_template import AnalyticTemplate


# ------------------------------------------------------------------ #
#  Circle trajectory
# ------------------------------------------------------------------ #

class CircleReference:
    """Circular trajectory in the xy-plane at fixed altitude."""

    def __init__(self, radius=0.3, period=8.0):
        self.radius = radius
        self.period = period
        self.w = 2 * np.pi / period

    def generate_reference(self, t):
        """12D reference: [p(3), angles(3), v(3), omega(3)]."""
        s = min(t / 2.0, 1.0)  # smooth start over 2s
        R, w = self.radius, self.w
        ref = np.zeros(12)
        ref[0] = R * np.cos(w * t) * s
        ref[1] = R * np.sin(w * t) * s
        ref[6] = -R * w * np.sin(w * t) * s
        ref[7] =  R * w * np.cos(w * t) * s
        return ref

    def compute_alpha(self, t):
        """Flat parameter alpha = [ax, ay, az, psi] along circle."""
        if t < 2.0:
            return np.zeros(4)
        R, w = self.radius, self.w
        ax = -R * w**2 * np.cos(w * t)
        ay = -R * w**2 * np.sin(w * t)
        return np.array([ax, ay, 0.0, 0.0])


# ------------------------------------------------------------------ #
#  Helper: per-step horizon bounds
# ------------------------------------------------------------------ #

def compute_horizon_bounds(traj, t, dt, N, L_bound):
    """Compute per-step state bounds for the entire horizon.

    Absolute constraint: |pos_x| <= L, |pos_y| <= L
    Error coords at step j: dx[0] = pos_x - ref_x(t+j*dt)
    So: -L - ref_x(t+j*dt) <= dx[0] <= L - ref_x(t+j*dt)
    """
    nx = 12
    xmin_h = np.full((nx, N), -1000.0)
    xmax_h = np.full((nx, N),  1000.0)
    for j in range(N):
        ref = traj.generate_reference(t + j * dt)
        xmin_h[0, j] = -L_bound - ref[0]
        xmax_h[0, j] =  L_bound - ref[0]
        xmin_h[1, j] = -L_bound - ref[1]
        xmax_h[1, j] =  L_bound - ref[1]
    return xmin_h, xmax_h


# ------------------------------------------------------------------ #
#  Variant 1: Online Relinearization + ADMM
# ------------------------------------------------------------------ #

def sim_online_relin(x0, quad, qp, Q, R, N, rho, NSIM, dt,
                     traj, L_bound):
    """Online relinearization: fresh linearise() + DARE + ADMM each step.

    Uses FlatTemplateMPC with NO grid cache, so each call to solve()
    triggers a fresh linearise() + DARE solve -- the O(n^3) classical
    NMPC baseline.  Per-step horizon bounds via xmin_horizon/xmax_horizon.
    """
    # Empty grid => every solve() call recomputes DARE from scratch
    flat_mpc = FlatTemplateMPC(qp, Q, R, N, rho, grid_alphas=None)
    flat_mpc.set_bounds(
        umax=[0.3]*4, umin=[-0.3]*4,
        xmax=[10.]*12, xmin=[-10.]*12,
    )
    flat_mpc.set_tols_iters(max_iter=500, abs_pri_tol=1e-3, abs_dua_tol=1e-3)

    x_curr = np.copy(x0)
    t = 0.0
    x_all, u_all, iters_all = [], [], []

    t0 = time.perf_counter()
    for step in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        alpha = traj.compute_alpha(t)
        xr = traj.generate_reference(t)
        ref_pos, ref_vel = xr[0:3], xr[6:9]

        x_eq, u_eq = flat_equilibrium(alpha, qp)
        dx = np.zeros(12)
        dx[0:3]  = x_euler[0:3] - ref_pos
        dx[3:6]  = x_euler[3:6] - x_eq[3:6]
        dx[6:9]  = x_euler[6:9] - ref_vel
        dx[9:12] = x_euler[9:12]

        # Per-step horizon bounds in error coords
        xmin_h, xmax_h = compute_horizon_bounds(traj, t, dt, N, L_bound)

        xo, uo, st, k = flat_mpc.solve(dx, alpha,
                                         xmin_horizon=xmin_h,
                                         xmax_horizon=xmax_h)
        u = u_eq + uo[:, 0]
        u = np.clip(u, 0.0, None)

        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        x_all.append(np.copy(x_curr))
        u_all.append(u)
        iters_all.append(k)

    wall = time.perf_counter() - t0
    return np.array(x_all), np.array(u_all), iters_all, wall


# ------------------------------------------------------------------ #
#  Variant 2: Flat Template MPC (ADMM with per-step horizon bounds)
# ------------------------------------------------------------------ #

def sim_flat_template(x0, quad, qp, Q, R, N, rho, NSIM, dt,
                      traj, L_bound, grid_alphas):
    """Flat Template MPC: cached DARE + ADMM with per-step horizon bounds."""
    flat_mpc = FlatTemplateMPC(qp, Q, R, N, rho, grid_alphas=grid_alphas)
    flat_mpc.set_bounds(
        umax=[0.3]*4, umin=[-0.3]*4,
        xmax=[10.]*12, xmin=[-10.]*12,
    )
    flat_mpc.set_tols_iters(max_iter=500, abs_pri_tol=1e-3, abs_dua_tol=1e-3)

    x_curr = np.copy(x0)
    t = 0.0
    x_all, u_all, iters_all = [], [], []

    t0 = time.perf_counter()
    for step in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        alpha = traj.compute_alpha(t)
        xr = traj.generate_reference(t)
        ref_pos, ref_vel = xr[0:3], xr[6:9]

        x_eq, u_eq = flat_equilibrium(alpha, qp)
        dx = np.zeros(12)
        dx[0:3]  = x_euler[0:3] - ref_pos
        dx[3:6]  = x_euler[3:6] - x_eq[3:6]
        dx[6:9]  = x_euler[6:9] - ref_vel
        dx[9:12] = x_euler[9:12]

        # Per-step horizon bounds in error coords
        xmin_h, xmax_h = compute_horizon_bounds(traj, t, dt, N, L_bound)

        xo, uo, st, k = flat_mpc.solve(dx, alpha,
                                         xmin_horizon=xmin_h,
                                         xmax_horizon=xmax_h)
        u = u_eq + uo[:, 0]
        u = np.clip(u, 0.0, None)

        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        x_all.append(np.copy(x_curr))
        u_all.append(u)
        iters_all.append(k)

    wall = time.perf_counter() - t0
    return np.array(x_all), np.array(u_all), iters_all, wall


# ------------------------------------------------------------------ #
#  Variant 3: Barrier Template (zero-iteration, constraint-aware)
# ------------------------------------------------------------------ #

def sim_barrier(x0, quad, qp, Q, R, rho, NSIM, dt, traj, L_bound):
    """Barrier Template: zero-iteration, soft constraint awareness via Q_mu + R_mu."""
    _, u_eq_h = flat_equilibrium(np.zeros(4), qp)

    bt = BarrierTemplate(qp, Q, R, rho=rho, mu=50.0,
                         u_min=np.zeros(4), u_max=u_eq_h + 0.3,
                         x_min=np.array([-L_bound, -L_bound] + [-1000.]*10),
                         x_max=np.array([ L_bound,  L_bound] + [ 1000.]*10))
    bt.precompute()

    x_curr = np.copy(x0)
    t = 0.0
    x_all, u_all = [], []

    t0 = time.perf_counter()
    for _ in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        alpha = traj.compute_alpha(t)
        xr = traj.generate_reference(t)
        ref_pos, ref_vel = xr[0:3], xr[6:9]

        x_for_ctrl = np.zeros(12)
        x_for_ctrl[0:3]  = x_euler[0:3] - ref_pos
        x_for_ctrl[3:6]  = x_euler[3:6]
        x_for_ctrl[6:9]  = x_euler[6:9] - ref_vel
        x_for_ctrl[9:12] = x_euler[9:12]

        u = bt.control(x_for_ctrl, alpha,
                       u_min=np.zeros(4), u_max=u_eq_h + 0.3)
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        x_all.append(np.copy(x_curr))
        u_all.append(u)

    wall = time.perf_counter() - t0
    return np.array(x_all), np.array(u_all), wall


# ------------------------------------------------------------------ #
#  Variant 4: Analytic Template (zero-iteration, no constraints)
# ------------------------------------------------------------------ #

def sim_analytic(x0, quad, qp, Q, R, rho, NSIM, dt, traj):
    """Analytic Template: zero-iteration, no state constraint mechanism."""
    at = AnalyticTemplate(qp, Q, R, rho=rho)
    at.precompute()
    _, u_eq_h = flat_equilibrium(np.zeros(4), qp)

    x_curr = np.copy(x0)
    t = 0.0
    x_all, u_all = [], []

    t0 = time.perf_counter()
    for _ in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        alpha = traj.compute_alpha(t)
        xr = traj.generate_reference(t)
        ref_pos, ref_vel = xr[0:3], xr[6:9]

        x_for_ctrl = np.zeros(12)
        x_for_ctrl[0:3]  = x_euler[0:3] - ref_pos
        x_for_ctrl[3:6]  = x_euler[3:6]
        x_for_ctrl[6:9]  = x_euler[6:9] - ref_vel
        x_for_ctrl[9:12] = x_euler[9:12]

        u = at.control(x_for_ctrl, alpha,
                       u_min=np.zeros(4), u_max=u_eq_h + 0.3)
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        x_all.append(np.copy(x_curr))
        u_all.append(u)

    wall = time.perf_counter() - t0
    return np.array(x_all), np.array(u_all), wall


# ------------------------------------------------------------------ #
#  Plotting
# ------------------------------------------------------------------ #

def plot_results(traj, L_bound, results, dt, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    NSIM = max(len(r['x_all']) for r in results)
    t_ref = np.linspace(0, NSIM * dt, 500)
    ref_xy = np.array([traj.generate_reference(t)[0:2] for t in t_ref])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], 'k--', linewidth=1.5,
            label='Circle reference', alpha=0.4)

    # Square boundary
    ax.plot([-L_bound, L_bound, L_bound, -L_bound, -L_bound],
            [-L_bound, -L_bound, L_bound, L_bound, -L_bound],
            'r-', linewidth=2, label=f'Square bound ($L=R/\\sqrt{{2}}$)')
    sq = Rectangle((-L_bound, -L_bound), 2*L_bound, 2*L_bound,
                   linewidth=0, facecolor='red', alpha=0.06)
    for sx, sy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        ax.plot(sx*L_bound, sy*L_bound, 'ro', markersize=6, zorder=5)
    ax.add_patch(sq)

    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    for i, r in enumerate(results):
        xy = r['x_all'][:, 0:2]
        ax.plot(xy[:, 0], xy[:, 1], color=colors[i], linewidth=1.8,
                label=r['name'], alpha=0.85)

    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('Circle Tracking with Square Bound', fontsize=13)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    lim = traj.radius * 1.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax2 = axes[1]
    for i, r in enumerate(results):
        xy = r['x_all'][:, 0:2]
        violation = np.maximum(np.abs(xy) - L_bound, 0.0)
        max_viol = np.max(violation, axis=1)
        t_axis = np.arange(len(max_viol)) * dt
        ax2.plot(t_axis, max_viol * 1000, color=colors[i], linewidth=1.5,
                 label=r['name'])

    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Max position violation (mm)', fontsize=12)
    ax2.set_title('Constraint Violation', fontsize=13)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")
    plt.close()


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    quad = QuadrotorDynamics()
    qp = get_quad_params(quad)
    dt = quad.dt

    # Geometry: square inscribed in circle
    R_circle = 0.30    # m
    period = 8.0       # s
    L_bound = R_circle / np.sqrt(2)  # ~0.212 m

    NSIM = 500         # 10s at 50Hz (>1 full period)
    N_horizon = 20
    rho = 50.0

    # Cost matrices
    Q = np.diag(1.0 / np.array([
        0.05, 0.05, 0.05,     # position
        0.5, 0.5, 0.05,       # attitude
        0.5, 0.5, 0.5,        # velocity
        0.7, 0.7, 0.5,        # omega
    ])**2)
    R_cost = np.diag(1.0 / np.array([0.3, 0.3, 0.3, 0.3])**2)

    traj = CircleReference(radius=R_circle, period=period)

    # Initial state: at (R, 0, 0) on circle
    xg = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    x0 = np.copy(xg)
    xr0 = traj.generate_reference(0.0)
    x0[0:3] = xr0[0:3]
    x0[7:10] = xr0[6:9]

    # Grid alphas for Flat Template cache
    t_grid = np.linspace(0, period, 30)
    grid_alphas = [traj.compute_alpha(t) for t in t_grid]
    grid_alphas.append(np.zeros(4))

    print("=" * 70)
    print("  Circle-in-Square Constrained Tracking -- 4-Variant Cascade")
    print(f"  R={R_circle} m, L=R/sqrt(2)={L_bound:.4f} m")
    print(f"  Max excess at cardinal points: {R_circle-L_bound:.4f} m")
    print(f"  period={period} s, NSIM={NSIM}, N={N_horizon}, rho={rho}")
    print("=" * 70)

    results = []

    # Variant 1: Online Relinearization + ADMM
    print("\n  [1/4] Online Relinearization + ADMM ...")
    x1, u1, iters1, wall1 = sim_online_relin(
        x0, quad, qp, Q, R_cost, N_horizon, rho, NSIM, dt,
        traj, L_bound)
    results.append({
        'name': 'Online Relin (ADMM)',
        'x_all': x1, 'u_all': u1,
        'iters': iters1, 'wall': wall1,
    })

    # Variant 2: Flat Template + ADMM (per-step horizon bounds)
    print("  [2/4] Flat Template MPC (ADMM, per-step bounds) ...")
    x2, u2, iters2, wall2 = sim_flat_template(
        x0, quad, qp, Q, R_cost, N_horizon, rho, NSIM, dt,
        traj, L_bound, grid_alphas)
    results.append({
        'name': 'Flat Template (ADMM)',
        'x_all': x2, 'u_all': u2,
        'iters': iters2, 'wall': wall2,
    })

    # Variant 3: Barrier Template (zero-iteration)
    print("  [3/4] Barrier Template (zero-iteration) ...")
    x3, u3, wall3 = sim_barrier(
        x0, quad, qp, Q, R_cost, rho, NSIM, dt, traj, L_bound)
    results.append({
        'name': 'Barrier (zero-iter)',
        'x_all': x3, 'u_all': u3,
        'iters': None, 'wall': wall3,
    })

    # Variant 4: Analytic Template (zero-iteration)
    print("  [4/4] Analytic Template (zero-iteration) ...")
    x4, u4, wall4 = sim_analytic(
        x0, quad, qp, Q, R_cost, rho, NSIM, dt, traj)
    results.append({
        'name': 'Analytic (zero-iter)',
        'x_all': x4, 'u_all': u4,
        'iters': None, 'wall': wall4,
    })

    # --- Summary table ---
    print(f"\n{'=' * 90}")
    names = [r['name'] for r in results]
    print(f"  {'Metric':<28}", end='')
    for n in names:
        print(f"{n:>16}", end='')
    print()
    print(f"  {'-' * 86}")

    for r in results:
        xy = r['x_all'][:, 0:2]
        violation = np.maximum(np.abs(xy) - L_bound, 0.0)
        r['max_viol'] = np.max(violation)
        r['mean_viol'] = np.mean(np.max(violation, axis=1))
        pos_ref = np.array([traj.generate_reference(i * dt)[0:2]
                            for i in range(len(r['x_all']))])
        r['pos_err'] = np.mean(np.linalg.norm(xy - pos_ref, axis=1))

    for metric, key, fmt in [
        ('Avg pos error (m)', 'pos_err', '.4f'),
        ('Max violation (mm)', 'max_viol', '.1f'),
        ('Mean violation (mm)', 'mean_viol', '.1f'),
        ('Wall time (s)', 'wall', '.3f'),
    ]:
        print(f"  {metric:<28}", end='')
        for r in results:
            val = r[key]
            if 'violation' in metric:
                val *= 1000
            print(f"{val:>16{fmt}}", end='')
        print()

    # ADMM iters row
    print(f"  {'Avg ADMM iters':<28}", end='')
    for r in results:
        if r['iters'] is not None:
            print(f"{np.mean(r['iters']):>16.1f}", end='')
        else:
            print(f"{'0':>16}", end='')
    print()

    # Complexity row
    print(f"  {'Complexity':<28}", end='')
    complexities = ['O(n^3+Kn^2N)', 'K*O(n^2*N)', 'O(n*m)~260', 'O(n*m)~260']
    for c in complexities:
        print(f"{c:>16}", end='')
    print()

    # Plot
    out_dir = Path(__file__).parent
    plot_results(traj, L_bound, results, dt,
                 save_path=str(out_dir / 'circle_square.png'))
    fig_dir = Path(__file__).parent.parent.parent / 'write_up' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_results(traj, L_bound, results, dt,
                 save_path=str(fig_dir / 'fig_circle_square.png'))

    print(f"\n  DONE")


if __name__ == '__main__':
    main()
