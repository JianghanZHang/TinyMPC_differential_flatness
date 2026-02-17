"""Three-Variant Comparison Demo

Side-by-side benchmark of all three controller variants:
  1. Original TinyMPC          -- fixed hover linearization + ADMM
  2. Online Relinearization    -- autograd Jacobian each step + ADMM
  3. Flat Template MPC         -- analytic alpha-linearization + ADMM

Also demonstrates the Analytic Template (zero-iteration) as a bonus.

Usage:
    python examples/analytic/demo.py [--hover] [--traj]
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
import argparse
import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from src.flat_linearization import (
    get_quad_params, flat_equilibrium, linearise,
    quat_to_euler_state, euler_to_quat,
)
from src.flat_template import FlatTemplateMPC
from src.analytic_template import AnalyticTemplate
from utils.reference_trajectories import Figure8Reference


# ------------------------------------------------------------------ #
#  Alpha from trajectory
# ------------------------------------------------------------------ #

def compute_alpha_figure8(t, A, w):
    if t < 1.0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    ax = -A * w**2 * np.sin(w * t)
    az = -2 * A * w**2 * np.sin(2 * w * t)
    return np.array([ax, 0.0, az, 0.0])


# ------------------------------------------------------------------ #
#  Simulation wrappers  (one per controller variant)
# ------------------------------------------------------------------ #

def sim_original_tinympc(x0, quad, Q, R, N, rho, NSIM, dt,
                         trajectory=None, mode='hover'):
    """Variant 1: fixed hover linearization + ADMM."""
    xg = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0], dtype=np.float64)
    ug = np.array(quad.hover_thrust, dtype=np.float64)
    A_lin, B_lin = quad.get_linearized_dynamics(xg, ug)

    mpc = TinyMPC(A_lin, B_lin, Q, R, N, rho=rho, mode=mode)
    if mode == 'hover':
        mpc.set_bounds([1.0-ug[0]]*4, [-ug[0]]*4, [1000.]*12, [-1000.]*12)
    else:
        mpc.set_bounds([0.3]*4, [-0.3]*4,
                       [2.]*3+[1.]*3+[2.]*3+[2.]*3,
                       [-2.]*3+[-1.]*3+[-2.]*3+[-2.]*3)

    hover = trajectory is None
    x_curr = np.copy(x0)
    qg = np.array([1.0, 0, 0, 0])
    t, x_all, u_all, iters = 0.0, [], [], []

    t0 = time.perf_counter()
    for _ in range(NSIM):
        q = x_curr[3:7] / np.linalg.norm(x_curr[3:7])
        phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(qg).T @ q)
        if hover:
            dx = np.hstack([x_curr[0:3], phi, x_curr[7:10], x_curr[10:13]])
        else:
            xr = trajectory.generate_reference(t)
            dx = np.hstack([x_curr[0:3]-xr[0:3], phi,
                            x_curr[7:10]-xr[6:9], x_curr[10:13]])

        xi = np.copy(mpc.x_prev); xi[:,0] = dx
        ui = np.copy(mpc.u_prev)
        xo, uo, st, k = mpc.solve_admm(xi, ui)

        u = ug + uo[:,0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        x_all.append(np.copy(x_curr)); u_all.append(u); iters.append(k)
    wall = time.perf_counter() - t0

    return np.array(x_all), np.array(u_all), iters, wall


def sim_online_relin(x0, quad, Q, R, N, rho, NSIM, dt,
                     trajectory=None, mode='hover'):
    """Variant 2: online relinearization (autograd Jacobian each step) + ADMM.

    At each MPC step the system is relinearised around the current operating
    point and a **fresh** TinyMPC instance is built (new DARE cache).
    This is the most accurate but most expensive variant.
    """
    xg = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0], dtype=np.float64)
    ug = np.array(quad.hover_thrust, dtype=np.float64)
    hover = trajectory is None
    qg = np.array([1.0, 0, 0, 0])
    x_curr = np.copy(x0)
    t, x_all, u_all, iters = 0.0, [], [], []

    # We keep a warm-start buffer that persists across re-linearizations
    warm = None

    t0 = time.perf_counter()
    for _ in range(NSIM):
        # ---- relinearise around current state/input ----
        q_norm = x_curr[3:7] / np.linalg.norm(x_curr[3:7])
        x_lin_pt = np.copy(x_curr); x_lin_pt[3:7] = q_norm
        A_new, B_new = quad.get_linearized_dynamics(x_lin_pt, ug)

        mpc = TinyMPC(A_new, B_new, Q, R, N, rho=rho, mode=mode)
        if mode == 'hover':
            mpc.set_bounds([1.0-ug[0]]*4, [-ug[0]]*4, [1000.]*12, [-1000.]*12)
        else:
            mpc.set_bounds([0.3]*4, [-0.3]*4,
                           [2.]*3+[1.]*3+[2.]*3+[2.]*3,
                           [-2.]*3+[-1.]*3+[-2.]*3+[-2.]*3)

        # Restore warm start
        if warm is not None:
            mpc.x_prev = warm['x']; mpc.u_prev = warm['u']
            mpc.v_prev = warm['v']; mpc.z_prev = warm['z']
            mpc.g_prev = warm['g']; mpc.y_prev = warm['y']
            mpc.q_prev = warm['q']

        # ---- compute error state ----
        phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(qg).T @ q_norm)
        if hover:
            dx = np.hstack([x_curr[0:3], phi, x_curr[7:10], x_curr[10:13]])
        else:
            xr = trajectory.generate_reference(t)
            dx = np.hstack([x_curr[0:3]-xr[0:3], phi,
                            x_curr[7:10]-xr[6:9], x_curr[10:13]])

        xi = np.copy(mpc.x_prev); xi[:,0] = dx
        ui = np.copy(mpc.u_prev)
        xo, uo, st, k = mpc.solve_admm(xi, ui)

        # Save warm start
        warm = dict(x=mpc.x_prev, u=mpc.u_prev, v=mpc.v_prev,
                    z=mpc.z_prev, g=mpc.g_prev, y=mpc.y_prev, q=mpc.q_prev)

        u = ug + uo[:,0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        x_all.append(np.copy(x_curr)); u_all.append(u); iters.append(k)
    wall = time.perf_counter() - t0

    return np.array(x_all), np.array(u_all), iters, wall


def sim_flat_template(x0, quad, qp, Q, R, N, rho, NSIM, dt,
                      trajectory=None, A_traj=0.5, w_traj=1.0,
                      grid_alphas=None):
    """Variant 3: Flat Template MPC (analytic alpha-linearization + ADMM)."""
    hover = trajectory is None
    flat_mpc = FlatTemplateMPC(qp, Q, R, N, rho, grid_alphas=grid_alphas)
    _, u_eq_h = flat_equilibrium(np.zeros(4), qp)
    if hover:
        flat_mpc.set_bounds([1.0-u_eq_h[0]]*4, [-u_eq_h[0]]*4,
                            [1000.]*12, [-1000.]*12)
        flat_mpc.set_tols_iters(max_iter=500, abs_pri_tol=1e-2, abs_dua_tol=1e-2)
    else:
        flat_mpc.set_bounds([0.3]*4, [-0.3]*4,
                            [2.]*3+[1.]*3+[2.]*3+[2.]*3,
                            [-2.]*3+[-1.]*3+[-2.]*3+[-2.]*3)

    x_curr = np.copy(x0)
    t, x_all, u_all, iters = 0.0, [], [], []

    t0 = time.perf_counter()
    for _ in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        if hover:
            alpha = np.zeros(4); ref_pos = np.zeros(3); ref_vel = np.zeros(3)
        else:
            alpha = compute_alpha_figure8(t, A_traj, w_traj)
            xr = trajectory.generate_reference(t)
            ref_pos, ref_vel = xr[0:3], xr[6:9]

        x_eq, u_eq = flat_equilibrium(alpha, qp)
        dx = np.zeros(12)
        dx[0:3]  = x_euler[0:3]  - ref_pos
        dx[3:6]  = x_euler[3:6]  - x_eq[3:6]
        dx[6:9]  = x_euler[6:9]  - ref_vel
        dx[9:12] = x_euler[9:12]

        xo, uo, st, k = flat_mpc.solve(dx, alpha)
        u = u_eq + uo[:,0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        x_all.append(np.copy(x_curr)); u_all.append(u); iters.append(k)
    wall = time.perf_counter() - t0

    return np.array(x_all), np.array(u_all), iters, wall


def sim_analytic_template(x0, quad, qp, Q, R, rho, NSIM, dt,
                          trajectory=None, A_traj=0.5, w_traj=1.0):
    """Bonus: Analytic Template (zero-iteration, ~260 ops/step)."""
    hover = trajectory is None
    at = AnalyticTemplate(qp, Q, R, rho=rho)
    at.precompute()
    _, u_eq_h = flat_equilibrium(np.zeros(4), qp)

    if hover:
        u_min_abs = np.zeros(4)
        u_max_abs = np.ones(4)
    else:
        u_min_abs = u_eq_h - 0.3
        u_max_abs = u_eq_h + 0.3

    x_curr = np.copy(x0)
    t, x_all, u_all = 0.0, [], []

    t0 = time.perf_counter()
    for _ in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        if hover:
            alpha = np.zeros(4); ref_pos = np.zeros(3); ref_vel = np.zeros(3)
        else:
            alpha = compute_alpha_figure8(t, A_traj, w_traj)
            xr = trajectory.generate_reference(t)
            ref_pos, ref_vel = xr[0:3], xr[6:9]

        x_for_ctrl = np.zeros(12)
        x_for_ctrl[0:3]  = x_euler[0:3] - ref_pos
        x_for_ctrl[3:6]  = x_euler[3:6]
        x_for_ctrl[6:9]  = x_euler[6:9] - ref_vel
        x_for_ctrl[9:12] = x_euler[9:12]

        u = at.control(x_for_ctrl, alpha, u_min=u_min_abs, u_max=u_max_abs)
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        x_all.append(np.copy(x_curr)); u_all.append(u)
    wall = time.perf_counter() - t0

    return np.array(x_all), np.array(u_all), wall


# ------------------------------------------------------------------ #
#  Tracking error
# ------------------------------------------------------------------ #

def pos_errors(x_all, trajectory, dt, hover):
    errs = []
    for i, x in enumerate(x_all):
        ref = np.zeros(3) if hover else trajectory.generate_reference(i*dt)[0:3]
        errs.append(np.linalg.norm(x[0:3] - ref))
    return np.array(errs)


# ------------------------------------------------------------------ #
#  Main: 3-variant comparison
# ------------------------------------------------------------------ #

def run_comparison(hover=True):
    quad = QuadrotorDynamics()
    qp   = get_quad_params(quad)
    dt   = quad.dt

    xg = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0], dtype=np.float64)

    if hover:
        label = "Hover Stabilization"
        Q = np.diag(1.0 / np.array([0.1,0.1,0.1, 0.5,0.5,0.05,
                                      0.5,0.5,0.5, 0.7,0.7,0.2])**2)
        R = np.diag(1.0 / np.array([0.5,0.5,0.5,0.5])**2)
        N, rho, NSIM = 10, 85.0, 100
        x0 = np.copy(xg); x0[0:3] = [0.2, 0.2, -0.2]
        trajectory = None
        grid_alphas = [np.zeros(4)]
        A_traj, w_traj = 0.5, 1.0
    else:
        label = "Figure-8 Trajectory"
        Q = np.diag(1.0 / np.array([0.01,0.01,0.01, 0.5,0.5,0.05,
                                      0.5,0.5,0.5, 0.7,0.7,0.5])**2)
        R = np.diag(1.0 / np.array([0.1,0.1,0.1,0.1])**2)
        N, rho, NSIM = 15, 5.0, 200
        A_traj, w_traj = 0.5, 2*np.pi/3.7
        trajectory = Figure8Reference(A=A_traj, w=w_traj, segment_type='full')
        xr0 = trajectory.generate_reference(0.0)
        x0 = np.copy(xg); x0[0:3] = xr0[0:3]; x0[7:10] = xr0[6:9]
        t_grid = np.linspace(0, 2*np.pi/w_traj, 20)
        grid_alphas = [compute_alpha_figure8(t, A_traj, w_traj) for t in t_grid]
        grid_alphas.append(np.zeros(4))

    mode = 'hover' if hover else 'traj'

    print(f"\n{'='*60}")
    print(f"  {label}  --  3-variant comparison")
    print(f"{'='*60}")

    # Variant 1
    print("\n  [1/3] Original TinyMPC (fixed hover linearization) ...")
    xa, ua, ia, wa = sim_original_tinympc(
        x0, quad, Q, R, N, rho, NSIM, dt, trajectory, mode)

    # Variant 2
    print("  [2/3] Online relinearization + TinyMPC ...")
    xb, ub, ib, wb = sim_online_relin(
        x0, quad, Q, R, N, rho, NSIM, dt, trajectory, mode)

    # Variant 3
    print("  [3/3] Flat Template MPC ...")
    xc, uc, ic, wc = sim_flat_template(
        x0, quad, qp, Q, R, N, rho, NSIM, dt,
        trajectory, A_traj, w_traj, grid_alphas)

    ea = pos_errors(xa, trajectory, dt, hover)
    eb = pos_errors(xb, trajectory, dt, hover)
    ec = pos_errors(xc, trajectory, dt, hover)

    # Print table
    names = ['TinyMPC', 'Online Relin', 'Flat Template']
    errs  = [ea, eb, ec]
    its   = [ia, ib, ic]
    walls = [wa, wb, wc]

    print(f"\n  {'Metric':<28} ", end='')
    for n in names: print(f"{n:>14}", end='')
    print()
    print(f"  {'-'*70}")
    for metric, fn in [('Avg pos error (m)', np.mean),
                       ('Max pos error (m)', np.max)]:
        print(f"  {metric:<28} ", end='')
        for e in errs: print(f"{fn(e):>14.4f}", end='')
        print()
    print(f"  {'Final pos error (m)':<28} ", end='')
    for e in errs: print(f"{e[-1]:>14.4f}", end='')
    print()
    print(f"  {'Avg ADMM iters':<28} ", end='')
    for it in its: print(f"{np.mean(it):>14.1f}", end='')
    print()
    print(f"  {'Total ADMM iters':<28} ", end='')
    for it in its: print(f"{sum(it):>14d}", end='')
    print()
    print(f"  {'Wall time (s)':<28} ", end='')
    for w in walls: print(f"{w:>14.3f}", end='')
    print()

    # Bonus: analytic template
    print(f"\n  --- Bonus: Analytic Template (zero iteration) ---")
    xd, ud, wd = sim_analytic_template(
        x0, quad, qp, Q, R, rho, NSIM, dt,
        trajectory, A_traj, w_traj)
    ed = pos_errors(xd, trajectory, dt, hover)
    print(f"  Avg pos error : {np.mean(ed):.4f} m")
    print(f"  Max pos error : {np.max(ed):.4f} m")
    print(f"  Wall time     : {wd:.3f} s  (0 ADMM iters)")

    # Save
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    tag = 'hover' if hover else 'traj'
    np.savetxt(data_dir / f'comparison_{tag}_errors.txt',
               np.column_stack([ea, eb, ec, ed]),
               header='TinyMPC OnlineRelin FlatTemplate AnalyticTemplate')
    print(f"\n  Saved to {data_dir}/comparison_{tag}_errors.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hover', action='store_true',
                        help='Run hover comparison only')
    parser.add_argument('--traj',  action='store_true',
                        help='Run trajectory comparison only')
    args = parser.parse_args()

    if args.hover:
        run_comparison(hover=True)
    elif args.traj:
        run_comparison(hover=False)
    else:
        run_comparison(hover=True)
        run_comparison(hover=False)


if __name__ == '__main__':
    main()
