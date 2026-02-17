"""Flat Template MPC Demo

Compares original TinyMPC (fixed hover linearization) with the
Flat Template MPC (alpha-parameterized analytic linearization) on:
  1. Hover stabilization
  2. Figure-8 trajectory tracking

Usage:
    python examples/flat_template/demo.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from src.flat_linearization import (
    get_quad_params, flat_equilibrium, linearise, quat_to_euler_state,
)
from src.flat_template import FlatTemplateMPC
from utils.reference_trajectories import Figure8Reference


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def compute_alpha_figure8(t, A, w):
    """Flat parameter alpha from figure-8 trajectory acceleration."""
    if t < 1.0:
        return np.array([0.0, 0.0, 0.0, 0.0])   # startup: hover
    ax = -A * w**2 * np.sin(w * t)
    az = -2 * A * w**2 * np.sin(2 * w * t)
    return np.array([ax, 0.0, az, 0.0])


def simulate_tinympc(x0, mpc, quad, NSIM, dt, trajectory=None):
    """Simulate with original TinyMPC (fixed hover linearization)."""
    hover = trajectory is None
    x_curr = np.copy(x0)
    x_all, u_all, iters_all = [], [], []
    uhover = np.array(quad.hover_thrust, dtype=np.float64)
    qg = np.array([1.0, 0, 0, 0])
    t = 0.0

    for _ in range(NSIM):
        q = x_curr[3:7] / np.linalg.norm(x_curr[3:7])
        phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(qg).T @ q)

        if hover:
            delta_x = np.hstack([x_curr[0:3], phi, x_curr[7:10], x_curr[10:13]])
        else:
            x_ref = trajectory.generate_reference(t)
            delta_x = np.hstack([
                x_curr[0:3] - x_ref[0:3], phi,
                x_curr[7:10] - x_ref[6:9], x_curr[10:13],
            ])

        x_init = np.copy(mpc.x_prev);  x_init[:, 0] = delta_x
        u_init = np.copy(mpc.u_prev)
        x_out, u_out, status, k = mpc.solve_admm(x_init, u_init)

        u = uhover + u_out[:, 0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt

        x_all.append(np.copy(x_curr))
        u_all.append(np.copy(u))
        iters_all.append(k)

    return np.array(x_all), np.array(u_all), iters_all


def simulate_flat_template(x0, flat_mpc, quad, quad_params, NSIM, dt,
                           trajectory=None, A_traj=0.5, w_traj=1.0):
    """Simulate with Flat Template MPC (alpha-specific linearization)."""
    hover = trajectory is None
    x_curr = np.copy(x0)
    x_all, u_all, iters_all = [], [], []
    t = 0.0

    for _ in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))

        if hover:
            alpha   = np.array([0.0, 0.0, 0.0, 0.0])
            ref_pos = np.zeros(3)
            ref_vel = np.zeros(3)
        else:
            alpha   = compute_alpha_figure8(t, A_traj, w_traj)
            x_ref   = trajectory.generate_reference(t)
            ref_pos = x_ref[0:3]
            ref_vel = x_ref[6:9]

        x_eq, u_eq = flat_equilibrium(alpha, quad_params)

        delta_x = np.zeros(12)
        delta_x[0:3]  = x_euler[0:3]  - ref_pos
        delta_x[3:6]  = x_euler[3:6]  - x_eq[3:6]   # attitude error
        delta_x[6:9]  = x_euler[6:9]  - ref_vel
        delta_x[9:12] = x_euler[9:12]

        x_out, u_out, status, k = flat_mpc.solve(delta_x, alpha)

        u = u_eq + u_out[:, 0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt

        x_all.append(np.copy(x_curr))
        u_all.append(np.copy(u))
        iters_all.append(k)

    return np.array(x_all), np.array(u_all), iters_all


def tracking_errors(x_all, trajectory, dt, hover=True):
    """Position tracking error at each timestep."""
    errs = []
    for i, x in enumerate(x_all):
        ref = np.zeros(3) if hover else trajectory.generate_reference(i * dt)[0:3]
        errs.append(np.linalg.norm(x[0:3] - ref))
    return np.array(errs)


def print_table(label, err_a, iters_a, err_b, iters_b, name_a="TinyMPC", name_b="FlatTemplate"):
    """Pretty-print comparison table."""
    print(f"\n  {label}")
    print(f"  {'Metric':<30} {name_a:>12} {name_b:>12}")
    print(f"  {'-'*56}")
    print(f"  {'Avg position error (m)':<30} {np.mean(err_a):>12.4f} {np.mean(err_b):>12.4f}")
    print(f"  {'Max position error (m)':<30} {np.max(err_a):>12.4f} {np.max(err_b):>12.4f}")
    print(f"  {'Final position error (m)':<30} {err_a[-1]:>12.4f} {err_b[-1]:>12.4f}")
    print(f"  {'Avg ADMM iters':<30} {np.mean(iters_a):>12.1f} {np.mean(iters_b):>12.1f}")
    print(f"  {'Total ADMM iters':<30} {sum(iters_a):>12d} {sum(iters_b):>12d}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    print("=" * 60)
    print("  Flat Template MPC Demo")
    print("=" * 60)

    quad = QuadrotorDynamics()
    qp   = get_quad_params(quad)
    dt   = quad.dt

    # ----- cost matrices -----
    max_dev_x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])
    Q = np.diag(1.0 / max_dev_x**2)
    R = np.diag(1.0 / max_dev_u**2)

    xg = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    ug = np.array(quad.hover_thrust, dtype=np.float64)

    # ===== HOVER =====
    print("\n--- Hover Stabilization ---")
    x0 = np.copy(xg);  x0[0:3] = [0.2, 0.2, -0.2]
    N_h, rho_h, NSIM_h = 10, 85.0, 100

    # Original TinyMPC
    A_lin, B_lin = quad.get_linearized_dynamics(xg, ug)
    mpc = TinyMPC(A_lin, B_lin, Q, R, N_h, rho=rho_h, mode='hover')
    mpc.set_bounds([1.0 - ug[0]] * 4, [-ug[0]] * 4, [1000.] * 12, [-1000.] * 12)

    print("  Running TinyMPC ...")
    x_t, u_t, it_t = simulate_tinympc(x0, mpc, quad, NSIM_h, dt)

    # Flat Template
    flat_h = FlatTemplateMPC(qp, Q, R, N_h, rho_h,
                             grid_alphas=[np.zeros(4)])
    _, u_eq_h = flat_equilibrium(np.zeros(4), qp)
    flat_h.set_bounds([1.0 - u_eq_h[0]] * 4, [-u_eq_h[0]] * 4,
                      [1000.] * 12, [-1000.] * 12)
    flat_h.set_tols_iters(max_iter=500, abs_pri_tol=1e-2, abs_dua_tol=1e-2)

    print("  Running Flat Template MPC ...")
    x_f, u_f, it_f = simulate_flat_template(x0, flat_h, quad, qp, NSIM_h, dt)

    err_t = tracking_errors(x_t, None, dt, hover=True)
    err_f = tracking_errors(x_f, None, dt, hover=True)
    print_table("Hover comparison", err_t, it_t, err_f, it_f)

    # ===== FIGURE-8 =====
    print("\n--- Figure-8 Trajectory Tracking ---")
    amplitude, w = 0.5, 2 * np.pi / 3.7
    trajectory = Figure8Reference(A=amplitude, w=w, segment_type='full')

    max_dev_x_t = np.array([0.01, 0.01, 0.01, 0.5, 0.5, 0.05,
                             0.5, 0.5, 0.5, 0.7, 0.7, 0.5])
    max_dev_u_t = np.array([0.1, 0.1, 0.1, 0.1])
    Qt = np.diag(1.0 / max_dev_x_t**2)
    Rt = np.diag(1.0 / max_dev_u_t**2)
    N_t, rho_t, NSIM_t = 15, 5.0, 200

    x0t = np.copy(xg)
    xr0 = trajectory.generate_reference(0.0)
    x0t[0:3] = xr0[0:3];  x0t[7:10] = xr0[6:9]

    # Original TinyMPC (fixed hover linearization)
    mpc_t = TinyMPC(A_lin, B_lin, Qt, Rt, N_t, rho=rho_t, mode='traj')
    mpc_t.set_bounds([0.3]*4, [-0.3]*4,
                     [2.]*3+[1.]*3+[2.]*3+[2.]*3,
                     [-2.]*3+[-1.]*3+[-2.]*3+[-2.]*3)

    print("  Running TinyMPC (traj) ...")
    xt_t, ut_t, itt_t = simulate_tinympc(x0t, mpc_t, quad, NSIM_t, dt,
                                          trajectory=trajectory)

    # Flat Template grid from trajectory
    t_grid = np.linspace(0, 2 * np.pi / w, 20)
    grid_a = [compute_alpha_figure8(t, amplitude, w) for t in t_grid]
    grid_a.append(np.zeros(4))

    flat_t = FlatTemplateMPC(qp, Qt, Rt, N_t, rho_t, grid_alphas=grid_a)
    flat_t.set_bounds([0.3]*4, [-0.3]*4,
                      [2.]*3+[1.]*3+[2.]*3+[2.]*3,
                      [-2.]*3+[-1.]*3+[-2.]*3+[-2.]*3)

    print("  Running Flat Template MPC (traj) ...")
    xt_f, ut_f, itt_f = simulate_flat_template(
        x0t, flat_t, quad, qp, NSIM_t, dt,
        trajectory=trajectory, A_traj=amplitude, w_traj=w)

    errt_t = tracking_errors(xt_t, trajectory, dt, hover=False)
    errt_f = tracking_errors(xt_f, trajectory, dt, hover=False)
    print_table("Figure-8 comparison", errt_t, itt_t, errt_f, itt_f)

    # ----- save metrics -----
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(data_dir / 'flat_template_hover_errors.txt',
               np.column_stack([err_t, err_f]),
               header='TinyMPC FlatTemplate')
    np.savetxt(data_dir / 'flat_template_traj_errors.txt',
               np.column_stack([errt_t, errt_f]),
               header='TinyMPC FlatTemplate')
    print(f"\n  Metrics saved to {data_dir}/")
    print("  Done.")


if __name__ == '__main__':
    main()
