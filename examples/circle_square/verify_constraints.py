"""Step-by-step constraint verification.

Test 1: Hover at origin, initial velocity pushes drone right.
        Constrain |x| <= 0.15. Verify ADMM clamps the position.

Test 2: Track a reference at (0.1, 0, 0) from origin.
        Both controllers should reach x=0.1.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.flat_linearization import (
    get_quad_params, flat_equilibrium,
    quat_to_euler_state,
)
from src.flat_template import FlatTemplateMPC
from src.analytic_template import AnalyticTemplate


def test_hover_constraint():
    """Hover at origin with initial vx=0.5 m/s.
    Constrain |x| <= 0.15 m. Should clamp overshoot."""
    print("=" * 60)
    print("  Test 1: Hover + velocity perturbation + position constraint")
    print("=" * 60)

    quad = QuadrotorDynamics()
    qp = get_quad_params(quad)
    dt = quad.dt
    alpha = np.zeros(4)
    x_eq, u_eq = flat_equilibrium(alpha, qp)

    N, rho = 10, 500.0
    NSIM = 100
    L = 0.05

    Q = np.diag(1.0 / np.array([
        0.1, 0.1, 0.1, 0.5, 0.5, 0.05,
        0.5, 0.5, 0.5, 0.7, 0.7, 0.2])**2)
    R = np.diag(1.0 / np.array([0.5, 0.5, 0.5, 0.5])**2)

    # Start at origin with vx = 0.5 m/s
    xg = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0], dtype=np.float64)
    x0 = np.copy(xg)
    x0[7] = 0.5  # vx = 0.5 m/s

    # --- Constrained ---
    mpc = FlatTemplateMPC(qp, Q, R, N, rho, grid_alphas=[alpha])
    mpc.set_bounds(
        umax=[1.0 - u_eq[0]]*4, umin=[-u_eq[0]]*4,
        xmax=[L, 1000., 1000., 1., 1., 1., 2., 2., 2., 2., 2., 2.],
        xmin=[-L, -1000., -1000., -1., -1., -1., -2., -2., -2., -2., -2., -2.],
    )
    mpc.set_tols_iters(max_iter=500, abs_pri_tol=1e-3, abs_dua_tol=1e-3)

    x_curr = np.copy(x0)
    x_hist_c = [np.copy(x_curr)]

    for step in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        dx = np.zeros(12)
        dx[0:3]  = x_euler[0:3]  # error from origin
        dx[3:6]  = x_euler[3:6]  # attitude
        dx[6:9]  = x_euler[6:9]  # velocity
        dx[9:12] = x_euler[9:12]

        xo, uo, st, k = mpc.solve(dx, alpha)
        u = u_eq + uo[:, 0]

        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        x_hist_c.append(np.copy(x_curr))

        if step < 5 or step % 20 == 0:
            print(f"  t={step*dt:.2f}s: x={x_curr[0]:.4f}, "
                  f"vx={x_curr[7]:.4f}, iters={k}, st={st}")

    x_hist_c = np.array(x_hist_c)
    max_x = np.max(x_hist_c[:, 0])
    print(f"\n  Max x (constrained):  {max_x:.4f}  (limit: {L})")
    print(f"  Constraint {'SATISFIED' if max_x <= L + 0.01 else 'VIOLATED'}")

    # --- Unconstrained ---
    mpc_u = FlatTemplateMPC(qp, Q, R, N, rho, grid_alphas=[alpha])
    mpc_u.set_bounds(
        umax=[1.0 - u_eq[0]]*4, umin=[-u_eq[0]]*4,
        xmax=[1000.]*12, xmin=[-1000.]*12,
    )
    mpc_u.set_tols_iters(max_iter=500, abs_pri_tol=1e-3, abs_dua_tol=1e-3)

    x_curr = np.copy(x0)
    x_hist_u = [np.copy(x_curr)]

    for step in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        dx = np.zeros(12)
        dx[0:3]  = x_euler[0:3]
        dx[3:6]  = x_euler[3:6]
        dx[6:9]  = x_euler[6:9]
        dx[9:12] = x_euler[9:12]

        xo, uo, st, k = mpc_u.solve(dx, alpha)
        u = u_eq + uo[:, 0]

        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        x_hist_u.append(np.copy(x_curr))

    x_hist_u = np.array(x_hist_u)
    max_x_u = np.max(x_hist_u[:, 0])
    print(f"  Max x (unconstrained): {max_x_u:.4f}")
    print(f"  Overshoot difference:  {max_x - max_x_u:.4f} m")

    return x_hist_c, x_hist_u


def test_analytic_tracking():
    """Simple test: track (0.1, 0, 0) from origin.
    Verify Analytic Template tracks correctly."""
    print(f"\n{'=' * 60}")
    print("  Test 2: Analytic Template point tracking (0.1, 0, 0)")
    print("=" * 60)

    quad = QuadrotorDynamics()
    qp = get_quad_params(quad)
    dt = quad.dt

    Q = np.diag(1.0 / np.array([
        0.1, 0.1, 0.1, 0.5, 0.5, 0.05,
        0.5, 0.5, 0.5, 0.7, 0.7, 0.2])**2)
    R = np.diag(1.0 / np.array([0.5, 0.5, 0.5, 0.5])**2)
    rho = 85.0

    at = AnalyticTemplate(qp, Q, R, rho=rho)
    at.precompute()
    _, u_eq_h = flat_equilibrium(np.zeros(4), qp)

    target = np.array([0.1, 0.0, 0.0])
    xg = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0], dtype=np.float64)
    x_curr = np.copy(xg)

    for step in range(100):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        x_ctrl = np.zeros(12)
        x_ctrl[0:3]  = x_euler[0:3] - target
        x_ctrl[3:6]  = x_euler[3:6]
        x_ctrl[6:9]  = x_euler[6:9]
        x_ctrl[9:12] = x_euler[9:12]

        alpha = np.zeros(4)
        u = at.control(x_ctrl, alpha,
                       u_min=np.zeros(4), u_max=u_eq_h * 2)
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)

        if step < 5 or step % 20 == 0:
            print(f"  t={step*dt:.2f}s: pos=({x_curr[0]:.4f}, "
                  f"{x_curr[1]:.4f}, {x_curr[2]:.4f})")

    final_err = np.linalg.norm(x_curr[0:3] - target)
    print(f"\n  Final pos error: {final_err:.4f} m")
    print(f"  {'PASSED' if final_err < 0.05 else 'FAILED'}")


if __name__ == '__main__':
    test_hover_constraint()
    test_analytic_tracking()
