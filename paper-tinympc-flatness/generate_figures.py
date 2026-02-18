"""Generate paper figures for the write-up.

Produces:
  1. fig_hierarchy.pdf   -- block diagram of the 4-variant hierarchy
  2. fig_hover.pdf       -- hover tracking error over time (4 variants)
  3. fig_traj.pdf        -- figure-8 tracking error over time (4 variants)
  4. fig_traj_xy.pdf     -- figure-8 3D trajectories (x vs z)

Usage:
    python write_up/generate_figures.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUT = Path(__file__).parent / 'figures'
OUT.mkdir(exist_ok=True)

COLORS = {
    'TinyMPC':   '#1f77b4',
    'Online':    '#ff7f0e',
    'Flat':      '#2ca02c',
    'Analytic':  '#d62728',
}


# ================================================================
# Figure 1: Hierarchy / pipeline diagram
# ================================================================
def fig_hierarchy():
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')

    def box(x, y, w, h, text, color='#e8e8e8', textcolor='black', fontsize=8):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, weight='bold')

    def arrow(x1, y1, x2, y2, label='', above=True):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#444444', lw=1.2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            offset = 0.15 if above else -0.15
            ax.text(mx, my + offset, label, ha='center', va='center',
                    fontsize=7, style='italic', color='#444444')

    # Top row: the pipeline
    box(0, 2.2, 1.6, 0.8, r'$\mathbf{y}^*(t)$' + '\nTrajectory', '#dbeafe')
    box(2.2, 2.2, 1.8, 0.8, r'Flat Map $\Phi$' + '\n' + r'$\alpha \to (x^*, u^*)$', '#fef3c7')
    box(4.6, 2.2, 2.0, 0.8, 'Linearise' + '\n' + r'$\alpha \to (A_d, B_d)$', '#fef3c7')
    box(7.2, 2.2, 1.6, 0.8, 'DARE\n' + r'$\to K_\infty$', '#fde8e8')

    arrow(1.6, 2.6, 2.2, 2.6, r'$\alpha$')
    arrow(4.0, 2.6, 4.6, 2.6)
    arrow(6.6, 2.6, 7.2, 2.6)

    # Bottom row: the 4 variants
    y0 = 0.0
    box(0, y0, 2.0, 0.8, 'Online Relin\n' + r'$O(n^3) + K$ ADMM', '#ffe0cc')
    box(2.4, y0, 2.2, 0.8, 'Flat Template\nGrid + $K$ ADMM', '#d4edda')
    box(5.0, y0, 2.2, 0.8, 'Analytic Template\n' + r'$K_0 + \sum \delta\alpha_i \partial_i K$', '#f8d7da')
    box(7.6, y0, 2.2, 0.8, 'TinyMPC\nFixed + $K$ ADMM', '#cce5ff')

    # Arrows from pipeline to variants
    arrow(3.0, 2.2, 1.0, 0.8, 'per step', above=True)
    arrow(5.6, 2.2, 3.5, 0.8, 'offline grid', above=True)
    arrow(8.0, 2.2, 6.1, 0.8, r'$\partial K/\partial\alpha$', above=True)

    # Label: "decreasing compute"
    ax.annotate('', xy=(7.6, -0.35), xytext=(2.4, -0.35),
                arrowprops=dict(arrowstyle='<->', color='#888888', lw=1.0))
    ax.text(5.0, -0.5, 'decreasing online computation',
            ha='center', fontsize=7, color='#888888')

    fig.savefig(OUT / 'fig_hierarchy.pdf')
    plt.close(fig)
    print(f"  Saved {OUT / 'fig_hierarchy.pdf'}")


# ================================================================
# Simulation helpers (reuse from demos)
# ================================================================
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from src.flat_linearization import (
    get_quad_params, flat_equilibrium, linearise, quat_to_euler_state,
)
from src.flat_template import FlatTemplateMPC
from src.analytic_template import AnalyticTemplate
from utils.reference_trajectories import Figure8Reference


def run_hover_simulations():
    """Run all 4 variants on hover, return position error arrays."""
    quad = QuadrotorDynamics()
    qp = get_quad_params(quad)
    dt = quad.dt

    max_dev_x = np.array([0.1,0.1,0.1, 0.5,0.5,0.05, 0.5,0.5,0.5, 0.7,0.7,0.2])
    max_dev_u = np.array([0.5,0.5,0.5,0.5])
    Q = np.diag(1.0 / max_dev_x**2)
    R = np.diag(1.0 / max_dev_u**2)

    xg = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0], dtype=np.float64)
    ug = np.array(quad.hover_thrust, dtype=np.float64)
    x0 = np.copy(xg); x0[0:3] = [0.2, 0.2, -0.2]
    N, rho, NSIM = 10, 85.0, 100

    results = {}

    # --- TinyMPC ---
    A_lin, B_lin = quad.get_linearized_dynamics(xg, ug)
    mpc = TinyMPC(A_lin, B_lin, Q, R, N, rho=rho, mode='hover')
    mpc.set_bounds([1.0-ug[0]]*4, [-ug[0]]*4, [1000.]*12, [-1000.]*12)
    x_curr = np.copy(x0)
    qg = np.array([1.0,0,0,0])
    errs = []
    for _ in range(NSIM):
        q = x_curr[3:7]/np.linalg.norm(x_curr[3:7])
        phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(qg).T @ q)
        dx = np.hstack([x_curr[0:3], phi, x_curr[7:10], x_curr[10:13]])
        xi = np.copy(mpc.x_prev); xi[:,0] = dx
        ui = np.copy(mpc.u_prev)
        xo, uo, st, k = mpc.solve_admm(xi, ui)
        u = ug + uo[:,0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        errs.append(np.linalg.norm(x_curr[0:3]))
    results['TinyMPC'] = np.array(errs)

    # --- Online Relin ---
    x_curr = np.copy(x0)
    errs = []
    prev_x = None; prev_u = None
    for _ in range(NSIM):
        q = x_curr[3:7]/np.linalg.norm(x_curr[3:7])
        phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(qg).T @ q)
        dx = np.hstack([x_curr[0:3], phi, x_curr[7:10], x_curr[10:13]])
        Al, Bl = quad.get_linearized_dynamics(xg, ug)
        mpc_r = TinyMPC(Al, Bl, Q, R, N, rho=rho, mode='hover')
        mpc_r.set_bounds([1.0-ug[0]]*4, [-ug[0]]*4, [1000.]*12, [-1000.]*12)
        if prev_x is not None:
            mpc_r.x_prev = prev_x; mpc_r.u_prev = prev_u
        xi = np.copy(mpc_r.x_prev); xi[:,0] = dx
        ui = np.copy(mpc_r.u_prev)
        xo, uo, st, k = mpc_r.solve_admm(xi, ui)
        prev_x = mpc_r.x_prev; prev_u = mpc_r.u_prev
        u = ug + uo[:,0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        errs.append(np.linalg.norm(x_curr[0:3]))
    results['Online'] = np.array(errs)

    # --- Flat Template ---
    flat_h = FlatTemplateMPC(qp, Q, R, N, rho, grid_alphas=[np.zeros(4)])
    _, u_eq = flat_equilibrium(np.zeros(4), qp)
    flat_h.set_bounds([1.0-u_eq[0]]*4, [-u_eq[0]]*4, [1000.]*12, [-1000.]*12)
    flat_h.set_tols_iters(max_iter=500, abs_pri_tol=1e-2, abs_dua_tol=1e-2)
    x_curr = np.copy(x0)
    errs = []
    for _ in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        alpha = np.zeros(4)
        x_eq, u_eq = flat_equilibrium(alpha, qp)
        dx = np.zeros(12)
        dx[0:3] = x_euler[0:3]; dx[3:6] = x_euler[3:6] - x_eq[3:6]
        dx[6:9] = x_euler[6:9]; dx[9:12] = x_euler[9:12]
        xo, uo, st, k = flat_h.solve(dx, alpha)
        u = u_eq + uo[:,0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        errs.append(np.linalg.norm(x_curr[0:3]))
    results['Flat'] = np.array(errs)

    # --- Analytic ---
    at = AnalyticTemplate(qp, Q, R, rho=rho)
    at.precompute()
    x_curr = np.copy(x0)
    errs = []
    for _ in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        alpha = np.zeros(4)
        u = at.control(x_euler, alpha)
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        errs.append(np.linalg.norm(x_curr[0:3]))
    results['Analytic'] = np.array(errs)

    return results, dt


def compute_alpha_figure8(t, A, w):
    if t < 1.0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    ax = -A * w**2 * np.sin(w * t)
    az = -2 * A * w**2 * np.sin(2 * w * t)
    return np.array([ax, 0.0, az, 0.0])


def run_traj_simulations():
    """Run all 4 variants on figure-8, return error arrays + trajectories."""
    quad = QuadrotorDynamics()
    qp = get_quad_params(quad)
    dt = quad.dt

    amplitude, w = 0.5, 2 * np.pi / 3.7
    trajectory = Figure8Reference(A=amplitude, w=w, segment_type='full')

    max_dev_x = np.array([0.01,0.01,0.01, 0.5,0.5,0.05, 0.5,0.5,0.5, 0.7,0.7,0.5])
    max_dev_u = np.array([0.1,0.1,0.1,0.1])
    Qt = np.diag(1.0 / max_dev_x**2)
    Rt = np.diag(1.0 / max_dev_u**2)
    N, rho, NSIM = 15, 5.0, 200

    xg = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0], dtype=np.float64)
    ug = np.array(quad.hover_thrust, dtype=np.float64)
    qg = np.array([1.0,0,0,0])
    xr0 = trajectory.generate_reference(0.0)
    x0 = np.copy(xg); x0[0:3] = xr0[0:3]; x0[7:10] = xr0[6:9]

    A_lin, B_lin = quad.get_linearized_dynamics(xg, ug)

    results = {}
    trajectories = {}

    # --- TinyMPC ---
    mpc_t = TinyMPC(A_lin, B_lin, Qt, Rt, N, rho=rho, mode='traj')
    mpc_t.set_bounds([0.3]*4, [-0.3]*4,
                     [2.]*3+[1.]*3+[2.]*3+[2.]*3,
                     [-2.]*3+[-1.]*3+[-2.]*3+[-2.]*3)
    x_curr = np.copy(x0)
    errs = []; traj = []; t = 0.0
    for _ in range(NSIM):
        q = x_curr[3:7]/np.linalg.norm(x_curr[3:7])
        phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(qg).T @ q)
        x_ref = trajectory.generate_reference(t)
        dx = np.hstack([x_curr[0:3]-x_ref[0:3], phi, x_curr[7:10]-x_ref[6:9], x_curr[10:13]])
        xi = np.copy(mpc_t.x_prev); xi[:,0] = dx
        ui = np.copy(mpc_t.u_prev)
        xo, uo, st, k = mpc_t.solve_admm(xi, ui)
        u = ug + uo[:,0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        errs.append(np.linalg.norm(x_curr[0:3] - trajectory.generate_reference(t)[0:3]))
        traj.append(x_curr[0:3].copy())
    results['TinyMPC'] = np.array(errs)
    trajectories['TinyMPC'] = np.array(traj)

    # --- Flat Template ---
    t_grid = np.linspace(0, 2*np.pi/w, 20)
    grid_a = [compute_alpha_figure8(tg, amplitude, w) for tg in t_grid]
    grid_a.append(np.zeros(4))
    flat_t = FlatTemplateMPC(qp, Qt, Rt, N, rho, grid_alphas=grid_a)
    flat_t.set_bounds([0.3]*4, [-0.3]*4,
                      [2.]*3+[1.]*3+[2.]*3+[2.]*3,
                      [-2.]*3+[-1.]*3+[-2.]*3+[-2.]*3)
    x_curr = np.copy(x0)
    errs = []; traj = []; t = 0.0
    for _ in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        alpha = compute_alpha_figure8(t, amplitude, w)
        x_ref = trajectory.generate_reference(t)
        x_eq, u_eq = flat_equilibrium(alpha, qp)
        dx = np.zeros(12)
        dx[0:3] = x_euler[0:3] - x_ref[0:3]
        dx[3:6] = x_euler[3:6] - x_eq[3:6]
        dx[6:9] = x_euler[6:9] - x_ref[6:9]
        dx[9:12] = x_euler[9:12]
        xo, uo, st, k = flat_t.solve(dx, alpha)
        u = u_eq + uo[:,0]
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        errs.append(np.linalg.norm(x_curr[0:3] - trajectory.generate_reference(t)[0:3]))
        traj.append(x_curr[0:3].copy())
    results['Flat'] = np.array(errs)
    trajectories['Flat'] = np.array(traj)

    # --- Analytic ---
    at = AnalyticTemplate(qp, Qt, Rt, rho=rho)
    at.precompute()
    x_curr = np.copy(x0)
    errs = []; traj = []; t = 0.0
    for _ in range(NSIM):
        x_euler = quat_to_euler_state(np.array(x_curr, dtype=np.float64))
        alpha = compute_alpha_figure8(t, amplitude, w)
        x_ref = trajectory.generate_reference(t)
        x_eq, u_eq = flat_equilibrium(alpha, qp)
        x_ctrl = np.copy(x_euler)
        x_ctrl[0:3] -= x_ref[0:3]
        x_ctrl[6:9] -= x_ref[6:9]
        u = at.control(x_ctrl, alpha)
        x_curr = np.array(quad.dynamics_rk4(x_curr, u, dt=dt), dtype=np.float64)
        t += dt
        errs.append(np.linalg.norm(x_curr[0:3] - trajectory.generate_reference(t)[0:3]))
        traj.append(x_curr[0:3].copy())
    results['Analytic'] = np.array(errs)
    trajectories['Analytic'] = np.array(traj)

    # Reference trajectory
    ref_traj = []
    for i in range(NSIM):
        ref_traj.append(trajectory.generate_reference((i+1)*dt)[0:3])
    trajectories['Reference'] = np.array(ref_traj)

    return results, trajectories, dt


# ================================================================
# Figure 2: Hover error over time
# ================================================================
def fig_hover(results, dt):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    t = np.arange(1, len(results['TinyMPC'])+1) * dt
    for name, c in COLORS.items():
        key = name if name in results else name.split()[0]
        if key in results:
            ax.plot(t, results[key], color=c, label=name, linewidth=1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([0, t[-1]])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / 'fig_hover.pdf')
    plt.close(fig)
    print(f"  Saved {OUT / 'fig_hover.pdf'}")


# ================================================================
# Figure 3: Trajectory error over time
# ================================================================
def fig_traj_error(results, dt):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    t = np.arange(1, len(results['TinyMPC'])+1) * dt
    for name, c in COLORS.items():
        key = name if name in results else name.split()[0]
        if key in results:
            ax.plot(t, results[key], color=c, label=name, linewidth=1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([0, t[-1]])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / 'fig_traj_error.pdf')
    plt.close(fig)
    print(f"  Saved {OUT / 'fig_traj_error.pdf'}")


# ================================================================
# Figure 4: Trajectory XZ view
# ================================================================
def fig_traj_xz(trajectories):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ref = trajectories['Reference']
    ax.plot(ref[:,0], ref[:,2], 'k--', linewidth=1.0, label='Reference', alpha=0.5)
    for name, c in [('TinyMPC', COLORS['TinyMPC']),
                     ('Flat', COLORS['Flat']),
                     ('Analytic', COLORS['Analytic'])]:
        if name in trajectories:
            tr = trajectories[name]
            ax.plot(tr[:,0], tr[:,2], color=c, linewidth=1.0, label=name)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / 'fig_traj_xz.pdf')
    plt.close(fig)
    print(f"  Saved {OUT / 'fig_traj_xz.pdf'}")


# ================================================================
if __name__ == '__main__':
    print("Generating paper figures...")

    print("\n[1/4] Hierarchy diagram")
    fig_hierarchy()

    print("\n[2/4] Hover simulations")
    hover_results, dt_h = run_hover_simulations()
    fig_hover(hover_results, dt_h)

    print("\n[3/4] Trajectory simulations")
    traj_results, traj_trajs, dt_t = run_traj_simulations()
    fig_traj_error(traj_results, dt_t)

    print("\n[4/4] Trajectory XZ view")
    fig_traj_xz(traj_trajs)

    print("\nAll figures saved to", OUT)
