# src/flat_linearization.py
"""Analytic linearization via differential flatness (Algorithm 3).

Computes A(alpha), B(alpha) without autograd -- pure numpy trigonometry.
FPGA-portable: no symbolic differentiation, no autograd.

State ordering (12D): [p(3), angles(3), v(3), omega(3)]
  p       = [p_x, p_y, p_z]           position
  angles  = [phi, theta, psi]          roll, pitch, yaw (ZYX Euler)
  v       = [v_x, v_y, v_z]           velocity
  omega   = [omega_x, omega_y, omega_z] body angular velocity

Input (4D): [u1, u2, u3, u4] motor thrusts

alpha = (a_x, a_y, a_z, psi) -- flat parameter (commanded acceleration + yaw)
"""

import numpy as np


def get_quad_params(quad):
    """Extract parameters from QuadrotorDynamics into a plain dict.

    Converts autograd arrays to standard numpy for FPGA-compatible code.
    """
    return {
        'mass': float(quad.mass),
        'g':    float(quad.g),
        'kt':   float(quad.kt),
        'km':   float(quad.km),
        'el':   float(quad.el),
        'J':    np.array(quad.J, dtype=np.float64),
        'dt':   float(quad.dt),
    }


def flat_equilibrium(alpha, quad_params):
    """Compute equilibrium state and input from flat parameter alpha.

    Eqs. (6.2)-(6.4): given commanded acceleration and yaw, compute the
    required attitude and thrust.

    Args:
        alpha: (4,) array [a_x, a_y, a_z, psi]
        quad_params: dict from get_quad_params()

    Returns:
        x_eq: (12,) equilibrium state [p=0, angles, v=0, omega=0]
        u_eq: (4,)  equilibrium motor thrusts
    """
    ax, ay, az, psi = alpha
    m  = quad_params['mass']
    g  = quad_params['g']
    kt = quad_params['kt']

    # Total acceleration magnitude (gravity-inclusive)
    T_norm = np.sqrt(ax**2 + ay**2 + (az + g)**2)

    # Equilibrium attitude
    sp, cp = np.sin(psi), np.cos(psi)
    phi_eq   = np.arcsin(np.clip((ax * sp - ay * cp) / T_norm, -1.0, 1.0))
    theta_eq = np.arctan2(ax * cp + ay * sp, az + g)

    x_eq = np.zeros(12)
    x_eq[3] = phi_eq
    x_eq[4] = theta_eq
    x_eq[5] = psi

    # Equal thrust distribution across 4 motors
    T_star = m * T_norm
    u_eq = (T_star / (4.0 * kt)) * np.ones(4)

    return x_eq, u_eq


def linearise(alpha, quad_params):
    """Algorithm 3: analytic linearization at flat parameter alpha.

    Pure numpy -- no autograd, no symbolic differentiation.
    Discretized via forward Euler: A_d = I + A_c*dt,  B_d = B_c*dt.

    Args:
        alpha: (4,) array [a_x, a_y, a_z, psi]
        quad_params: dict from get_quad_params()

    Returns:
        A_d: (12, 12) discrete-time state matrix
        B_d: (12, 4)  discrete-time input matrix
    """
    ax, ay, az, psi = alpha
    m  = quad_params['mass']
    g  = quad_params['g']
    kt = quad_params['kt']
    km = quad_params['km']
    el = quad_params['el']
    J  = quad_params['J']
    dt = quad_params['dt']

    T_norm = np.sqrt(ax**2 + ay**2 + (az + g)**2)

    # Equilibrium attitude
    sp, cp = np.sin(psi), np.cos(psi)
    phi_eq   = np.arcsin(np.clip((ax * sp - ay * cp) / T_norm, -1.0, 1.0))
    theta_eq = np.arctan2(ax * cp + ay * sp, az + g)

    sf, cf = np.sin(phi_eq), np.cos(phi_eq)
    st, ct = np.sin(theta_eq), np.cos(theta_eq)

    # ----- Continuous-time A_c (12x12) -----
    A_c = np.zeros((12, 12))

    # dp/dt = v
    A_c[0:3, 6:9] = np.eye(3)

    # Gravity coupling  G(alpha) = dv_dot / d(phi, theta, psi)
    # v_dot = R(phi,theta,psi) @ [0, 0, T_norm]^T + [0, 0, -g]^T
    G = np.zeros((3, 3))
    G[0, 0] = T_norm * (-cp * st * sf + sp * cf)
    G[0, 1] = T_norm * ( cp * ct * cf)
    G[0, 2] = T_norm * (-sp * st * cf + cp * sf)
    G[1, 0] = T_norm * (-sp * st * sf - cp * cf)
    G[1, 1] = T_norm * ( sp * ct * cf)
    G[1, 2] = T_norm * ( cp * st * cf + sp * sf)
    G[2, 0] = T_norm * (-ct * sf)
    G[2, 1] = T_norm * (-st * cf)
    G[2, 2] = 0.0
    A_c[6:9, 3:6] = G

    # Euler-angle kinematics:  d(angles)/dt = W(phi,theta) @ omega
    sec_t = 1.0 / max(ct, 1e-12)
    tt    = st * sec_t
    W = np.array([
        [1.0, sf * tt,    cf * tt   ],
        [0.0, cf,        -sf        ],
        [0.0, sf * sec_t, cf * sec_t],
    ])
    A_c[3:6, 9:12] = W

    # omega_dot at equilibrium (omega*=0): gyroscopic terms vanish

    # ----- Continuous-time B_c (12x4) -----
    B_c = np.zeros((12, 4))

    # Thrust direction in world frame: R(phi*,theta*,psi*) @ [0,0,1]
    body_z = np.array([
        cp * st * cf + sp * sf,
        sp * st * cf - cp * sf,
        ct * cf,
    ])
    for i in range(4):
        B_c[6:9, i] = (kt / m) * body_z

    # Torque allocation: d(omega_dot)/du = J^{-1} @ M
    M = np.array([
        [-el * kt, -el * kt,  el * kt,  el * kt],
        [-el * kt,  el * kt,  el * kt, -el * kt],
        [-km,       km,       -km,       km     ],
    ])
    J_inv = np.linalg.inv(J)
    B_c[9:12, :] = J_inv @ M

    # ----- Forward-Euler discretization -----
    A_d = np.eye(12) + A_c * dt
    B_d = B_c * dt

    return A_d, B_d


# ---- State-conversion utilities ----

def quat_to_euler_state(x_quat):
    """Convert 13D quaternion state to 12D Euler-angle state.

    Args:
        x_quat: (13,) array [p(3), q(4), v(3), omega(3)]

    Returns:
        x_euler: (12,) array [p(3), angles(3), v(3), omega(3)]
    """
    p     = x_quat[0:3]
    q     = x_quat[3:7] / np.linalg.norm(x_quat[3:7])
    v     = x_quat[7:10]
    omega = x_quat[10:13]

    w, x, y, z = q
    phi   = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    theta = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    psi   = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

    return np.hstack([p, [phi, theta, psi], v, omega])


def euler_to_quat(phi, theta, psi):
    """Convert ZYX Euler angles to quaternion [w, x, y, z].

    Returns:
        q: (4,) quaternion in scalar-first convention
    """
    cy, sy = np.cos(psi / 2),   np.sin(psi / 2)
    cp, sp = np.cos(theta / 2), np.sin(theta / 2)
    cr, sr = np.cos(phi / 2),   np.sin(phi / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])
