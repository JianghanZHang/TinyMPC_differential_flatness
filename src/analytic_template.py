# src/analytic_template.py
"""Analytic Template Controller (Algorithms 2 + 4).

Zero-iteration control law using first-order gain approximation.
~260 arithmetic ops per control output -- FPGA-portable datapath.

Offline (Algorithm 2):
    Precompute K0 and dK/d(alpha_i) via central differences at hover.
    Storage: 5 x (4x12) = 240 floats.

Online (Algorithm 4):
    K(alpha) = K0 + sum_i  d_alpha_i * dK_i      (192 MACs)
    u        = u*(alpha) - K(alpha) (x - x*(alpha))  (48 MACs)
    Total: ~260 arithmetic operations.
"""

import numpy as np
from src.flat_linearization import linearise, flat_equilibrium


class AnalyticTemplate:
    def __init__(self, quad_params, Q, R, rho=5.0):
        """
        Args:
            quad_params: dict from get_quad_params()
            Q: (12,12) state cost
            R: (4,4)   input cost
            rho: ADMM penalty parameter (for DARE consistency with MPC)
        """
        self.quad_params = quad_params
        self.Q   = np.array(Q, dtype=np.float64)
        self.R   = np.array(R, dtype=np.float64)
        self.rho = float(rho)
        self.nx  = 12
        self.nu  = 4

        # Hover equilibrium: alpha_0 = (0, 0, 0, 0)
        self.alpha0 = np.array([0.0, 0.0, 0.0, 0.0])

        # Gains (populated by precompute)
        self.K0 = None            # (4, 12) base gain at hover
        self.dK = [None] * 4     # dK/d(alpha_i), each (4, 12)

    # ------------------------------------------------------------------ #
    #  Offline  (Algorithm 2)
    # ------------------------------------------------------------------ #

    def precompute(self):
        """Offline: compute K0 and gain sensitivities dK/d(alpha_i).

        1. Linearise at hover  ->  solve DARE  ->  K0
        2. For i = 1..4, central difference:
               dK_i = (K(alpha0 + eps*e_i) - K(alpha0 - eps*e_i)) / (2*eps)
        """
        alpha0 = self.alpha0

        # Base gain at hover
        A0, B0 = linearise(alpha0, self.quad_params)
        self.K0 = self._solve_dare_gain(A0, B0)

        # Central differences
        eps = 1e-3
        for i in range(4):
            ap = alpha0.copy();  ap[i] += eps
            am = alpha0.copy();  am[i] -= eps

            Ap, Bp = linearise(ap, self.quad_params)
            Am, Bm = linearise(am, self.quad_params)

            Kp = self._solve_dare_gain(Ap, Bp)
            Km = self._solve_dare_gain(Am, Bm)

            self.dK[i] = (Kp - Km) / (2.0 * eps)

        print(f"Analytic template precomputed: K0 {self.K0.shape}, "
              f"4 gain derivatives, {5 * self.nu * self.nx} floats total")

    def _solve_dare_gain(self, A, B):
        """Solve augmented DARE and return the infinite-horizon gain K.

        Matches FlatTemplateMPC._compute_dare_cache (same P_lqr + rho convention).
        """
        # DLQR terminal cost
        P = np.copy(self.Q)
        for _ in range(500):
            K = np.linalg.solve(
                self.R + B.T @ P @ B + 1e-8 * np.eye(self.nu),
                B.T @ P @ A,
            )
            P = self.Q + A.T @ P @ (A - B @ K)
        P_lqr = P

        # Augmented DARE (TinyMPC convention)
        Q_aug = P_lqr + self.rho * np.eye(self.nx)
        R_aug = self.R  + self.rho * np.eye(self.nu)

        Pinf = np.copy(Q_aug)
        Kinf = np.zeros((self.nu, self.nx))
        for _ in range(5000):
            Kinf_prev = Kinf.copy()
            Kinf = np.linalg.inv(R_aug + B.T @ Pinf @ B) @ (B.T @ Pinf @ A)
            Pinf = Q_aug + A.T @ Pinf @ (A - B @ Kinf)
            if np.linalg.norm(Kinf - Kinf_prev, 2) < 1e-10:
                break

        return Kinf

    # ------------------------------------------------------------------ #
    #  Online   (Algorithm 4)
    # ------------------------------------------------------------------ #

    def control(self, x, alpha, u_min=None, u_max=None):
        """Online control law (~260 ops).

        Args:
            x: (12,) state.
               Position and velocity should be *errors* relative to the
               trajectory reference.  Angles and omega are absolute.
               x = [pos - pos_ref, phi, theta, psi, vel - vel_ref, omega]
            alpha: (4,) current flat parameter [a_x, a_y, a_z, psi]
            u_min: (4,) absolute motor thrust lower bound
            u_max: (4,) absolute motor thrust upper bound

        Returns:
            u: (4,) motor thrusts (absolute, clipped if bounds provided)
        """
        # delta_alpha = alpha - alpha_0
        dalpha = alpha - self.alpha0

        # Equilibrium from flat parameter
        x_eq, u_eq = flat_equilibrium(alpha, self.quad_params)

        # K(alpha) = K0 + sum_i  dalpha_i * dK_i   (192 MACs)
        K = self.K0.copy()
        for i in range(4):
            K += dalpha[i] * self.dK[i]

        # u = u*(alpha) - K(alpha) * (x - x*(alpha))   (48 MACs)
        u = u_eq - K @ (x - x_eq)

        # Box constraints
        if u_min is not None and u_max is not None:
            u = np.clip(u, u_min, u_max)

        return u
