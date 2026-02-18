# src/barrier_template.py
"""Barrier-Augmented Template Controller.

Zero-iteration control law with log-barrier constraint awareness.
~260 arithmetic ops per control output -- same datapath as AnalyticTemplate,
but the DARE cache incorporates barrier Hessians for both input (R_mu)
and state (Q_mu) constraints.

Offline:
    Precompute K0_mu and dK_mu/d(alpha_i) via central differences at hover,
    using barrier-augmented cost matrices R_mu = R + H_{u,mu} and
    Q_mu = Q_lqr + H_{x,mu} + rho*I.
    Storage: 5 x (4x12) = 240 floats.

Online:
    K_mu(alpha) = K0_mu + sum_i  d_alpha_i * dK_mu_i   (192 MACs)
    u           = u*(alpha) - K_mu(alpha) (x - x*(alpha))  (48 MACs)
    Total: ~260 arithmetic operations.
"""

import numpy as np
from src.flat_linearization import linearise, flat_equilibrium


class BarrierTemplate:
    def __init__(self, quad_params, Q, R, rho=5.0, mu=1e-3,
                 u_min=None, u_max=None, x_min=None, x_max=None):
        """
        Args:
            quad_params: dict from get_quad_params()
            Q: (12,12) state cost
            R: (4,4)   input cost
            rho: ADMM penalty parameter (for DARE consistency with MPC)
            mu: barrier parameter (larger = more conservative near bounds)
            u_min: (4,) input lower bounds (absolute motor thrust)
            u_max: (4,) input upper bounds (absolute motor thrust)
            x_min: (12,) state lower bounds (absolute, for constrained dims)
            x_max: (12,) state upper bounds (absolute, for constrained dims)
        """
        self.quad_params = quad_params
        self.Q   = np.array(Q, dtype=np.float64)
        self.R   = np.array(R, dtype=np.float64)
        self.rho = float(rho)
        self.mu  = float(mu)
        self.nx  = 12
        self.nu  = 4

        # Constraint bounds
        self.u_min = np.array(u_min, dtype=np.float64) if u_min is not None else None
        self.u_max = np.array(u_max, dtype=np.float64) if u_max is not None else None
        self.x_min = np.array(x_min, dtype=np.float64) if x_min is not None else None
        self.x_max = np.array(x_max, dtype=np.float64) if x_max is not None else None

        # Hover equilibrium
        self.alpha0 = np.array([0.0, 0.0, 0.0, 0.0])

        # Gains (populated by precompute)
        self.K0 = None            # (4, 12) base gain at hover
        self.dK = [None] * 4     # dK/d(alpha_i), each (4, 12)

    # ------------------------------------------------------------------ #
    #  Barrier Hessians
    # ------------------------------------------------------------------ #

    def _barrier_hessian_u(self, u_eq):
        """Input barrier Hessian: H_{u,mu} = mu * diag(1/(u-u_min)^2 + 1/(u_max-u)^2)."""
        if self.u_min is None or self.u_max is None:
            return np.zeros((self.nu, self.nu))
        d_lo = np.maximum(u_eq - self.u_min, 1e-6)
        d_hi = np.maximum(self.u_max - u_eq, 1e-6)
        diag = self.mu * (1.0 / d_lo**2 + 1.0 / d_hi**2)
        return np.diag(diag)

    def _barrier_hessian_x(self, x_eq):
        """State barrier Hessian: H_{x,mu} for constrained state dimensions.

        Only applies barrier to dimensions where x_min/x_max are finite.
        For position constraints |pos| <= L in error coords (x_eq_pos = 0),
        the barrier depends on the distance from the equilibrium (= reference)
        position to the constraint boundary.
        """
        if self.x_min is None or self.x_max is None:
            return np.zeros((self.nx, self.nx))
        H = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            if np.isfinite(self.x_min[i]) and np.isfinite(self.x_max[i]):
                d_lo = max(x_eq[i] - self.x_min[i], 1e-6)
                d_hi = max(self.x_max[i] - x_eq[i], 1e-6)
                H[i, i] = self.mu * (1.0 / d_lo**2 + 1.0 / d_hi**2)
        return H

    # ------------------------------------------------------------------ #
    #  Barrier-augmented DARE
    # ------------------------------------------------------------------ #

    def _solve_barrier_dare_gain(self, A, B, alpha):
        """Solve barrier-augmented DARE and return the gain K_mu.

        R_mu = R + H_{u,mu}(alpha)
        Q_mu = Q_lqr + H_{x,mu}(alpha)
        Then solve augmented DARE with Q_mu + rho*I and R_mu + rho*I.
        """
        # Equilibrium for barrier Hessian computation
        x_eq, u_eq = flat_equilibrium(alpha, self.quad_params)

        # Barrier Hessians
        H_u = self._barrier_hessian_u(u_eq)
        H_x = self._barrier_hessian_x(x_eq)

        R_mu = self.R + H_u

        # DLQR terminal cost (with barrier-augmented R)
        P = np.copy(self.Q)
        for _ in range(500):
            K = np.linalg.solve(
                R_mu + B.T @ P @ B + 1e-8 * np.eye(self.nu),
                B.T @ P @ A,
            )
            P = self.Q + A.T @ P @ (A - B @ K)
        P_lqr = P

        # Barrier-augmented state cost
        Q_mu = P_lqr + H_x

        # Augmented DARE (TinyMPC convention: + rho*I)
        Q_aug = Q_mu + self.rho * np.eye(self.nx)
        R_aug = R_mu + self.rho * np.eye(self.nu)

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
    #  Offline  (Algorithm 2, barrier variant)
    # ------------------------------------------------------------------ #

    def precompute(self):
        """Offline: compute K0_mu and gain sensitivities dK_mu/d(alpha_i).

        1. Linearise at hover  ->  solve barrier-augmented DARE  ->  K0_mu
        2. For i = 1..4, central difference:
               dK_mu_i = (K_mu(alpha0 + eps*e_i) - K_mu(alpha0 - eps*e_i)) / (2*eps)
        """
        alpha0 = self.alpha0

        # Base gain at hover
        A0, B0 = linearise(alpha0, self.quad_params)
        self.K0 = self._solve_barrier_dare_gain(A0, B0, alpha0)

        # Central differences
        eps = 1e-3
        for i in range(4):
            ap = alpha0.copy();  ap[i] += eps
            am = alpha0.copy();  am[i] -= eps

            Ap, Bp = linearise(ap, self.quad_params)
            Am, Bm = linearise(am, self.quad_params)

            Kp = self._solve_barrier_dare_gain(Ap, Bp, ap)
            Km = self._solve_barrier_dare_gain(Am, Bm, am)

            self.dK[i] = (Kp - Km) / (2.0 * eps)

        print(f"Barrier template precomputed (mu={self.mu}): K0 {self.K0.shape}, "
              f"4 gain derivatives, {5 * self.nu * self.nx} floats total")

    # ------------------------------------------------------------------ #
    #  Online   (Algorithm 4, barrier variant)
    # ------------------------------------------------------------------ #

    def control(self, x, alpha, u_min=None, u_max=None):
        """Online control law (~260 ops).

        Args:
            x: (12,) error state relative to trajectory reference
            alpha: (4,) current flat parameter
            u_min: (4,) absolute motor thrust lower bound (for clipping)
            u_max: (4,) absolute motor thrust upper bound (for clipping)

        Returns:
            u: (4,) motor thrusts (absolute, clipped if bounds provided)
        """
        dalpha = alpha - self.alpha0

        # Equilibrium from flat parameter
        x_eq, u_eq = flat_equilibrium(alpha, self.quad_params)

        # K_mu(alpha) = K0_mu + sum_i dalpha_i * dK_mu_i   (192 MACs)
        K = self.K0.copy()
        for i in range(4):
            K += dalpha[i] * self.dK[i]

        # u = u*(alpha) - K_mu(alpha) * (x - x*(alpha))   (48 MACs)
        u = u_eq - K @ (x - x_eq)

        # Box constraints (clip)
        if u_min is not None and u_max is not None:
            u = np.clip(u, u_min, u_max)

        return u
