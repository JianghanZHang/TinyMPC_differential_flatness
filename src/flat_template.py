# src/flat_template.py
"""Flat Template MPC (Algorithm 1).

alpha-parameterized cache family with ADMM iterations.

Offline: precompute DARE solutions at a grid of alpha values.
Online : lookup nearest cache, run ADMM with alpha-specific (A,B,K,P,C1,C2).

The ADMM solver mirrors TinyMPC.solve_admm to ensure identical behavior
when alpha = alpha_hover.
"""

import numpy as np
from src.flat_linearization import linearise, flat_equilibrium


class FlatTemplateMPC:
    def __init__(self, quad_params, Q, R, N, rho, grid_alphas=None):
        """
        Args:
            quad_params: dict from get_quad_params()
            Q: (12,12) state cost matrix
            R: (4,4)   input cost matrix
            N: horizon length
            rho: ADMM penalty parameter
            grid_alphas: list of alpha vectors for offline precomputation
        """
        self.quad_params = quad_params
        self.Q   = np.array(Q, dtype=np.float64)
        self.R   = np.array(R, dtype=np.float64)
        self.N   = N
        self.nx  = 12
        self.nu  = 4
        self.rho = float(rho)

        # Template cache: maps alpha_key -> ADMM matrices
        self.cache_grid  = {}
        self.grid_alphas = []

        if grid_alphas is not None:
            self.precompute_template(grid_alphas)

        # Warm-start variables
        self._reset_warm_start()

        # Default bounds (unconstrained)
        self.umin = -np.inf * np.ones(self.nu)
        self.umax =  np.inf * np.ones(self.nu)
        self.xmin = -np.inf * np.ones(self.nx)
        self.xmax =  np.inf * np.ones(self.nx)

        # Default tolerances / iteration cap
        self.max_iter    = 10
        self.abs_pri_tol = 1e-3
        self.abs_dua_tol = 1e-3

    # ------------------------------------------------------------------ #
    #  Offline pre-computation
    # ------------------------------------------------------------------ #

    def precompute_template(self, grid_alphas):
        """Offline: solve DARE at each alpha in the grid."""
        for alpha in grid_alphas:
            alpha = np.asarray(alpha, dtype=np.float64)
            A_d, B_d = linearise(alpha, self.quad_params)
            cache = self._compute_dare_cache(A_d, B_d)
            key = tuple(np.round(alpha, 6))
            self.cache_grid[key] = cache
            self.grid_alphas.append(alpha.copy())
        print(f"Precomputed {len(self.grid_alphas)} template cache entries")

    def _compute_dlqr(self, A, B, n_steps=500):
        """Discrete LQR solution (terminal cost weight)."""
        P = np.copy(self.Q)
        for _ in range(n_steps):
            K = np.linalg.solve(
                self.R + B.T @ P @ B + 1e-8 * np.eye(self.nu),
                B.T @ P @ A,
            )
            P = self.Q + A.T @ P @ (A - B @ K)
        return P

    def _compute_dare_cache(self, A, B):
        """Solve augmented DARE and build full ADMM cache.

        Mirrors TinyMPC.compute_cache_terms:
          Q_aug = P_lqr + rho*I    (stage cost = terminal cost + penalty)
          R_aug = R + rho*I
          Then iterate Riccati until Kinf converges.
        """
        P_lqr = self._compute_dlqr(A, B)

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

        C1 = np.linalg.inv(R_aug + B.T @ Pinf @ B)   # Quu_inv
        C2 = (A - B @ Kinf).T                          # (A-BK)^T

        return {
            'A': A,  'B': B,
            'Q': Q_aug,  'R': R_aug,
            'Kinf': Kinf,  'Pinf': Pinf,
            'C1': C1,  'C2': C2,
            'rho': self.rho,
        }

    # ------------------------------------------------------------------ #
    #  Online lookup
    # ------------------------------------------------------------------ #

    def lookup(self, alpha):
        """Nearest-neighbour cache lookup for a given alpha.

        Falls back to on-the-fly computation when no grid exists.
        """
        if not self.grid_alphas:
            A_d, B_d = linearise(alpha, self.quad_params)
            return self._compute_dare_cache(A_d, B_d)

        dists = [np.linalg.norm(alpha - a) for a in self.grid_alphas]
        idx   = int(np.argmin(dists))
        key   = tuple(np.round(self.grid_alphas[idx], 6))
        return self.cache_grid[key]

    # ------------------------------------------------------------------ #
    #  ADMM solver  (mirrors TinyMPC.solve_admm)
    # ------------------------------------------------------------------ #

    def solve(self, x0, alpha, K_iters=None,
              xmin_horizon=None, xmax_horizon=None):
        """Algorithm 1: ADMM solve with alpha-specific cache.

        Args:
            x0: (12,)  initial error state
            alpha: (4,) current flat parameter
            K_iters: override for max ADMM iterations
            xmin_horizon: (12, N) per-step state lower bounds (optional)
            xmax_horizon: (12, N) per-step state upper bounds (optional)

        Returns:
            x: (12, N)   predicted state trajectory
            u: (4, N-1)  optimal input trajectory (delta from equilibrium)
            status: 1 if converged, 0 otherwise
            k: number of iterations used
        """
        max_iter = K_iters if K_iters is not None else self.max_iter
        cache    = self.lookup(alpha)
        rho      = cache['rho']

        # Initialise from warm start
        x = np.copy(self.x_prev);  x[:, 0] = x0
        u = np.copy(self.u_prev)

        v = np.copy(self.v_prev)
        z = np.copy(self.z_prev)
        g = np.copy(self.g_prev)
        y = np.copy(self.y_prev)
        q = np.copy(self.q_prev)

        z_old = np.copy(z)
        v_old = np.copy(v)

        r = np.zeros(u.shape)
        p = np.zeros(x.shape)
        d = np.zeros(u.shape)

        # Zero reference (error coordinates)
        x_ref = np.zeros(x.shape)
        u_ref = np.zeros(u.shape)

        status = 0
        k = 0

        for k in range(max_iter):
            # --- backward pass (gradient) ---
            for j in range(self.N - 2, -1, -1):
                d[:, j] = cache['C1'] @ (cache['B'].T @ p[:, j + 1] + r[:, j])
                p[:, j] = q[:, j] + cache['C2'] @ p[:, j + 1] \
                           - cache['Kinf'].T @ r[:, j]

            # --- forward pass ---
            for j in range(self.N - 1):
                u[:, j] = -cache['Kinf'] @ x[:, j] - d[:, j]
                x[:, j + 1] = cache['A'] @ x[:, j] + cache['B'] @ u[:, j]

            # --- slack update ---
            for j in range(self.N - 1):
                z[:, j] = np.clip(u[:, j] + y[:, j], self.umin, self.umax)
                if xmin_horizon is not None:
                    v[:, j] = np.clip(x[:, j] + g[:, j],
                                      xmin_horizon[:, j], xmax_horizon[:, j])
                else:
                    v[:, j] = np.clip(x[:, j] + g[:, j], self.xmin, self.xmax)
            if xmin_horizon is not None:
                v[:, self.N - 1] = np.clip(
                    x[:, self.N - 1] + g[:, self.N - 1],
                    xmin_horizon[:, self.N - 1], xmax_horizon[:, self.N - 1])
            else:
                v[:, self.N - 1] = np.clip(
                    x[:, self.N - 1] + g[:, self.N - 1], self.xmin, self.xmax)

            # --- dual update ---
            for j in range(self.N - 1):
                y[:, j] += u[:, j] - z[:, j]
                g[:, j] += x[:, j] - v[:, j]
            g[:, self.N - 1] += x[:, self.N - 1] - v[:, self.N - 1]

            # --- linear cost update ---
            for j in range(self.N - 1):
                r[:, j] = -cache['R'] @ u_ref[:, j] \
                           - rho * (z[:, j] - y[:, j])
                q[:, j] = -cache['Q'] @ x_ref[:, j] \
                           - rho * (v[:, j] - g[:, j])
            p[:, self.N - 1] = -cache['Pinf'] @ x_ref[:, self.N - 1] \
                                - rho * (v[:, self.N - 1] - g[:, self.N - 1])

            # --- convergence check ---
            pri_res_u = np.max(np.abs(u - z))
            pri_res_x = np.max(np.abs(x - v))
            dua_res_u = np.max(np.abs(rho * (z_old - z)))
            dua_res_x = np.max(np.abs(rho * (v_old - v)))

            z_old = np.copy(z)
            v_old = np.copy(v)

            if (pri_res_u < self.abs_pri_tol and dua_res_u < self.abs_dua_tol
                    and pri_res_x < self.abs_pri_tol
                    and dua_res_x < self.abs_dua_tol):
                status = 1
                break

        # Save for warm start
        self.x_prev = x;  self.u_prev = u
        self.v_prev = v;  self.z_prev = z
        self.g_prev = g;  self.y_prev = y
        self.q_prev = q

        return x, u, status, k

    # ------------------------------------------------------------------ #
    #  Configuration helpers
    # ------------------------------------------------------------------ #

    def set_bounds(self, umax=None, umin=None, xmax=None, xmin=None):
        if umin is not None and umax is not None:
            self.umin = np.array(umin, dtype=np.float64)
            self.umax = np.array(umax, dtype=np.float64)
        if xmin is not None and xmax is not None:
            self.xmin = np.array(xmin, dtype=np.float64)
            self.xmax = np.array(xmax, dtype=np.float64)

    def set_tols_iters(self, max_iter=10, abs_pri_tol=1e-3, abs_dua_tol=1e-3):
        self.max_iter    = max_iter
        self.abs_pri_tol = abs_pri_tol
        self.abs_dua_tol = abs_dua_tol

    def _reset_warm_start(self):
        N, nx, nu = self.N, self.nx, self.nu
        self.v_prev = np.zeros((nx, N))
        self.z_prev = np.zeros((nu, N - 1))
        self.g_prev = np.zeros((nx, N))
        self.y_prev = np.zeros((nu, N - 1))
        self.q_prev = np.zeros((nx, N))
        self.x_prev = np.zeros((nx, N))
        self.u_prev = np.zeros((nu, N - 1))
