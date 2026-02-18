# TinyMPC + Differential Flatness

Extending [TinyMPC](https://github.com/TinyMPC/TinyMPC) with differential-flatness-based
$\alpha$-parameterized templates for quadrotor control.
Pure Python/NumPy reference implementation.

## Controller Hierarchy

```
Online Relin  --cache-->  Flat Template  --K→0-->  Barrier Templ.  --μ→0-->  Analytic Templ.
 O(n³+Kn²)                 K·O(n²)                  O(n·m)                    O(n·m)
 Hard constr.              Hard constr.             Soft constr.              Clip only
```

| Variant | What it does | Online cost |
|---|---|---|
| **TinyMPC** (baseline) | Fixed hover linearization + ADMM | $K \cdot O(n^2)$ |
| **Online Relin** | Fresh Jacobian + DARE + ADMM per step | $O(n^3) + K \cdot O(n^2)$ |
| **Flat Template** | Precomputed DARE cache grid, lookup + ADMM | $K \cdot O(n^2)$ |
| **Barrier Template** | Log-barrier Hessian baked into DARE, zero-iteration | ~260 ops |
| **Analytic Template** | First-order gain interpolation, zero-iteration | ~260 ops |

The key idea: differential flatness compresses the scheduling space from $\mathbb{R}^{12}$ to $\mathbb{R}^4$
via $\alpha = (a_x, a_y, a_z, \psi)$, so only 4 sensitivity matrices (240 floats) are needed.

## Quick Start

```bash
# Install deps
pip install numpy matplotlib scipy autograd

# Run experiments
python examples/hover/hover.py              # hover stabilization
python examples/traj/traj.py                # figure-8 tracking
python examples/circle_square/circle_square.py  # constrained circle-in-square

# Run tests
python -m pytest tests/ -v
```

### Flags (hover.py / traj.py)

```
--adapt     enable rho adaptation
--wind      enable wind disturbance
--recache   recompute cached data
--heuristic use heuristic rho adapter
```

## Project Structure

```
├── src/
│   ├── quadrotor.py            # 13-state quadrotor dynamics (autograd)
│   ├── tinympc.py              # Core ADMM-based MPC solver
│   ├── flat_linearization.py   # α-parameterized analytic Jacobians (no autograd)
│   ├── flat_template.py        # Flat Template MPC: DARE cache grid + ADMM
│   ├── analytic_template.py    # Analytic Template: K₀ + ΣδαᵢdK/dαᵢ, ~260 ops
│   ├── barrier_template.py     # Barrier Template: log-barrier in DARE cache
│   ├── rho_adapter.py          # ADMM penalty adaptation
│   └── hybrid_rho_adapter.py   # Mixed fixed/adaptive rho
│
├── examples/
│   ├── hover/hover.py          # Hover benchmark (4 variants)
│   ├── traj/traj.py            # Figure-8 benchmark (4 variants)
│   ├── circle_square/
│   │   ├── circle_square.py    # Constrained tracking (4 variants, cascade demo)
│   │   └── verify_constraints.py
│   ├── flat_template/demo.py   # Flat Template vs TinyMPC
│   └── analytic/demo.py        # Three-variant comparison
│
├── tests/
│   └── test_flat_linearization.py  # 4 tests: linearization, gain approx,
│                                   #   control equivalence, flat equilibrium
│
├── paper-tinympc-flatness/         # Paper 1: α-Parameterized Templates
│   ├── main.tex                    #   (multi-file LaTeX, 10 pages)
│   ├── sections/                   #   with appendix on TTT/YM/ABC connections
│   └── connections.md              #   cross-paper theory map
│
├── paper-switched-lpv-template/    # Paper 2: Switched LPV Template MPC
│   └── main.tex
│
└── readings/                       # Reference papers (TinyMPC, SQP-OC)
```

## Key Results (Python reference, relative timing)

**Figure-8 tracking** (N=15, ρ=5, 200 steps):

| | Avg err (m) | ADMM iters | Wall time |
|---|---|---|---|
| TinyMPC | **0.033** | 8.0 | 1.16s |
| Online Relin | 0.035 | 5.1 | 22.02s |
| Flat Template | 0.035 | 3.6 | 0.75s |
| Analytic | 0.036 | 0 | **0.26s** |

**Circle-in-square** (R=0.3m, L=R/√2, 500 steps):
ADMM variants → 0mm violation; zero-iteration variants → 88mm (geometric limit).
Barrier = Analytic because the barrier Hessian can't override a reference outside the feasible region.

## Papers

1. **TinyMPC with Differential Flatness** (`paper-tinympc-flatness/`)
   — α-parameterized templates, complexity cascade, barrier-augmented DARE
2. **Template MPC for Switched LPV Robotic Systems** (`paper-switched-lpv-template/`)
   — learned contact scheduling with structured dynamics
