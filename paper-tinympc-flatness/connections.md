# Connections: TinyMPC-Flat ↔ TTT ↔ Yang-Mills ↔ ABC

This document maps the conceptual links between four papers:
1. **TinyMPC-Flat** (this paper): α-parameterized templates for quadrotor MPC
2. **TTT** (Zhang 2026): Tunnel, Threshold, Turbulence — unified iterative optimal control
3. **Yang-Mills** (Zhang 2026): Renormalization group structure of the DARE iteration
4. **ABC** (Zhang 2026): Adjoint, Barrier, and Controllability in constrained OC

---

## The Unifying Dictionary

| TinyMPC-Flat | TTT | Yang-Mills | ABC |
|---|---|---|---|
| Hover DARE solution P∞ | Seed (fixed-point controller) | Strong-coupling base case | DARE stabilizability |
| Constraint-feasible region | Tunnel (absorbing Lyapunov sublevel) | Mass gap (spectral gap > 0) | Hope function V* > 0 |
| ADMM iteration | Game iteration (MPC/ADMM/SQP) | Renormalization group step | Costate field update |
| Flat parameter α ∈ ℝ⁴ | Scheduling variable | Gauge parameter | Adjoint parameter |
| K(α) gain interpolation | Seed perturbation | UV regime linearization | Costate ∇V at one point |
| Barrier Hessian H_μ | Threshold function | Source term Q_j | Barrier certificate |
| Zero-iteration controller | Seed alone (no tunnel) | UV regime alone (no IR) | Single-point costate |
| ADMM projection | Tunnel construction | IR contraction | Full costate field |
| 4 sensitivity matrices ∂K/∂α_i | 4 perturbation directions | 4 generators of SU(2) | 4 adjoint components |
| R^12 → R^4 via flatness | Dimension reduction via seed | Gauge compression | Controllability reduction |

---

## Connection 1: Seed ↔ Zero-Iteration Controller

The zero-iteration control law `u = u*(α) - K(α)·δx` is a **seed** in the
TTT sense: a fixed-point controller at one operating point (hover), extended
by first-order sensitivity to nearby operating points.

- **TTT**: The seed is the base controller around which the tunnel (region of
  attraction) is constructed. The seed alone provides local stability but no
  global guarantees.
- **Yang-Mills**: The strong-coupling base case (hover DARE) corresponds to
  the UV fixed point. The DARE iteration P_j converges = the RG flow reaches
  the IR fixed point. Convergence of P_j ↔ mass gap existence.
- **ABC**: The DARE stabilizability condition (A,B) stabilizable guarantees
  the seed exists. This is the controllability prerequisite.

**Key identity**: TTT's Main Theorem states
  bounded P_j ⟺ spectral gap > 0 ⟺ Lyapunov descent ⟺ tunnel exists.
Our DARE solution P∞ is exactly this bounded cost-to-go.
The tunnel radius r(j) = ||P_j||^{-1/2} gives the guaranteed region of
attraction around the equilibrium.

---

## Connection 2: Tunnel ↔ Constraint Satisfaction

The constraint-feasible region in MPC corresponds to the **tunnel** in TTT:
a sublevel set of the Lyapunov function where the controller maintains both
stability and constraint satisfaction.

- **TTT**: The tunnel is the absorbing Lyapunov sublevel set
  {x : V(x) ≤ V_max} where the closed-loop system remains for all time.
  Inside the tunnel, the seed controller is sufficient.
- **Yang-Mills**: The mass gap (spectral gap > 0 in the DARE eigenvalues)
  ensures exponential contraction of the error state. This is the
  closed-loop stability margin — the larger the gap, the faster the
  convergence and the wider the tunnel.
- **ABC**: V* > 0 (the Hope function is strictly positive) serves as a
  constraint feasibility certificate. When V* > 0 everywhere in the
  constrained region, the barrier method can find a feasible path.

**For our paper**: The circle-in-square experiment shows that the tunnel
(ADMM-feasible region) can handle references outside the square, while
the seed alone (zero-iteration controller) cannot. The tunnel requires
iterative construction (ADMM iterations).

---

## Connection 3: Path Integral ↔ ADMM Iteration

The ADMM iteration in TinyMPC has structural parallels to path integral
methods and renormalization group flows.

- **TTT**: The Boltzmann weighting e^{-βV} in MPPI (Model Predictive Path
  Integral) corresponds to a soft version of the constraint projection.
  Sampling + reweighting ≈ ADMM minimize + project.
- **Yang-Mills**: Mirror descent on the DARE = renormalization group step.
  Each RG step integrates out one scale of fluctuations; each ADMM
  iteration incorporates one layer of constraint information.
  The iterative DARE refinement P_{j+1} = f(P_j) is a discrete RG flow.
- **ABC**: The ε-relaxation V*·μ = ε in interior point methods corresponds
  to the log-barrier parameter μ. As μ → 0, the barrier solution
  approaches the constrained optimum — analogous to the RG flow reaching
  the IR fixed point.

---

## Connection 4: Flatness ↔ Gauge Compression

Differential flatness compresses the 12D state space to 4D via the flat
parameter α = (a_x, a_y, a_z, ψ). This has structural parallels to gauge
reduction in Yang-Mills theory.

- **Our paper**: R^12 → R^4 via the flat map Φ. The 4 sensitivity matrices
  ∂K/∂α_i span the tangent space of the gain manifold at hover.
- **Yang-Mills**: Gauge symmetry reduces the infinite-dimensional connection
  space to a finite-dimensional moduli space. The 4 generators of SU(2)
  parameterize the gauge group — analogous to our 4 flat parameters.
- **Key parallel**: The α-scheduling variable acts as a **gauge connection**
  on the control manifold. Changing α corresponds to a gauge transformation
  that preserves the physical dynamics while changing the coordinate
  representation.

This is why flatness is so powerful: it identifies the minimal scheduling
space (4D instead of 12D) by exploiting the geometric structure of the
dynamics — just as gauge theory identifies minimal degrees of freedom
by exploiting symmetry.

---

## Connection 5: Why the Barrier Can't Work (The Theorem)

The circle-in-square experiment demonstrates a fundamental limitation:
the zero-iteration barrier controller exhibits identical violation to the
unconstrained analytic controller (both 88.4mm). This has deep theoretical
roots:

- **TTT**: The seed alone ≠ tunnel. The seed provides a local controller,
  but the tunnel (region of guaranteed constraint satisfaction) requires
  iterative construction. No amount of local information at the seed point
  can create global constraint guarantees.
- **Yang-Mills**: The UV regime alone ≠ mass gap. The UV (high-energy /
  local) analysis provides the seed, but the mass gap (IR / global property)
  requires the full RG flow. Both UV and IR regimes must cooperate.
- **ABC**: The costate λ = ∇V evaluated at one point ≠ global directional
  information. The barrier Hessian H_μ at the equilibrium encodes constraint
  information at that point only. The costate *field* ∇V(x) rotates near
  constraint boundaries — a constant matrix K cannot capture this rotation.

**Formal argument**: K(α) is the costate ∇V evaluated at the reference
(equilibrium). Near constraint boundaries, the costate field ∇V(x) develops
strong spatial variation (it "rotates" to point away from the constraint).
A constant-gain approximation K ≈ ∇V(x*) cannot capture this rotation.
This is why Barrier = Analytic = 88.4mm: the barrier Hessian modifies the
gain magnitude but not its direction, and it is the directional information
that is needed for constraint satisfaction when the reference exits the
feasible region.

---

## Connection 6: Possible Extensions

Three paths to constraint-aware real-time control, ordered by computational
budget, each motivated by the theoretical connections above:

### 6.1 MPPI-Seed Hybrid (~1,500 ops)

Use K(α) as the base policy (seed), sample N_s ≈ 10 perturbations,
weight by Boltzmann e^{-β·cost} where cost includes constraint penalty.

- **TTT connection**: Creates a soft "tunnel" via ensemble averaging.
  The samples explore the tunnel boundary; the Boltzmann weighting
  concentrates mass inside.
- **Complexity**: O(N_s · n) ≈ 10 × 12 × 12 ≈ 1,500 ops.
- **Property**: Deterministic-time, no convergence loop. Soft constraints.

### 6.2 Costate Lookup Table (~500 ops)

Precompute ∇V(x) at a grid of positions near constraint boundaries.
Online: lookup nearest costate, correct K(α) with directional information.

- **ABC connection**: Adds the "missing ingredient" — the costate field
  ∇V(x) — to the zero-iteration controller. The DRM functor maps the
  gain K to the adjoint variable λ = ∇V; the lookup table provides the
  full field instead of a single-point evaluation.
- **Complexity**: O(n·m) + table lookup ≈ 500 ops.
- **Property**: Deterministic-time. Captures constraint direction.

### 6.3 Constraint-Shaped Terminal Cost (~260 ops, same budget)

Design P∞ via DARE whose terminal cost encodes constraint geometry
(not just barrier Hessian at equilibrium, but the barrier evaluated
along the predicted trajectory).

- **Yang-Mills connection**: Modifies both the UV source Q_j and the
  IR boundary condition simultaneously. The terminal cost shapes the
  entire cost-to-go landscape, not just the local Hessian.
- **Complexity**: O(n·m) ≈ 260 ops (same as current Analytic Template).
- **Property**: Requires one-step prediction x_{k+1} = Ax + Bu before
  gain lookup. Could achieve constraint-aware behavior at zero
  additional online cost.

### Connection to TTT's SOP Algorithm

These three extensions correspond to the three nested games in TTT:
- MPPI-Seed ↔ outer game (sampling/exploration)
- Costate Lookup ↔ middle game (adjoint/directional information)
- Constraint-Shaped Cost ↔ inner game (cost landscape shaping)

The tunnel emerges from the game structure, not from any single mechanism.
This suggests that full constraint satisfaction at zero-iteration cost
may require combining all three approaches.

---

## Summary

The key insight across all four papers: **local information (seed, UV,
single-point costate) is necessary but not sufficient for global
guarantees (tunnel, mass gap, full costate field)**. The zero-iteration
controller provides the local part; the ADMM iterations / path integral
sampling / costate field provides the global part. The theoretical
frameworks from TTT, Yang-Mills, and ABC make this tradeoff precise
and suggest specific paths to bridge the gap.
