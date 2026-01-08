# The Geometer's Path

## A 30-month curriculum for deep mastery of differential geometry

*For the mathematician-physicist-programmer who wants to understand,
not just know.*

---

## Prelude: What this curriculum is

This is a path to genuine understanding of differential geometry —
from the curvature of curves to the symplectic structure of phase
space to the fiber bundles of gauge theory. It is designed for someone
with solid calculus, linear algebra, and basic analysis who wants
research-level comprehension of geometric mechanics and its
computational implementation.

**What makes this curriculum different:**

It is *honest about time*. Deep mathematical understanding cannot be
rushed. Thirty months of serious work — not because the material is
impossibly difficult, but because absorption, connection-making, and
intuition-building take time that cannot be compressed.

It *spirals rather than marches*. You will meet each major idea
multiple times. First encounter: get the shape. Second encounter:
understand with new tools. Third encounter: see connections invisible
before. The curriculum explicitly schedules these returns.

It *computes alongside theory*. Every theoretical concept gets a
computational companion within days. When you learn pushforward, you
implement `jvp`. When you learn parallel transport, you code it. This
is not application — it is embodiment.

It *creates to learn*. Each phase produces content: a video
explanation, a blog post, a talk, a First Principles Club session. The
act of explaining reveals gaps that reading hides. Your learning
becomes a gift to others.

It *acknowledges limitations*. This curriculum does not cover
algebraic topology deeply, complex geometry, or several other
important topics. These are noted as "branches not taken" — paths you
may choose to explore later, from a position of strength.

**A word about difficulty:**

There will be evenings when the notation swims and the theorems won't
stick. There will be weeks when you wonder if you're really learning
or just going through motions. This is normal. Understanding comes in
waves: long plateaus, then sudden clarity. Trust the structure. It is
designed to carry you through the plateaus.

The mathematics you're pursuing is genuinely beautiful. Generations of
brilliant minds have found meaning in these structures. You are
joining a conversation that spans centuries. That's worth some
difficult evenings.

---

## The architecture

### Four strands, woven throughout

Rather than sequential topics, four strands develop in parallel:

| Strand | Focus | Character |
|--------|-------|-----------|
| **Geometric core** | The mathematics itself | Rigorous, precise, building toward abstraction |
| **Physical grounding** | Mechanics and physics motivation | Why these structures matter, where they came from |
| **Computational embodiment** | Implementation and numerical methods | Making ideas concrete through code |
| **Content creation** | Explaining to learn, building in public | Videos, posts, talks that cement understanding |

Every month touches at least three strands. Some weeks emphasise one;
the weave is the point.

### A typical week

A sustainable pace for 30 months requires rhythm. Here's a template
for a 10-12 hour week:

- **3-4 reading sessions** (1-1.5 hours each): Primary text, work
  through proofs with pencil in hand
- **2 problem sessions** (1-1.5 hours each): Exercises from the text,
  struggle productively
- **1 coding session** (1.5-2 hours): Implement the week's concepts in
  JAX
- **1 synthesis session** (1 hour): Write notes, update content
  drafts, reflect on connections

This cadence is a guide, not a mandate. Some weeks will be heavier on
reading; others will be dominated by a single hard problem. The rhythm
matters more than the precise allocation. Fifteen minutes on a hard
day beats zero minutes.

### Spiral returns

At the end of each major phase, you return to earlier material with
new eyes:

- After Phase 2 (manifolds): revisit surfaces as embedded submanifolds
- After Phase 3 (Riemannian): revisit geodesics as Riemannian objects,
  preview their Hamiltonian nature
- After Phase 4 (symplectic): revisit Riemannian geodesic flow as
  Hamiltonian flow
- After Phase 5 (bundles): revisit connections on the frame bundle as
  the proper setting for Riemannian geometry

These returns are scheduled, not optional. They are where deep
understanding crystallises.

### Content creation pipeline

Each phase produces specific deliverables:

| Phase | Primary content | Secondary content |
|-------|-----------------|-------------------|
| 1 | Video: "Why You Can't Flatten a Sphere" | Blog: Implementing Frenet-Serret |
| 2 | Video: "Automatic Differentiation IS Differential Geometry" | First Principles Club (FPC) session: Forms and integration |
| 3 | Interactive: Parallel transport visualiser | Blog: The geometry behind general relativity |
| 4 | Talk: "Mechanics IS Geometry" | Video series: Building symplectic integrators |
| 5 | Blog series: "Fiber Bundles for Programmers" | FPC session: From Maxwell to Yang-Mills |
| 6 | Project writeup (publishable) | Tutorial: Geometric deep learning from scratch |

Creating this content is not separate from learning — it *is*
learning.

---

## The Capstone Object

### The capstone: one object, six views

Throughout this curriculum, you will return to a single geometric
object: **geodesic flow on the 2-sphere**.

The sphere is simple enough to compute with, rich enough to illustrate
every concept. As your understanding deepens, the same object reveals
new structure:

| Phase | What you see |
|-------|--------------|
| 1 | Great circles as curves with constant curvature, zero torsion |
| 2 | S² as a manifold; geodesics as integral curves of a vector field on TS² |
| 3 | Geodesics via Levi-Civita; parallel transport; holonomy around triangles |
| 4 | Geodesic flow as Hamiltonian system on T*S²; angular momentum as momentum map |
| 5 | Frame bundle F(S²); Levi-Civita as connection; curvature as field strength |
| 6 | Learning geodesics; the sphere in geometric deep learning |

Each phase ends with a capstone exercise: revisit the sphere with your
new tools. Watch the same object become richer. This is what
understanding feels like — not learning new facts, but seeing familiar
things with new eyes.

---

## When the path is hard

You will get stuck. This is not a sign of failure; it's a sign you're
doing real mathematics.

**When you've been stuck for days:**

Try working a simpler special case — if a theorem confuses you in
general, see what it says for circles, or spheres, or ℝ². Draw
pictures, even bad ones. Put the problem down for three days and work
on something else; your unconscious mind will continue processing.
Return with fresh eyes.

**When motivation fades:**

Do fifteen minutes rather than zero. Read one page. Work one exercise.
The goal is to keep the thread alive. Momentum matters more than
speed. A week of fifteen-minute sessions beats a week of nothing
followed by a guilt-laden attempt at a marathon.

**When life intervenes:**

It will. The kids will get sick. Work will explode. You'll miss two
weeks, or a month. When you return, don't pick up where you left off —
return to the last spiral return point. Rebuild context before pushing
forward. This isn't losing ground; it's how memory works.

**When you need help:**

Post to Mathematics Stack Exchange. Describe what you're trying to
understand and what you've attempted. The community is generous with
genuine questions. You can also revisit Schuller's lectures — hearing
an idea explained differently often unlocks it.

**Return without shame:**

If you step away for months, you can come back. The mathematics isn't
going anywhere. The structures are patient. They've been waiting for
centuries; they'll wait for you.

---

## Companion texts

These books are not read sequentially but consulted constantly. Keep
them within arm's reach.

### The Rosetta Stone

**Theodore Frankel, *The Geometry of Physics*** (3rd ed., Cambridge,
2011)

Whenever pure mathematics feels untethered from meaning, Frankel shows
why physicists care. Covers forms, manifolds, Riemannian and
symplectic geometry, with relentless attention to physical
applications — elasticity, electromagnetism, Yang-Mills,
thermodynamics. Read the relevant chapter *after* struggling with the
pure treatment. It will click.

*Engagement level: Read for perspective, consult constantly*

### The Beautiful One

**Michael Spivak, *A Comprehensive Introduction to Differential
Geometry*, Volume 1** (3rd ed., Publish or Perish, 1999)

The most beautifully written differential geometry text in existence.
When something seems unmotivated in Tu or Lee, Spivak shows how the
ideas actually developed. His historical notes are treasures. His
treatment of the intrinsic geometry of surfaces is unmatched. Volume 2
(on connections and curvature) is worth consulting in Phase 3.

*Engagement level: Read key chapters, savour the historical notes*

### The Lectures

**Frederic Schuller, *Lectures on the Geometrical Anatomy of
Theoretical Physics***

- [Video lectures](https://www.youtube.com/playlist?list=PLPH7f_7ZlzxTi6kS4vCmv4ZKm9u8g5yic)
- [Simon Rea's notes](https://github.com/lazierthanthou/Lecture_Notes_GR)

Twenty-eight lectures of ~90 minutes each. Schuller builds from logic
and set theory through fiber bundles and Yang-Mills. Mathematical
precision exceeding most textbooks, yet remarkably clear. Won a
prestigious German teaching award. Perfect for transcription practice.

*Engagement level: Transcribe selected lectures (schedule below)*

---

## A note on conventions

Different books use different sign conventions, index placements, and
notational choices. This causes endless confusion. Here are the
conventions this curriculum follows:

**Riemann curvature tensor**: $R(X,Y)Z = \nabla_X \nabla_Y Z -
\nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z$. This is Lee's convention and
most modern differential geometry texts. *Warning*: Do Carmo and some
physics texts use the opposite sign. If your calculation differs by a
sign, check conventions first.

**Christoffel symbols**: $\Gamma^i_{jk}$ denotes the components of
$\nabla_{\partial_j}\partial_k = \Gamma^i_{jk}\,\partial_i$. Reading
the indices: $i$ is the output component, $j$ is the differentiation
direction, $k$ is the input field.

**Symplectic form**: We use $\omega = \sum_i dq^i \wedge dp_i$ on
cotangent bundles, with coordinates ordered
$(q^1,\ldots,q^n,p_1,\ldots,p_n)$. The matrix representation is $J =
\begin{bmatrix} 0 & I \\ -I & 0 \end{bmatrix}$.

**Musical isomorphisms**: ♭ ("flat") lowers indices using the metric:
$v^\flat = g(v, \cdot)$. ♯ ("sharp") raises them: $\alpha^\sharp =
g^{-1}(\alpha, \cdot)$.

When in doubt, compute a simple example and check against a trusted
source.

---

## A note on debugging geometry code

When your geometric computation gives wrong answers, check these in
order:

1. **Index conventions** — Is the first index the output or the input?
   Different sources differ.
2. **Sign conventions** — Especially for curvature tensors and
   symplectic forms. A sign error doesn't mean your code is broken; it
   may mean your source uses a different convention.
3. **Numerical precision** — Are you using float64? Geometric
   computations involving many derivatives accumulate error quickly.
4. **Coordinate ordering** — Is it (q, p) or (p, q)? (θ, φ) or (φ, θ)?

Most bugs in geometric code are convention mismatches, not algorithmic
errors. When your answer differs from the textbook by a sign, or by a
factor of 2, check conventions before assuming you've made a mistake.

---

## Phase 0: Calibration

**Duration: 2-3 weeks**

*Goal: Honestly assess where you are, not where you think you are. Set
up your computational environment.*

These diagnostic problems reveal where your foundations are solid and
where they need reinforcement. Approach them as a gift to your future
self — honest assessment now prevents frustration later.

### Diagnostic problems

Work through these without references. Be honest about gaps.

**Linear algebra:**

1. What is the rank of a linear map? Prove rank-nullity.
2. Diagonalise a 3×3 symmetric matrix. What guarantees real
   eigenvalues?
3. Define dual space V*. If dim(V) = n, what is dim(V*)?
4. What is a bilinear form? When is it non-degenerate?

**Analysis:**

5. Define continuity using ε-δ. Define it using open sets.
6. State and prove the inverse function theorem (or at least state it
   precisely).
7. What does it mean for a function f: ℝⁿ → ℝᵐ to be smooth?
8. Define the derivative Df(p) of such a function at a point. What is
   its type?

**Topology (light):**

9. What is a topological space? What are open sets?
10. Define compactness. Prove that [0,1] is compact.
11. Define connectedness. Is ℝ \ {0} connected?

**Self-assessment:**

- If linear algebra is shaky: Work through **Axler, *Linear Algebra
  Done Right*** (3-4 weeks)
- If analysis is shaky: Work through **Abbott, *Understanding
  Analysis*** (4-6 weeks)
- If topology is foreign: Read **Munkres, Chapters 2-4** for
  familiarity (2-3 weeks)

Be honest with yourself. Time invested in foundations pays compound
interest throughout everything that follows.

### Computational setup

```python
# Create dedicated environment
# conda create -n geometry python=3.11
# conda activate geometry

# For geometric computations, always use float64 precision.
# Add this at the start of every notebook:
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Core geometry libraries
import geomstats
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.hyperbolic import Hyperbolic

# ODE solvers with symplectic methods
import diffrax

# Symbolic computation (for forms, exact derivatives)
import sympy
from sympy import symbols, Function, diff, simplify
```

### First computation

Before any theory, make something work:

```python
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from geomstats.geometry.hypersphere import Hypersphere

# The 2-sphere as a Riemannian manifold
sphere = Hypersphere(dim=2)

# A point on the sphere (in ambient ℝ³ coordinates)
point = sphere.random_point()
print(f"Point on S²: {point}")
print(f"Verify on sphere: |p|² = {jnp.sum(point**2):.6f}")

# A tangent vector at that point (lives in TₚS²)
tangent = sphere.random_tangent_vec(point)
print(f"Tangent vector: {tangent}")
print(f"Verify tangent: p · v = {jnp.dot(point, tangent):.6f}")

# Exponential map: flow along geodesic for unit time
new_point = sphere.metric.exp(tangent, point)
print(f"After exponential map: {new_point}")
print(f"Still on sphere: |p'|² = {jnp.sum(new_point**2):.6f}")
```

You don't fully understand this code yet. That's the point. Return to
it after Phase 1 and again after Phase 3 — watch your understanding
deepen.

### Deliverable

Write a 500-word reflection: "What I know, what I don't know, what I
want to understand." This becomes the first entry in your learning
journal.

---

## Phase 1: Curves, surfaces, and geometric intuition

**Duration: 4 months (Months 1-4)**

*Goal: Build geometric intuition through objects you can visualise and
compute with.*

### Month 1: Curves in space

**Primary text: Do Carmo, *Differential Geometry of Curves and
Surfaces*, Chapter 1**

The geometry of curves is where intuition begins. A curve is a map
from an interval into space. Its geometry is captured by curvature
(how fast it turns) and torsion (how it twists out of a plane).

**Week 1-2: Parametrised curves**

- Regular curves, arc length parametrisation
- The tangent vector T as velocity
- Reparametrisation and geometric invariants

**Week 3-4: Frenet-Serret apparatus**

- Curvature κ, the principal normal N
- Torsion τ, the binormal B
- The Frenet-Serret equations: how T, N, B evolve

**Key insight**: The Frenet-Serret frame is an orthonormal basis that
moves with the curve. The equations describe how this frame rotates as
you traverse the curve. Curvature measures rotation toward N; torsion
measures rotation toward B.

**Visual anchor**: Picture an orthonormal frame {T, N, B} rigidly
attached to a bead sliding along a wire. As the bead moves, the frame
rotates: T always points forward along the curve, N points toward the
center of the "osculating circle" (the best-fit circle at that
instant), and B = T × N completes the right-hand system. The
Frenet-Serret equations quantify how fast this frame rotates: κ
measures rotation from T toward N; τ measures rotation from N toward
B. For a helix, both rotations are constant. For a plane curve, τ = 0
and the frame never twists out of the plane. Sketch the frame at
several points along a helix — this single picture will anchor
everything.

**Exercises**: Do Carmo Chapter 1, problems 1-5, 7, 9, 12, 15.

**Computation**: Implement Frenet-Serret numerically.

```python
import jax
import jax.numpy as jnp

def frenet_serret(curve, t):
    """
    Compute Frenet-Serret frame for a parametric curve using autodiff.

    Args:
        curve: function t → ℝ³ (must be JAX-compatible)
        t: parameter value

    Returns:
        T, N, B: unit tangent, normal, binormal
        kappa: curvature
        tau: torsion
    """
    # Derivatives via JAX autodiff
    r = curve(t)
    dr = jax.jacfwd(curve)(t)      # First derivative (velocity)
    ddr = jax.jacfwd(jax.jacfwd(curve))(t)   # Second derivative
    dddr = jax.jacfwd(jax.jacfwd(jax.jacfwd(curve)))(t)  # Third derivative

    # Speed and unit tangent
    speed = jnp.linalg.norm(dr)
    T = dr / speed

    # Curvature: κ = |dr × ddr| / |dr|³
    cross = jnp.cross(dr, ddr)
    cross_norm = jnp.linalg.norm(cross)
    kappa = cross_norm / speed**3

    # Handle inflection points (κ ≈ 0) gracefully
    safe_cross_norm = jnp.maximum(cross_norm, 1e-10)

    # Principal normal (direction of curvature)
    # N = (dr × ddr) × dr / |...| when κ ≠ 0
    dT_unnorm = ddr / speed - jnp.dot(ddr, T) * T / speed
    N = dT_unnorm / jnp.linalg.norm(dT_unnorm + 1e-10)

    # Binormal
    B = jnp.cross(T, N)

    # Torsion: τ = (dr × ddr) · dddr / |dr × ddr|²
    tau = jnp.dot(cross, dddr) / (safe_cross_norm**2)

    return T, N, B, kappa, tau


# Test on helix: γ(t) = (cos t, sin t, t)
def helix(t):
    return jnp.array([jnp.cos(t), jnp.sin(t), t])

T, N, B, kappa, tau = frenet_serret(helix, 1.0)
print(f"Helix at t=1:")
print(f"  Tangent T: {T}")
print(f"  Normal N:  {N}")
print(f"  Binormal B: {B}")
print(f"  Curvature κ = {kappa:.4f}")  # Should be 1/2 for this helix
print(f"  Torsion τ = {tau:.4f}")      # Should be 1/2 for this helix

# Verify: for helix (a cos t, a sin t, bt), κ = a/(a² + b²), τ = b/(a² + b²)
# Here a = b = 1, so κ = τ = 1/2 ✓
```

**Exercise**: Visualise the Frenet-Serret frame moving along curves of
your choosing: helix, trefoil knot, a curve with an inflection point
(where κ = 0).

### Month 2: Surfaces — first contact

**Primary text: Do Carmo, Chapter 2**

A surface is (locally) a map from ℝ² into ℝ³. The key object is the
tangent plane — the best linear approximation at each point.

**Week 1-2: Regular surfaces**

- Coordinate patches, smooth maps between surfaces
- The tangent plane as image of the differential
- The first fundamental form: measuring lengths on the surface

**Week 3-4: Maps and calculus on surfaces**

- Smooth functions on surfaces
- Differentials of maps
- Area, integration on surfaces

**Key insight**: The first fundamental form I(v, w) = ⟨v, w⟩ is an inner
product on each tangent plane, inherited from the ambient space. It
lets you measure lengths and angles on the surface without leaving it.
In coordinates (u, v) with metric coefficients E, F, G:

$$I = E \, du^2 + 2F \, du \, dv + G \, dv^2$$

This single object encodes all intrinsic geometry.

**Exercises**: Do Carmo Chapter 2, problems 1, 3, 5, 8, 12, 16.

**Common sticking point**: The distinction between a surface as a
subset of ℝ³ and a surface as a parametrised map can be confusing. The
parametrisation is a tool; the intrinsic geometry doesn't depend on
which parametrisation you choose.

**Computation**: Implement the first fundamental form using automatic
differentiation.

```python
import jax
import jax.numpy as jnp

def first_fundamental_form(surface, u, v, du, dv):
    """
    Compute I(w, w) where w = du ∂/∂u + dv ∂/∂v.
    Uses JAX automatic differentiation for accuracy.

    Args:
        surface: function (u, v) → ℝ³
        u, v: coordinates on the surface
        du, dv: components of tangent vector

    Returns:
        I(w, w): squared length of tangent vector w
    """
    def surface_vec(uv):
        return surface(uv[0], uv[1])

    # Jacobian gives partial derivatives: columns are Su and Sv
    J = jax.jacfwd(surface_vec)(jnp.array([u, v]))  # Shape: (3, 2)
    Su, Sv = J[:, 0], J[:, 1]

    # Metric coefficients
    E = jnp.dot(Su, Su)
    F = jnp.dot(Su, Sv)
    G = jnp.dot(Sv, Sv)

    # First fundamental form
    return E * du**2 + 2 * F * du * dv + G * dv**2


# Test: torus with major radius R, minor radius r
def torus(u, v, R=2.0, r=0.5):
    """Torus parametrised by (u, v) ∈ [0, 2π) × [0, 2π)."""
    return jnp.array([
        (R + r * jnp.cos(v)) * jnp.cos(u),
        (R + r * jnp.cos(v)) * jnp.sin(u),
        r * jnp.sin(v)
    ])

u, v = 0.5, 1.0
du, dv = 0.1, 0.2

I_val = first_fundamental_form(torus, u, v, du, dv)
print(f"First fundamental form I(w,w) = {I_val:.6f}")
```

**Exercise**: Compute the length of a curve on the torus. Verify that
circles of constant u or constant v have the lengths you'd expect from
the geometry.

### Month 3: Curvature of surfaces

**Primary text: Do Carmo, Chapter 3**

The Gauss map sends each point on a surface to its unit normal. How
this map stretches and rotates tells you about the surface's
curvature.

**Week 1-2: The Gauss map and the second fundamental form**

- The Gauss map N: S → S²
- Shape operator (Weingarten map): dN_p: T_pS → T_pS
- Second fundamental form II(v, w) = -⟨dN_p(v), w⟩

**Week 3-4: Principal, Gaussian, and mean curvature**

- Principal curvatures κ₁, κ₂ as eigenvalues of shape operator
- Gaussian curvature K = κ₁κ₂ = det(shape operator)
- Mean curvature H = (κ₁ + κ₂)/2 = ½ trace(shape operator)
- Classification: elliptic (K > 0), hyperbolic (K < 0), parabolic (K =
  0)

**Key insight**: The shape operator asks: "If I move in direction v
along the surface, how does the normal turn?" The answer is another
tangent vector. Principal curvatures are how fast the normal turns in
the directions of maximum and minimum turning.

**Common sticking point**: The shape operator can feel abstract.
Visualise it: at a point on a sphere, pushing in any direction makes
the normal turn toward you (positive curvature). On a saddle, pushing
one direction makes the normal turn toward you; pushing perpendicular
makes it turn away (opposite signs, so K < 0).

**Exercises**: Do Carmo Chapter 3, problems 2, 5, 8, 11, 14, 17.

**Computation**: Compute and visualise curvatures.

```python
import jax
import jax.numpy as jnp

def surface_curvatures(surface, u, v):
    """
    Compute Gaussian and mean curvature at a point.

    Returns: K (Gaussian), H (mean), kappa1, kappa2 (principal)
    """
    def surf(uv):
        return surface(uv[0], uv[1])

    # First derivatives via Jacobian
    J = jax.jacfwd(surf)(jnp.array([u, v]))
    Su, Sv = J[:, 0], J[:, 1]

    # Second derivatives via Hessian of each component
    def Su_func(uv):
        return jax.jacfwd(surf)(uv)[:, 0]
    def Sv_func(uv):
        return jax.jacfwd(surf)(uv)[:, 1]

    uv = jnp.array([u, v])
    H_u = jax.jacfwd(Su_func)(uv)
    H_v = jax.jacfwd(Sv_func)(uv)

    Suu = H_u[:, 0]
    Suv = H_u[:, 1]
    Svv = H_v[:, 1]

    # Unit normal
    N = jnp.cross(Su, Sv)
    N = N / jnp.linalg.norm(N)

    # First fundamental form coefficients
    E = jnp.dot(Su, Su)
    F = jnp.dot(Su, Sv)
    G = jnp.dot(Sv, Sv)

    # Second fundamental form coefficients
    e = jnp.dot(Suu, N)
    f = jnp.dot(Suv, N)
    g = jnp.dot(Svv, N)

    # Gaussian curvature: K = (eg - f²) / (EG - F²)
    denom = E * G - F**2
    K = (e * g - f**2) / denom

    # Mean curvature: H = (eG - 2fF + gE) / (2(EG - F²))
    H = (e * G - 2 * f * F + g * E) / (2 * denom)

    # Principal curvatures from K and H: roots of λ² - 2Hλ + K = 0
    discriminant = jnp.sqrt(jnp.maximum(H**2 - K, 0))
    kappa1 = H + discriminant
    kappa2 = H - discriminant

    return K, H, kappa1, kappa2


# Test on unit sphere (should have K = 1, H = 1)
def sphere_param(u, v, R=1.0):
    """Sphere parametrised by (u, v) = (θ, φ) ∈ (0,π) × [0, 2π)."""
    return jnp.array([
        R * jnp.sin(u) * jnp.cos(v),
        R * jnp.sin(u) * jnp.sin(v),
        R * jnp.cos(u)
    ])

K, H, k1, k2 = surface_curvatures(sphere_param, 1.0, 0.5)
print(f"Unit sphere curvatures:")
print(f"  Gaussian K = {K:.4f} (expected: 1.0)")
print(f"  Mean H = {H:.4f} (expected: 1.0)")
print(f"  Principal: κ₁ = {k1:.4f}, κ₂ = {k2:.4f} (expected: 1.0, 1.0)")

# Test on saddle z = xy (should have K < 0 at origin)
def saddle(u, v):
    return jnp.array([u, v, u * v])

K, H, k1, k2 = surface_curvatures(saddle, 0.0, 0.0)
print(f"\nSaddle at origin:")
print(f"  Gaussian K = {K:.4f} (expected: -1.0)")
print(f"  Mean H = {H:.4f} (expected: 0.0)")
print(f"  Principal: κ₁ = {k1:.4f}, κ₂ = {k2:.4f}")
```

**Exercise**: Create a gallery of surfaces colored by Gaussian
curvature. Include: sphere, torus, hyperboloid, catenoid (minimal
surface with H = 0).

### Month 4: Intrinsic geometry and the Theorema Egregium

**Primary text: Do Carmo, Chapter 4 (sections 4-1 through 4-5)**

The most profound result in surface theory: Gaussian curvature depends
only on the first fundamental form — only on measurements made within
the surface, not on how it's embedded in space.

**Week 1-2: Isometries and the intrinsic**

- Isometries: maps preserving the first fundamental form
- What properties are intrinsic (detectable from within)?
- Christoffel symbols and the intrinsic covariant derivative

**Week 3: The Theorema Egregium**

- Gauss's "remarkable theorem": K is intrinsic
- The formula for K in terms of E, F, G and their derivatives
- Why this is surprising and profound

**Week 4: Gauss-Bonnet theorem**

- Local Gauss-Bonnet: angle excess in geodesic triangles
- Global Gauss-Bonnet: ∫∫_S K dA = 2πχ(S)
- Topology from geometry: Euler characteristic determined by curvature

**Key insight**: Gauss-Bonnet connects local geometry (curvature) to
global topology (Euler characteristic). Total curvature is a
topological invariant — it doesn't change under continuous
deformations. This is the first glimpse of a pattern that recurs
throughout differential geometry: local differential invariants
constraining global topological structure.

**Common sticking point**: The proof of Theorema Egregium involves
computing Gaussian curvature purely from Christoffel symbols, which
derive from the metric. The calculation is technical — many indices,
many terms. Focus first on understanding *why* the result is
remarkable: you can detect curvature without leaving the surface. Then
work through the calculation, perhaps with computer assistance.

**Exercises**: Do Carmo Chapter 4, problems 1-4, 7, 9.

**Computation**: Verify Theorema Egregium by computing K two ways.

```python
def gaussian_curvature_intrinsic(surface, u, v):
    """
    Compute K purely from the metric (first fundamental form) using autodiff.
    This demonstrates the Theorema Egregium: K depends only on E, F, G
    and their derivatives — no reference to the embedding.

    Uses the Brioschi formula for Gaussian curvature with JAX derivatives.
    """
    def metric_coeffs(uv):
        """Return (E, F, G) as a function of (u, v) for differentiation."""
        u, v = uv[0], uv[1]
        def surf(uv_inner):
            return surface(uv_inner[0], uv_inner[1])
        J = jax.jacfwd(surf)(jnp.array([u, v]))
        Su, Sv = J[:, 0], J[:, 1]
        E = jnp.dot(Su, Su)
        F = jnp.dot(Su, Sv)
        G = jnp.dot(Sv, Sv)
        return jnp.array([E, F, G])

    uv = jnp.array([u, v])
    EFG = metric_coeffs(uv)
    E, F, G = EFG[0], EFG[1], EFG[2]

    # First derivatives via autodiff
    d_EFG = jax.jacfwd(metric_coeffs)(uv)  # Shape (3, 2): d(E,F,G)/d(u,v)
    E_u, E_v = d_EFG[0, 0], d_EFG[0, 1]
    F_u, F_v = d_EFG[1, 0], d_EFG[1, 1]
    G_u, G_v = d_EFG[2, 0], d_EFG[2, 1]

    # Second derivatives via autodiff
    dd_EFG = jax.jacfwd(jax.jacfwd(metric_coeffs))(uv)  # Shape (3, 2, 2)
    E_vv = dd_EFG[0, 1, 1]
    G_uu = dd_EFG[2, 0, 0]
    F_uv = dd_EFG[1, 0, 1]

    # Brioschi formula for K
    W = E * G - F**2

    K = (2 * F_uv - E_vv - G_uu) / (2 * W) + (
        (E_v * G_v - 2 * F_u * G_v + G_u**2) * E +
        (E_u * G_v - E_v * G_u) * F +
        (E_u * G_u - 2 * E_v * F_u + E_v**2) * G
    ) / (4 * W**2)

    return K


# Verify Theorema Egregium
print("Verifying Theorema Egregium:")
print("(K computed extrinsically should equal K computed intrinsically)\n")

for name, surf, u, v in [
    ("Sphere", sphere_param, 1.0, 0.5),
    ("Torus", torus, 0.5, 1.0),
    ("Saddle", saddle, 0.1, 0.1),  # Avoid exact origin for numerical stability
]:
    K_ext = gaussian_curvature_extrinsic(surf, u, v)
    K_int = gaussian_curvature_intrinsic(surf, u, v)
    print(f"{name}:")
    print(f"  Extrinsic K = {K_ext:.4f}")
    print(f"  Intrinsic K = {K_int:.4f}")
    print(f"  Agreement: {jnp.allclose(K_ext, K_int, rtol=1e-3)}\n")
```

### Celebrating Phase 1

If you've reached this point, pause and acknowledge it. You've rebuilt
your geometric intuition from curves through surfaces to the Theorema
Egregium. You understand curvature both extrinsically and
intrinsically. You've implemented these ideas in code.

Take a day off. Tell someone what you've learned. Then continue.

### Spiral Return 1 (end of Month 4, ~1 week)

Spend one week consolidating. This is not optional.

**Synthesis exercise**: Take a surface you know well (say, the torus).
Work through every concept:

- Parametrise it explicitly
- Compute E, F, G at a generic point
- Compute the Gaussian and mean curvature
- Identify the elliptic, parabolic, and hyperbolic points
- Compute the Euler characteristic from Gauss-Bonnet (it's 0)
- Explain *why* it's 0 in terms of the geometry

**Write**: A 2000-word essay titled "Intrinsic vs Extrinsic: What
Surfaces Teach Us About Geometry." This should explain, to an
intelligent reader unfamiliar with the material, why the Theorema
Egregium is profound and what it tells us about the nature of
geometry. This essay previews the manifold philosophy.

**Content retrospective**: Look back at the content you created this
phase. What worked? What would you do differently? How has explaining
changed your understanding?

**Phase 1 Capstone: The sphere as surface**

Great circles are geodesics of the sphere. Verify this:

- Parametrise a great circle as a curve in ℝ³
- Compute its curvature κ and torsion τ using Frenet-Serret
- Show κ = 1/R (constant), τ = 0 (planar)
- Compute the Gaussian curvature K of the sphere; verify K = 1/R²
- Use Gauss-Bonnet: ∫∫ K dA = 4π = 2πχ(S²), confirming χ = 2

*You see*: geodesics as curves, curvature as a number.

### Phase 1 Content Creation

**Primary deliverable: Video — "Why You Can't Flatten a Sphere" (15-20
minutes)**

*Time budget: 2-3 weeks of part-time work*

A note on video production: If this is your first video, expect it to
take longer than estimated. Consider starting with a simple format —
talking head with screen recording and hand-drawn diagrams — rather
than complex animations. Clarity of explanation matters more than
production polish. You can develop more sophisticated production
skills over time.

Explain the Theorema Egregium and Gauss-Bonnet to an intelligent
non-mathematician. Structure:

- 0:00-3:00 — Hook: the pizza slice trick (why flat pizza slices don't
  droop when you fold them)
- 3:00-7:00 — What curvature means (positive, negative, zero with
  visual examples)
- 7:00-12:00 — The remarkable theorem: curvature is intrinsic (can't
  change by bending without stretching)
- 12:00-16:00 — Why every map of Earth is wrong (geometry forces
  distortion)
- 16:00-18:00 — Gauss-Bonnet: total curvature determines topology
- 18:00-20:00 — So what? Connection to general relativity and modern
  geometry

**Secondary deliverable: Blog post — "Implementing Frenet-Serret in
JAX"**

*Time budget: 1 week*

A technical walkthrough of your curve computations:

- The mathematical theory (clear, self-contained)
- The implementation (clean, documented code)
- Visualisations (moving frame along various curves)
- What you learned from implementing it (insights that came from
  coding)

### Phase 1 Mastery Checkpoints

Before proceeding, verify you can:

- [ ] Derive the Frenet-Serret equations starting from the definition
      of T, κ, N, τ, B
- [ ] Compute E, F, G for a given parametric surface
- [ ] Use I to compute the length of a curve on a surface
- [ ] Explain the shape operator in one sentence without jargon
- [ ] Compute Gaussian curvature K for sphere, cylinder, torus, and
      saddle
- [ ] State the Theorema Egregium and give a one-paragraph explanation
      of why it's remarkable
- [ ] State Gauss-Bonnet and use it to compute χ for a surface from
      its curvature
- [ ] All code from this phase runs correctly and you understand every
      line

---

## Phase 2: Smooth manifolds — the abstract turn

**Duration: 5 months (Months 5-9)**

*Goal: Master the coordinate-free language of modern differential
geometry.*

### Month 5: Foundations of manifolds

**Primary text: Tu, *An Introduction to Manifolds*, Chapters 1-7**

This is where geometry becomes abstract. A manifold is a space that
looks locally like ℝⁿ. The power comes from how local pieces glue
together globally.

**Week 1-2: Smooth functions and Euclidean space**

- Smooth functions on ℝⁿ
- The derivative as a linear map: Df_p: ℝⁿ → ℝᵐ
- Taylor's theorem with remainder

**Week 3-4: Manifolds and smooth structure**

- Topological manifolds: locally Euclidean, Hausdorff, second
  countable
- Charts (φ, U), atlases, smooth compatibility
- Examples: spheres Sⁿ, projective spaces ℝPⁿ, Lie groups

**Key insight**: A manifold doesn't live in any ambient space. It's
defined intrinsically by its atlas of charts. Surfaces in ℝ³ are
manifolds, but manifolds need not be embedded anywhere. This is the
abstraction of "intrinsic geometry" from Phase 1, taken to its logical
conclusion.

**Common sticking point**: The quotient manifold construction (e.g.,
ℝPⁿ = Sⁿ/{±1}) requires comfort with equivalence relations and
quotient topology. If this feels shaky, review quotient spaces in
Munkres or work through the construction of ℝP² very carefully.

**Transcribe**: Schuller Lectures 1-3 (Logic, Set Theory, Topology).
These provide the foundational precision Tu assumes.

**Exercises**: Tu, end-of-chapter problems for Chapters 1-7 (selected:
focus on problems marked with * or that involve explicit
constructions).

### Month 6: Tangent spaces and bundles

**Primary text: Tu, Chapters 8-12**

The tangent space at a point is the fundamental object. It's where
velocities live, where derivatives land, where physics happens.

**Week 1-2: Tangent vectors as derivations**

- Tangent vectors as equivalence classes of curves (intuitive)
- Tangent vectors as derivations on germs (rigorous)
- The tangent space T_pM as a vector space of dimension n = dim M

**Week 3-4: The tangent bundle**

- TM = ∐_{p∈M} T_pM as a manifold of dimension 2n
- Vector fields as smooth sections X: M → TM with π ∘ X = id
- The cotangent bundle T*M (dual spaces at each point)

**Key insight**: A tangent vector at p is not a little arrow — it's an
operator that eats smooth functions and outputs numbers, satisfying
the Leibniz rule: v(fg) = v(f)g(p) + f(p)v(g). This definition is
coordinate-free and doesn't mention curves. The equivalence with the
"velocity of curves" definition is a theorem, not a definition.

**Computation**: Forward-mode automatic differentiation computes the
pushforward (differential).

```python
import jax
import jax.numpy as jnp

def pushforward_demonstration():
    """
    The pushforward (differential) of a smooth map f: M → N is
    a linear map df_p: T_pM → T_{f(p)}N between tangent spaces.

    Forward-mode AD computes exactly this:
    - Input: point p, tangent vector v ∈ T_pM
    - Output: f(p), and df_p(v) ∈ T_{f(p)}N

    This is what JAX's jvp (Jacobian-vector product) does.
    """
    # A smooth map f: ℝ² → ℝ³
    def f(x):
        return jnp.array([
            x[0]**2 + x[1],           # x² + y
            jnp.sin(x[0]) * x[1],     # sin(x) · y
            jnp.exp(x[0] - x[1])      # e^{x-y}
        ])

    # Point p ∈ ℝ² (think of ℝ² as our manifold M)
    p = jnp.array([1.0, 2.0])

    # Tangent vector v ∈ T_pM ≅ ℝ²
    v = jnp.array([0.3, -0.1])

    # Compute pushforward via jvp
    f_p, df_v = jax.jvp(f, (p,), (v,))

    print("Pushforward (differential) demonstration:")
    print(f"  Point p = {p}")
    print(f"  Tangent vector v ∈ T_pM: {v}")
    print(f"  f(p) = {f_p}")
    print(f"  df_p(v) ∈ T_{{f(p)}}N: {df_v}")

    # Verify: df_p(v) = J @ v where J is the Jacobian matrix
    J = jax.jacobian(f)(p)
    print(f"\n  Verification: Jacobian @ v = {J @ v}")
    print(f"  Match: {jnp.allclose(df_v, J @ v)}")

    # Key insight: jvp never forms the full Jacobian matrix.
    # It computes the action of J on v directly — efficient for high-dim M.

pushforward_demonstration()
```

```python
def pullback_demonstration():
    """
    The pullback of a map f: M → N acts on cotangent vectors (1-forms):
    f*: T*_{f(p)}N → T*_pM

    Reverse-mode AD computes the pullback:
    - Input: point p, cotangent vector α ∈ T*_{f(p)}N
    - Output: f(p), and f*(α) ∈ T*_pM

    This is what JAX's vjp (vector-Jacobian product) does.
    """
    def f(x):
        return jnp.array([
            x[0]**2 + x[1],
            jnp.sin(x[0]) * x[1],
            jnp.exp(x[0] - x[1])
        ])

    p = jnp.array([1.0, 2.0])

    # Cotangent vector α ∈ T*_{f(p)}N ≅ ℝ³
    # (Think of this as coefficients of a 1-form at f(p))
    alpha = jnp.array([1.0, 0.5, -0.2])

    # Compute pullback via vjp
    f_p, vjp_fun = jax.vjp(f, p)
    f_star_alpha, = vjp_fun(alpha)  # Note comma: vjp returns tuple

    print("Pullback demonstration:")
    print(f"  Point p = {p}")
    print(f"  f(p) = {f_p}")
    print(f"  Cotangent vector α ∈ T*_{{f(p)}}N: {alpha}")
    print(f"  f*(α) ∈ T*_pM: {f_star_alpha}")

    # Verify: f*(α) = Jᵀ @ α
    J = jax.jacobian(f)(p)
    print(f"\n  Verification: Jᵀ @ α = {J.T @ alpha}")
    print(f"  Match: {jnp.allclose(f_star_alpha, J.T @ alpha)}")

pullback_demonstration()
```

**Key insight**: Forward-mode AD (jvp) implements the pushforward df: TM
→ TN. Reverse-mode AD (vjp) implements the pullback f*: T*N → T*M.
This is not analogy — it is mathematical identity. The chain rule in
calculus is the functoriality of the tangent bundle construction.

**Exercises**: Tu, end-of-chapter problems for Chapters 8-12.

### Month 7: Vector fields and flows

**Primary text: Tu, Chapters 14-16**

A vector field assigns a tangent vector to each point. Its integral
curves are solutions to the ODE "follow the vector field." The flow is
the family of all solutions.

**Week 1-2: Vector fields and integral curves**

- Vector fields as sections X: M → TM
- Integral curves: γ'(t) = X(γ(t))
- Local existence and uniqueness (Picard-Lindelöf)

**Week 3-4: Flows and Lie brackets**

- One-parameter groups of diffeomorphisms φ_t: M → M
- The Lie bracket [X, Y] of vector fields
- Interpretation: [X, Y] measures failure of flows to commute

**Key insight**: Flow along X for time ε, then along Y for ε, then back
along X for ε, then back along Y for ε. You don't return to where you
started. The discrepancy is approximately ε²[X, Y]. The Lie bracket
captures this infinitesimal non-commutativity.

**Visual anchor**: Draw two vector fields X and Y on a small patch of
the plane. From a point p, flow along X for small time ε to reach
point a. From a, flow along Y for time ε to reach b. Now go back: from
p, flow along Y first to reach c, then along X to reach d. If X and Y
were coordinate vector fields (∂/∂x and ∂/∂y), you'd have b = d — a
closed parallelogram. For general vector fields, b ≠ d. The gap
between them is approximately ε²[X,Y]. The Lie bracket measures how
much the parallelogram fails to close. Arnold's *Mathematical Methods
of Classical Mechanics*, §34, has the canonical figure. Sketch it
yourself for X = ∂/∂x and Y = x·∂/∂y on ℝ².

**Computation**: Visualise flows and compute Lie brackets.

```python
import jax
import jax.numpy as jnp
import diffrax

def flow_of_vector_field(X, p0, t_span, dt=0.01):
    """
    Compute the flow φ_t(p0) of vector field X starting at p0.
    Returns trajectory: array of shape (num_steps, dim).
    """
    def dynamics(t, y, args):
        return X(y)

    t0, t1 = t_span
    ts = jnp.linspace(t0, t1, int((t1 - t0) / dt))

    term = diffrax.ODETerm(dynamics)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=ts)

    solution = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=dt, y0=p0, saveat=saveat
    )
    return solution.ts, solution.ys


def lie_bracket(X, Y, p):
    """
    Compute [X, Y] at point p.

    In coordinates: [X, Y]^i = X^j ∂_j Y^i - Y^j ∂_j X^i

    Or equivalently: [X, Y] = JY · X - JX · Y
    where JX, JY are the Jacobians of X and Y.
    """
    JX = jax.jacobian(X)(p)  # Jacobian of X at p
    JY = jax.jacobian(Y)(p)  # Jacobian of Y at p

    # [X, Y] = JY @ X(p) - JX @ Y(p)
    return JY @ X(p) - JX @ Y(p)


# Example: two vector fields on ℝ²
def X(p):
    """X = ∂/∂x (translation in x)."""
    return jnp.array([1.0, 0.0])

def Y(p):
    """Y = x ∂/∂y (shear)."""
    return jnp.array([0.0, p[0]])

# The Lie bracket [X, Y]:
# X = ∂/∂x, Y = x ∂/∂y
# [X, Y] = X(x) ∂/∂y - Y(1) · 0 = 1 · ∂/∂y = ∂/∂y
# So [X, Y] should be (0, 1)

p = jnp.array([2.0, 1.0])
bracket = lie_bracket(X, Y, p)
print(f"Lie bracket [X, Y] at p = {p}:")
print(f"  Computed: {bracket}")
print(f"  Expected: [0, 1]")


def commutator_deficit(X, Y, p0, eps):
    """
    Flow around a small square and measure the deficit.
    The deficit should be approximately ε² [X, Y](p0).
    """
    # Flow X for time eps
    _, traj1 = flow_of_vector_field(X, p0, (0, eps), dt=eps/10)
    p1 = traj1[-1]

    # Flow Y for time eps
    _, traj2 = flow_of_vector_field(Y, p1, (0, eps), dt=eps/10)
    p2 = traj2[-1]

    # Flow -X for time eps
    neg_X = lambda p: -X(p)
    _, traj3 = flow_of_vector_field(neg_X, p2, (0, eps), dt=eps/10)
    p3 = traj3[-1]

    # Flow -Y for time eps
    neg_Y = lambda p: -Y(p)
    _, traj4 = flow_of_vector_field(neg_Y, p3, (0, eps), dt=eps/10)
    p4 = traj4[-1]

    return p4 - p0

eps = 0.1
deficit = commutator_deficit(X, Y, p, eps)
expected = eps**2 * lie_bracket(X, Y, p)
print(f"\nCommutator deficit test (ε = {eps}):")
print(f"  Actual deficit: {deficit}")
print(f"  Expected (ε²[X,Y]): {expected}")
print(f"  Ratio: {deficit / (expected + 1e-10)}")  # Should be close to 1
```

**Transcribe**: Schuller Lectures 7-9 (Tangent spaces, Tensor spaces,
Grassmann algebra).

**Exercises**: Tu, end-of-chapter problems for Chapters 14-16.

### Month 8: Differential forms

**Primary text: Tu, Chapters 17-21**

Differential forms are what you integrate. They unify gradient, curl,
divergence into a single operator: the exterior derivative d.

**Week 1-2: Differential k-forms**

- 1-forms as dual to vector fields: ω ∈ Ω¹(M) eats vector fields
- k-forms via wedge product: Ωᵏ(M) = sections of ∧ᵏT*M
- The exterior algebra Ω*(M) = ⊕ₖ Ωᵏ(M)

**Week 3-4: The exterior derivative**

- d: Ωᵏ(M) → Ωᵏ⁺¹(M), uniquely determined by:
  - d(f) = df for functions (the differential)
  - d(dω) = 0 for all ω
  - d(α ∧ β) = dα ∧ β + (-1)^{deg α} α ∧ dβ (graded Leibniz)
- Closed forms (dω = 0), exact forms (ω = dη)
- Poincaré lemma: on contractible domains, closed ⟹ exact

**Key insight**: The exterior derivative d generalises all classical
differential operators:
- On 0-forms (functions): df is the gradient (as a 1-form)
- On 1-forms in ℝ³: *d* gives the curl
- On 2-forms in ℝ³: *d* gives the divergence

And d² = 0 encodes both "curl of gradient = 0" and "div of curl = 0."

**Common sticking point**: The coordinate-free definition of d can
feel like magic. Work through the coordinate formula first, verify d²
= 0 by explicit computation, then return to appreciate why the
coordinate-free axioms uniquely determine d.

**Computation**: Implement exterior algebra and exterior derivative.

```python
import sympy
from sympy import symbols, Function, diff, simplify, sin, cos, exp

def exterior_derivative_demo():
    """
    Demonstrate the exterior derivative on ℝ³.
    """
    x, y, z = symbols('x y z', real=True)

    # A 0-form (function)
    f = x**2 * y + sin(z)

    # df is a 1-form: df = (∂f/∂x)dx + (∂f/∂y)dy + (∂f/∂z)dz
    df_coeffs = [diff(f, x), diff(f, y), diff(f, z)]
    print("0-form f =", f)
    print("df =", df_coeffs[0], "dx +", df_coeffs[1], "dy +", df_coeffs[2], "dz")

    # A 1-form ω = P dx + Q dy + R dz
    P, Q, R = x*y, y*z, z*x

    # dω = (∂R/∂y - ∂Q/∂z) dy∧dz + (∂P/∂z - ∂R/∂x) dz∧dx + (∂Q/∂x - ∂P/∂y) dx∧dy
    # This is the curl!
    dw_1 = diff(R, y) - diff(Q, z)  # coefficient of dy∧dz
    dw_2 = diff(P, z) - diff(R, x)  # coefficient of dz∧dx
    dw_3 = diff(Q, x) - diff(P, y)  # coefficient of dx∧dy

    print("\n1-form ω =", P, "dx +", Q, "dy +", R, "dz")
    print("dω =", dw_1, "dy∧dz +", dw_2, "dz∧dx +", dw_3, "dx∧dy")

    # Verify d² = 0: d(df) = 0
    # df = 2xy dx + x² dy + cos(z) dz
    # d(df) involves ∂²f/∂x∂y - ∂²f/∂y∂x = 0 by symmetry of mixed partials
    ddx = diff(df_coeffs[0], y) - diff(df_coeffs[1], x)
    ddy = diff(df_coeffs[1], z) - diff(df_coeffs[2], y)
    ddz = diff(df_coeffs[2], x) - diff(df_coeffs[0], z)
    print("\nVerify d² = 0:")
    print("d(df) has coefficients:", simplify(ddx), simplify(ddy), simplify(ddz))
    print("All zero: d² = 0 ✓")

exterior_derivative_demo()
```

**Read**: Spivak Vol. 1, Chapter 7 for beautiful historical
perspective on differential forms. This is where to understand *why*
forms are the natural objects to integrate.

**Transcribe**: Schuller Lectures 10-11 (Differential forms).

### Month 9: Integration and de Rham cohomology

**Primary text: Tu, Chapters 22-24**

Stokes' theorem is the fundamental theorem of calculus in all
dimensions.

**Week 1-2: Integration of forms**

- Orientation of manifolds
- Integration of n-forms on n-manifolds
- Pullback and change of variables

**Week 3: Stokes' theorem**

- The statement: ∫_M dω = ∫_{∂M} ω
- Recovering classical theorems: Green, Divergence, Classical Stokes
- The boundary ∂ and the exterior derivative d are adjoint

**Week 4: De Rham cohomology**

- H^k_{dR}(M) = {closed k-forms} / {exact k-forms}
- What cohomology measures: "k-dimensional holes"
- H^0(M) = ℝ^{#components}, H^1(S¹) = ℝ, H^2(S²) = ℝ

*The topological intuition*: If you've seen algebraic topology,
cohomology is the dual picture to homology. Homology counts holes
(cycles that don't bound); cohomology counts the differential forms
that "get stuck" on those holes (closed forms that aren't exact). The
torus has two independent 1-dimensional holes (around and through), so
H¹(T²) = ℝ². You don't need algebraic topology to proceed — de Rham
cohomology stands on its own — but if the "hole-counting" intuition
helps, use it.

**Key insight**: Stokes' theorem says that d and ∂ are adjoint operators
in a precise sense. Integration pairs forms with chains (oriented
submanifolds). Cohomology detects when closed forms fail to be exact —
this happens precisely when there are topological obstructions.

**Concrete experience with cohomology**: To make cohomology feel real,
work through H¹(T²) for the torus explicitly. Consider the 1-form dθ
on the torus (where θ is the "around the hole" angle). Verify it's
closed: d(dθ) = 0. Is it exact? If dθ = df for some function f, then
f(θ + 2π) = f(θ) + 2π, which is impossible for a single-valued
function. The 1-form dθ detects the hole in the torus. This takes
perhaps two hours but delivers genuine understanding that abstract
definitions cannot.

**Exercises**: Tu, end-of-chapter problems for Chapters 22-24.

### Celebrating Phase 2

You've made the abstract turn. You now speak the language of modern
differential geometry: manifolds, tangent bundles, forms, cohomology.
This is a substantial achievement.

Take a day to rest. Go for a walk. Let the ideas settle. Then
continue.

### Spiral Return 2 (end of Month 9, ~1.5 weeks)

**Return to surfaces with full manifold language:**

A surface S ⊂ ℝ³ is a 2-dimensional embedded submanifold of the
3-manifold ℝ³. Now reinterpret everything:

| Classical concept | Manifold interpretation |
|-------------------|------------------------|
| Tangent plane T_pS | Tangent space of the manifold |
| First fundamental form I | Pullback i*g of Euclidean metric via inclusion i: S → ℝ³ |
| Gauss map N: S → S² | Smooth map between 2-manifolds |
| Area element dA | The 2-form induced by the metric |
| Gauss-Bonnet ∫K dA = 2πχ | Integration of curvature 2-form |

**Concrete exercise**: Take the torus T² = S¹ × S¹.
1. Write down an explicit atlas (two charts suffice locally, four
   charts for a global atlas avoiding overlap issues).
2. Compute the tangent space at a point using both definitions: (a) as
   equivalence classes of curves, (b) as derivations.
3. Verify they give the same 2-dimensional space.
4. Write the first fundamental form as a section of Sym²(T*T²).

**Write**: A 3000-word essay titled "From Surfaces to Manifolds: The
Same Ideas, Properly Said." This should explain to a reader who has
finished Phase 1 why the abstract framework is worth the trouble.

**Content retrospective**: Review your content from this phase. What
explanations worked best? Where did the audience (real or imagined)
get stuck? What would you do differently?

**Phase 2 Capstone: The sphere as manifold**

Now see the sphere abstractly:

- Write an atlas for S² (stereographic projection from north and south
  poles)
- Verify the transition function is smooth
- Express a great circle as a curve γ: ℝ → S² in each chart
- The velocity γ'(t) lives in T_{γ(t)}S² — visualise this
- Write the vector field on TS² whose integral curves are geodesics
  (you'll need the geodesic equation, which you'll derive properly in
  Phase 3)

*You see*: the sphere as intrinsic object, geodesics as flows.

### Phase 2 Content Creation

**Primary deliverable: Video — "Automatic Differentiation IS
Differential Geometry" (20-25 minutes)**

*Time budget: 3-4 weeks*

This is your signature piece for this phase. Structure:

- 0:00-3:00 — Hook: AD libraries compute something geometric without
  knowing it
- 3:00-8:00 — The tangent bundle: what tangent vectors really are
- 8:00-12:00 — Forward mode = pushforward: df: TM → TN
- 12:00-16:00 — Reverse mode = pullback: f*: T*N → T*M
- 16:00-20:00 — The chain rule is functoriality
- 20:00-23:00 — Demo: implementing pushforward in JAX
- 23:00-25:00 — Why this matters: geometry guides algorithm design

Include live coding. This could anchor a conference talk.

**Secondary deliverable: First Principles Club session — "Differential
Forms and Integration"**

*Time budget: 2 weeks preparation*

**Session structure (90 minutes):**

- 0:00-5:00 — Welcome, context setting
- 5:00-25:00 — Motivation: why are forms natural? The
  change-of-variables puzzle, orientation
- 25:00-50:00 — The algebra: wedge product, exterior derivative, d² =
  0
- 50:00-70:00 — The punchline: Stokes' theorem as the master theorem
  of calculus
- 70:00-85:00 — Hands-on: compute dω for specific examples, verify
  Stokes on a square
- 85:00-90:00 — Wrap-up, questions, pointers to resources

**Tertiary deliverable: Blog post — "De Rham Cohomology: Detecting
Holes with Calculus"**

*Time budget: 1 week*

Explain cohomology to a programmer:

- The setup: closed forms, exact forms, the quotient
- Why closed ≠ exact on the circle (wind around and integrate)
- Simple examples: H^1(S¹) = ℝ, H^2(S²) = ℝ, H^1(T²) = ℝ²
- The punchline: differential equations have global obstructions

### Phase 2 Mastery Checkpoints

Before proceeding, verify you can:

- [ ] Define smooth manifold precisely: charts, atlas, smooth
      compatibility condition
- [ ] Define tangent vector as a derivation, state the Leibniz rule it
      satisfies
- [ ] Explain why dim(TM) = 2·dim(M) and why TM is itself a manifold
- [ ] Compute the Lie bracket [X, Y] of two explicit vector fields
- [ ] Give the geometric interpretation of [X, Y] in terms of flows
      (one sentence)
- [ ] Define k-form, wedge product, exterior derivative
- [ ] Prove d² = 0 starting from the axioms of d
- [ ] State Stokes' theorem; derive Green's theorem as a special case
- [ ] Explain what H^1(M) measures and give an example where it's
      nonzero
- [ ] Implement pushforward via jvp and explain why it works
      geometrically

---

## Phase 3: Riemannian geometry — metric and curvature

**Duration: 5 months (Months 10-14)**

*Goal: Understand how metric structure enables measurement and how
curvature emerges from parallel transport.*

### Month 10: Riemannian metrics

**Primary text: Lee, *Introduction to Riemannian Manifolds* (2nd ed.,
2018), Chapters 1-3**

A Riemannian metric is a smoothly varying inner product on each
tangent space. It lets you measure lengths, angles, and volumes.

**Week 1-2: Riemannian manifolds**

- Riemannian metric g: a smooth section of Sym²(T*M) that's positive
  definite
- The metric in local coordinates: g = g_{ij} dx^i ⊗ dx^j
- Lengths of curves, Riemannian distance, isometries

**Week 3-4: Model spaces**

- Euclidean space (ℝⁿ, δ_{ij})
- Spheres Sⁿ with round metric (constant positive curvature)
- Hyperbolic space Hⁿ (constant negative curvature): multiple models
  (hyperboloid, Poincaré ball, upper half-space)

**Key insight**: The metric g is the fundamental datum. Everything else
— lengths, angles, volumes, geodesics, curvature — derives from it.
Two Riemannian manifolds with the same metric are geometrically
indistinguishable (isometric).

**Read**: Frankel, Chapters 9-10 for physical motivation: how metrics
appear in mechanics, relativity, and continuum mechanics.

**Computation**: Work with metrics in Geomstats.

```python
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere

def compare_geometries():
    """
    Compare distances in Euclidean, spherical, and hyperbolic geometry.
    Same coordinate distance, different geometric distance.
    """
    # Three 2-dimensional geometries
    E2 = Euclidean(dim=2)
    S2 = Hypersphere(dim=2)  # Embedded in ℝ³
    H2 = Hyperbolic(dim=2)   # Poincaré ball model

    # Two points in the Euclidean plane
    p1_euc = jnp.array([0.0, 0.0])
    p2_euc = jnp.array([0.5, 0.0])

    # Corresponding points on the sphere
    p1_sph = jnp.array([0.0, 0.0, 1.0])  # North pole
    p2_sph = jnp.array([jnp.sin(0.5), 0.0, jnp.cos(0.5)])  # 0.5 radians away

    # Points in Poincaré ball (same coordinates as Euclidean)
    p1_hyp = jnp.array([0.0, 0.0])
    p2_hyp = jnp.array([0.5, 0.0])

    print("Distance comparison:\n")
    print(f"Euclidean distance: {E2.metric.dist(p1_euc, p2_euc):.4f}")
    print(f"Spherical distance: {S2.metric.dist(p1_sph, p2_sph):.4f}")
    print(f"Hyperbolic distance: {H2.metric.dist(p1_hyp, p2_hyp):.4f}")

    print("\nNote: hyperbolic distance > Euclidean > spherical")
    print("Hyperbolic space 'stretches' near the boundary of the disk.")

compare_geometries()
```

### Month 11: Connections and covariant derivatives

**Primary text: Lee, Chapters 4-5**

How do you compare vectors at different points? You can't just
subtract them — they live in different vector spaces. A connection
tells you how to "connect" nearby tangent spaces.

**Week 1-2: Connections on vector bundles**

- The problem: V ∈ T_pM and W ∈ T_qM live in different spaces
- Connections as covariant derivative: ∇: 𝔛(M) × Γ(E) → Γ(E)
- Properties: ℝ-linear in Y, tensorial in X, Leibniz rule

**Week 3-4: The Levi-Civita connection**

- Metric compatibility: ∇g = 0 (parallel transport preserves inner
  products)
- Torsion-freeness: ∇_X Y - ∇_Y X = [X, Y]
- **Fundamental Theorem**: Given a Riemannian metric g, there exists a
  unique connection satisfying both. This is the Levi-Civita
  connection.
- Christoffel symbols Γ^i_{jk}: the components of ∇_{∂_j}∂_k =
  Γ^i_{jk} ∂_i

**Key insight**: The Levi-Civita connection is determined by the metric
alone. The Christoffel symbols Γ^i_{jk} that appear in physics (GR)
are exactly the components of this connection. The formula

$$\Gamma^i_{jk} = \frac{1}{2} g^{il} \left( \partial_j g_{kl} +
\partial_k g_{jl} - \partial_l g_{jk} \right)$$

shows the connection is built from the metric and its first
derivatives.

**Transcribe**: Schuller Lectures 15-17 (Connections, Parallel
transport, Curvature). Schuller's treatment is among the most rigorous
available.

**Computation**: Compute Christoffel symbols from a metric.

```python
import jax
import jax.numpy as jnp

def christoffel_symbols(metric_fn, x):
    """
    Compute Christoffel symbols Γ^i_{jk} from a metric function.

    Convention: Γ^i_{jk} = (1/2) g^{il} (∂_j g_{kl} + ∂_k g_{jl} - ∂_l g_{jk})

    The upper index i is the output direction.
    The lower indices j, k are the differentiation direction and input field.

    Args:
        metric_fn: function x ↦ g_{ij}(x), returns (n, n) matrix
        x: point, shape (n,)

    Returns:
        Gamma: array of shape (n, n, n), where Gamma[i, j, k] = Γ^i_{jk}
    """
    n = len(x)

    # Metric and its inverse
    g = metric_fn(x)
    g_inv = jnp.linalg.inv(g)

    # Partial derivatives of metric: dg[l, i, j] = ∂_l g_{ij}
    dg = jax.jacfwd(metric_fn)(x)  # Shape: (n, n, n)
    # jacfwd gives dg[i, j, l] = ∂g_{ij}/∂x^l, so transpose
    dg = jnp.transpose(dg, (2, 0, 1))  # Now dg[l, i, j] = ∂_l g_{ij}

    # Christoffel symbols via einsum for clarity
    # Γ^i_{jk} = (1/2) g^{il} (∂_j g_{kl} + ∂_k g_{jl} - ∂_l g_{jk})
    Gamma = jnp.zeros((n, n, n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                val = 0.0
                for l in range(n):
                    val += g_inv[i, l] * (dg[j, k, l] + dg[k, j, l] - dg[l, j, k])
                Gamma = Gamma.at[i, j, k].set(0.5 * val)

    return Gamma


# Example: Poincaré half-plane model of hyperbolic space
# Metric: ds² = (dx² + dy²) / y²

def poincare_half_plane_metric(x):
    """Metric for Poincaré half-plane at point x = (x, y)."""
    y = x[1]
    return jnp.array([[1/y**2, 0], [0, 1/y**2]])

point = jnp.array([1.0, 2.0])  # Point in upper half-plane
Gamma = christoffel_symbols(poincare_half_plane_metric, point)

print("Christoffel symbols for Poincaré half-plane at", point)
print(f"Γ^0_{{01}} = Γ^0_{{10}} = {Gamma[0,0,1]:.4f} (expected: -1/y = -0.5)")
print(f"Γ^1_{{00}} = {Gamma[1,0,0]:.4f} (expected: 1/y = 0.5)")
print(f"Γ^1_{{11}} = {Gamma[1,1,1]:.4f} (expected: -1/y = -0.5)")
```

### Month 12: Geodesics and the exponential map

**Primary text: Lee, Chapter 6**

Geodesics are the "straightest possible" curves — they parallel
transport their own tangent vectors. On a Riemannian manifold, they're
also locally shortest paths.

**Week 1-2: Geodesics**

- Definition: ∇_{γ'}γ' = 0 (autoparallel curves)
- Geodesic equation in coordinates: γ̈^i + Γ^i_{jk} γ̇^j γ̇^k = 0
- Existence and uniqueness: given p ∈ M and v ∈ T_pM, there's a unique
  geodesic through p with initial velocity v

**Week 3-4: The exponential map**

- exp_p: T_pM → M sends v to γ_v(1) where γ_v is the geodesic with
  γ_v(0) = p, γ_v'(0) = v
- Normal coordinates: coordinates where Christoffel symbols vanish at
  p
- Geodesic completeness and the Hopf-Rinow theorem

**Key insight**: The exponential map "wraps" the tangent space onto the
manifold via geodesics. In normal coordinates centered at p, geodesics
through p are straight lines (at least near p). This is the sense in
which geodesics are "straight."

**Computation**: Implement geodesic solver.

```python
import jax
import jax.numpy as jnp
import diffrax

def geodesic_ivp(metric_fn, x0, v0, t_span, dt=0.01):
    """
    Solve the geodesic initial value problem.

    Geodesic equation: ẍ^i + Γ^i_{jk} ẋ^j ẋ^k = 0

    Args:
        metric_fn: function x ↦ g_{ij}(x)
        x0: initial position, shape (n,)
        v0: initial velocity, shape (n,)
        t_span: (t_start, t_end)
        dt: time step

    Returns:
        ts: time points
        xs: positions along geodesic
        vs: velocities along geodesic
    """
    n = len(x0)

    def geodesic_ode(t, state, args):
        x, v = state[:n], state[n:]
        Gamma = christoffel_symbols(metric_fn, x)

        # Geodesic acceleration: a^i = -Γ^i_{jk} v^j v^k
        a = jnp.zeros(n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    a = a.at[i].add(-Gamma[i, j, k] * v[j] * v[k])

        return jnp.concatenate([v, a])

    t0, t1 = t_span
    ts = jnp.linspace(t0, t1, int((t1 - t0) / dt))

    term = diffrax.ODETerm(geodesic_ode)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=ts)

    state0 = jnp.concatenate([x0, v0])
    solution = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=dt, y0=state0, saveat=saveat
    )

    xs = solution.ys[:, :n]
    vs = solution.ys[:, n:]
    return solution.ts, xs, vs


# Geodesics in Poincaré half-plane are vertical lines and semicircles
x0 = jnp.array([0.0, 1.0])
v0 = jnp.array([1.0, 0.0])  # Initially horizontal → will curve into semicircle

ts, xs, vs = geodesic_ivp(poincare_half_plane_metric, x0, v0, (0, 2), dt=0.01)

print(f"Geodesic from {x0} with initial velocity {v0}")
print(f"Final position: ({xs[-1, 0]:.4f}, {xs[-1, 1]:.4f})")
print("(This traces part of a semicircle centered on the x-axis)")
```

### Month 13: Curvature

**Primary text: Lee, Chapters 7-8**

Curvature measures the failure of parallel transport to return vectors
to their starting position when transported around a loop.

**Week 1-2: The Riemann curvature tensor**

- Definition: R(X,Y)Z = ∇_X ∇_Y Z - ∇_Y ∇_X Z - ∇_{[X,Y]} Z
- In coordinates: R^i_{jkl} = ∂_k Γ^i_{jl} - ∂_l Γ^i_{jk} +
  Γ^i_{km}Γ^m_{jl} - Γ^i_{lm}Γ^m_{jk}
- Symmetries: R_{ijkl} = -R_{jikl} = -R_{ijlk} = R_{klij}, and Bianchi
  identity

**Week 3-4: Sectional, Ricci, and scalar curvature**

- Sectional curvature K(σ) for a 2-plane σ ⊂ T_pM
- Ricci curvature Ric(v,v) = trace of w ↦ R(w,v)v
- Scalar curvature S = trace of Ricci
- Constant curvature spaces: K ≡ const characterises space forms

**Key insight**: Sectional curvature K(σ) is the Gaussian curvature of
the 2-dimensional surface swept out by geodesics in directions
spanning σ. This connects Riemann curvature directly to your Phase 1
intuition.

**A note on component counting**: The Riemann tensor has 20
independent components in 4 dimensions (the dimension of spacetime),
which reduces to 10 for the Ricci tensor — exactly the number of
components in the symmetric metric tensor g_{μν}. This is why
Einstein's equations, which equate Ricci curvature to stress-energy,
form a determined system of 10 equations for the 10 metric components.
Not a coincidence.

**Connection to GR**: The Riemann tensor R^ρ_{σμν} in physics notation
is exactly the Riemann curvature tensor. The Ricci tensor R_{μν} =
R^ρ_{μρν} appears in Einstein's equations. Your old GR knowledge now
has a proper mathematical home.

**Transcribe**: Schuller Lectures 18-20 (Curvature in depth).

**Computation**: Implement Riemann tensor.

```python
import jax
import jax.numpy as jnp

def riemann_tensor(metric_fn, x):
    """
    Compute Riemann curvature tensor R^i_{jkl}.

    R^i_{jkl} = ∂_k Γ^i_{jl} - ∂_l Γ^i_{jk} + Γ^i_{km}Γ^m_{jl} - Γ^i_{lm}Γ^m_{jk}

    Returns:
        R: array of shape (n, n, n, n), where R[i, j, k, l] = R^i_{jkl}
    """
    n = len(x)

    Gamma = christoffel_symbols(metric_fn, x)

    # Partial derivatives of Christoffel symbols
    def Gamma_fn(y):
        return christoffel_symbols(metric_fn, y)

    dGamma = jax.jacfwd(Gamma_fn)(x)  # Shape: (n, n, n, n)
    # dGamma[i, j, k, l] = ∂Γ^i_{jk}/∂x^l

    R = jnp.zeros((n, n, n, n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    # R^i_{jkl} = ∂_k Γ^i_{jl} - ∂_l Γ^i_{jk} + Γ^i_{km}Γ^m_{jl} - Γ^i_{lm}Γ^m_{jk}
                    val = dGamma[i, j, l, k] - dGamma[i, j, k, l]
                    for m in range(n):
                        val += Gamma[i, k, m] * Gamma[m, j, l]
                        val -= Gamma[i, l, m] * Gamma[m, j, k]
                    R = R.at[i, j, k, l].set(val)

    return R


def scalar_curvature(metric_fn, x):
    """Scalar curvature: S = g^{jl} R^i_{jil}."""
    n = len(x)
    g = metric_fn(x)
    g_inv = jnp.linalg.inv(g)
    R = riemann_tensor(metric_fn, x)

    # Ricci: Ric_{jl} = R^i_{jil}
    Ric = jnp.zeros((n, n))
    for j in range(n):
        for l in range(n):
            Ric = Ric.at[j, l].set(sum(R[i, j, i, l] for i in range(n)))

    # Scalar: S = g^{jl} Ric_{jl}
    return jnp.einsum('jl,jl->', g_inv, Ric)


# Test: Poincaré half-plane has constant Gaussian curvature K = -1
# For 2D, scalar curvature S = 2K, so S = -2
point = jnp.array([1.0, 2.0])
S = scalar_curvature(poincare_half_plane_metric, point)
print(f"Scalar curvature of Poincaré half-plane: {S:.4f}")
print(f"Expected: -2.0 (since K = -1 and S = 2K in 2D)")
```

### Month 14: Global Riemannian geometry

**Primary text: Lee, Chapters 10-12 (selected sections)**

How does local curvature constrain global structure?

**Week 1-2: Jacobi fields**

- Jacobi fields measure geodesic deviation
- Equation: J'' + R(J, γ')γ' = 0 along geodesic γ
- Conjugate points: where Jacobi fields vanish

**Week 3-4: Global theorems**

- **Hopf-Rinow**: Geodesic completeness ⟺ metric completeness ⟺ closed
  bounded sets compact
- **Bonnet-Myers**: If Ric ≥ (n-1)κ > 0, then diam(M) ≤ π/√κ (positive
  curvature forces compactness)
- **Cartan-Hadamard**: If K ≤ 0 everywhere and M is complete simply
  connected, then exp_p: T_pM → M is a diffeomorphism (negative
  curvature forces "spread")

**Key insight**: Curvature bounds force global topology. Positive Ricci
curvature makes manifolds "close up" (bounded diameter). Non-positive
sectional curvature makes them "spread out" (exponential map is global
diffeomorphism). This is geometry constraining topology.

### Celebrating Phase 3

You now understand Riemannian geometry — the geometry of manifolds
with metrics. Connections, curvature, geodesics. This is the heart of
general relativity, of geometric analysis, of modern physics.

Pause. Look back at how far you've come from curves and surfaces.

### Spiral Return 3 (end of Month 14, ~1.5 weeks)

**Connect everything:**

- The Theorema Egregium says K depends only on the metric. Now proven:
  K is constructed from the Levi-Civita connection, which depends only
  on g.
- The geodesic equation from Do Carmo (for surfaces) is the same
  equation as Lee (for Riemannian manifolds) — different notations,
  same mathematics.
- Gauss-Bonnet is a 2D special case of the Chern-Gauss-Bonnet theorem.

**Preview symplectic geometry:**

The geodesic flow on T*M is Hamiltonian! Here's the setup:

- Phase space: T*M (cotangent bundle)
- Hamiltonian: H(q, p) = ½g^{ij}(q)p_i p_j (kinetic energy)
- Hamilton's equations recover the geodesic equation

This is not a coincidence — it's the bridge between Riemannian and
symplectic geometry.

**Write**: A 2500-word essay titled "Curvature: From Surfaces to
Spacetime." Trace the concept of curvature from Gaussian curvature of
surfaces through Riemann curvature of manifolds to Einstein's use in
general relativity. What remains constant? What generalises?

**Content retrospective**: How has your understanding of curvature
evolved? What was most difficult? What finally made it click?

**Phase 3 Capstone: The sphere with connection**

Now you have the tools:

- Compute Christoffel symbols for the round metric on S² in spherical
  coordinates
- Write the geodesic equation; verify great circles satisfy it
- Parallel transport a vector around a spherical triangle
- Compute the holonomy angle; verify it equals the enclosed area (the
  angular excess)
- This is Gauss-Bonnet in action: holonomy = ∫∫ K dA

*You see*: geodesics from connection, curvature from holonomy.

### Phase 3 Content Creation

**Primary deliverable: Interactive visualiser — "Parallel Transport
and Holonomy"**

*Time budget: 3-4 weeks*

Build an interactive tool that:
1. Displays a surface (sphere, torus, hyperbolic plane — user selects)
2. Lets user draw a closed loop on the surface
3. Animates parallel transport of a vector around the loop
4. Shows the holonomy angle (how much the vector rotates) and relates
   it to enclosed curvature via ∮ K dA ≈ rotation angle

This is a teaching tool you'll use repeatedly.

**Secondary deliverable: Blog post — "The Geometry Behind General
Relativity"**

*Time budget: 2 weeks*

For someone who learned GR from physics textbooks, explain:

- What Christoffel symbols really are: components of the Levi-Civita
  connection
- What the Riemann tensor really measures: parallel transport around
  infinitesimal loops
- Why torsion-freeness matters: we want geodesics to be autoparallel
  AND locally shortest
- Why Einstein's equations are geometric: curvature = mass-energy

**Tertiary deliverable: First Principles Club session — "Curvature:
The Shape of Space"**

*Time budget: 2 weeks preparation*

**Session structure (90 minutes):**

- 0:00-5:00 — Welcome, why curvature matters
- 5:00-20:00 — Gaussian curvature review: intrinsic vs extrinsic, the
  Theorema Egregium
- 20:00-40:00 — Connections and parallel transport: comparing vectors
  at different points
- 40:00-60:00 — Riemann curvature: parallel transport around loops
- 60:00-75:00 — Global implications: Bonnet-Myers, Cartan-Hadamard
- 75:00-85:00 — GR teaser: Einstein's equations
- 85:00-90:00 — Discussion, pointers to resources

### Phase 3 Mastery Checkpoints

Before proceeding, verify you can:

- [ ] Define Riemannian metric; write it in local coordinates
- [ ] State the two defining properties of the Levi-Civita connection
- [ ] Derive the formula for Christoffel symbols from metric
      compatibility and torsion-freeness
- [ ] Write the geodesic equation in coordinates; explain why it's
      second-order
- [ ] Define Riemann curvature tensor; give its geometric
      interpretation (one sentence)
- [ ] Relate sectional curvature to Gaussian curvature for surfaces
- [ ] State Hopf-Rinow; explain why completeness matters
- [ ] State Bonnet-Myers and Cartan-Hadamard; give the geometric
      intuition for each
- [ ] Translate between GR index notation (Γ^ρ_{μν}, R^ρ_{σμν}) and
      coordinate-free notation
- [ ] Your parallel transport visualiser works correctly

---

## Phase 4: Symplectic geometry and Hamiltonian mechanics

**Duration: 6 months (Months 15-20)**

*You've been climbing toward this view. Now you arrive.*

*This is your destination. Everything before was preparation.*

## The Lie Groups On-Ramp

### Prelude to Phase 4: Lie groups and Lie algebras

**Duration: 1.5-2 weeks**

Before entering symplectic geometry, you need basic fluency with Lie
groups. Momentum maps, gauge transformations, and the deepest parts of
mechanics all speak this language.

This is not a comprehensive treatment — that would be another
curriculum. This is the minimum you need: enough to recognise a Lie
group, find its Lie algebra, and understand how the adjoint and
coadjoint representations connect algebra to geometry.

**Week 1: Lie groups by example**


**What is a Lie group?**

A Lie group is a group that's also a smooth manifold, with smooth
group operations. The examples are more important than the definition:

- **S¹ = U(1)**: The circle, with group operation being addition of
  angles (or multiplication of unit complex numbers). This is the
  symmetry group of electromagnetism.

- **SO(3)**: Rotations of ℝ³. A 3-dimensional manifold (topologically
  ℝP³). The symmetry group of the sphere, and of any system with
  rotational symmetry.

- **SE(3)**: Rigid motions of ℝ³ (rotations + translations). A
  6-dimensional group. The symmetry group of a rigid body moving
  freely in space.

- **SU(2)**: 2×2 unitary matrices with determinant 1. Topologically
  S³. Double covers SO(3). The symmetry group of spin.

**The Lie algebra as tangent space at identity**

Every Lie group G has a Lie algebra 𝔤 = T_eG, the tangent space at the
identity element. This is a vector space with extra structure: the Lie
bracket [·, ·].

For matrix groups, the Lie algebra consists of matrices, and the
bracket is [A, B] = AB - BA.

- 𝔰𝔬(3) = {3×3 skew-symmetric matrices}. Dimension 3. Bracket is
  matrix commutator.
- 𝔰𝔢(3) = {4×4 matrices of the form [[ω, v], [0, 0]] with ω ∈ 𝔰𝔬(3), v
  ∈ ℝ³}. Dimension 6.
- 𝔰𝔲(2) = {2×2 skew-Hermitian traceless matrices}. Dimension 3.
  Isomorphic to 𝔰𝔬(3) as Lie algebras.

**The exponential map**

The exponential map exp: 𝔤 → G sends a Lie algebra element to a group
element. For matrix groups, this is the matrix exponential:

$$\exp(A) = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \cdots$$

Geometrically: exp(tξ) is the one-parameter subgroup generated by ξ ∈
𝔤. It's the flow of the left-invariant vector field determined by ξ.

For SO(3): exp maps skew-symmetric matrices to rotation matrices. The
axis of rotation is encoded in the matrix; the angle is the norm.

**Computation:**

```python
import jax.numpy as jnp
from jax.scipy.linalg import expm

def so3_basis():
    """Basis for so(3): skew-symmetric 3×3 matrices."""
    # L_x generates rotation around x-axis
    L_x = jnp.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    L_y = jnp.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
    L_z = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    return L_x, L_y, L_z

def hat(omega):
    """Map vector ω ∈ ℝ³ to skew-symmetric matrix ω̂ ∈ so(3)."""
    return jnp.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def exp_so3(omega):
    """Exponential map so(3) → SO(3)."""
    return expm(hat(omega))

# Rotation by angle θ around z-axis
theta = jnp.pi / 4
R = exp_so3(jnp.array([0, 0, theta]))
print(f"Rotation by π/4 around z-axis:\n{R}")

# Verify it's orthogonal with determinant 1
print(f"RᵀR = I: {jnp.allclose(R.T @ R, jnp.eye(3))}")
print(f"det(R) = {jnp.linalg.det(R):.4f}")
```

---

**Week 2: Adjoint and coadjoint representations**


**Why this matters for mechanics:**

When a Lie group G acts on a symplectic manifold, conserved quantities
live in 𝔤*, the dual of the Lie algebra. Angular momentum is an
element of 𝔰𝔬(3)*. The momentum map μ: M → 𝔤* packages all these
conserved quantities together.

To understand momentum maps, you need to know how G acts on its own
Lie algebra (adjoint) and on the dual (coadjoint).

**The adjoint representation**

G acts on 𝔤 by conjugation: for g ∈ G and ξ ∈ 𝔤,

$$\text{Ad}_g(\xi) = g \xi g^{-1}$$

For matrix groups, this is literal matrix conjugation. Geometrically:
Ad_g tells you how the Lie algebra "rotates" when you change your
reference frame by g.

For SO(3): Ad_R(ω̂) = R ω̂ R^T. This rotates the axis of the
infinitesimal rotation ω.

**The coadjoint representation**

G also acts on 𝔤*, the dual space:

$$\langle \text{Ad}^*_g(\mu), \xi \rangle = \langle \mu,
\text{Ad}_{g^{-1}}(\xi) \rangle$$

This is the action that matters for mechanics. Angular momentum lives
in 𝔰𝔬(3)* ≅ ℝ³, and the coadjoint action is how angular momentum
transforms under rotations.

**The punchline for momentum maps:**

When a Lie group G acts on a symplectic manifold (M, ω) preserving the
symplectic form, there's (often) a momentum map μ: M → 𝔤*. It
satisfies:

- μ is equivariant: μ(g · x) = Ad*_g(μ(x))
- For each ξ ∈ 𝔤, the function μ_ξ(x) = ⟨μ(x), ξ⟩ generates the
  infinitesimal action of ξ

This is abstract now. It will become concrete in Month 19 when you see
angular momentum as a momentum map for SO(3) acting on T*ℝ³.

**What you need to remember:**

- Lie group = smooth group. Examples: S¹, SO(3), SE(3), SU(2).
- Lie algebra 𝔤 = T_eG with bracket. For matrix groups: [A,B] = AB -
  BA.
- Exponential map: 𝔤 → G. For matrices: matrix exponential.
- Adjoint: G acts on 𝔤 by Ad_g(ξ) = gξg⁻¹.
- Coadjoint: G acts on 𝔤*. This is where momentum lives.
- Momentum map: μ: M → 𝔤*, equivariant, generates symmetry.

You don't need to master this now. You need to have seen it, so that
when momentum maps appear in Month 19, you have anchors.

**Read**: Marsden & Ratiu, *Introduction to Mechanics and Symmetry*,
Chapter 9 (Lie groups) — skim for intuition, don't get lost in proofs.

**Optional**: Stillwell, *Naive Lie Theory* — a gentler introduction
if you want more.

**Where Lie groups return**: This prelude gives you anchors. Lie
groups will reappear throughout:

- Phase 4, Months 19-20: Momentum maps package symmetry → conservation
- Phase 5: Gauge groups (U(1), SU(2), SU(3)) are Lie groups; gauge
  transformations are their action
- Phase 6: Equivariant neural networks are built on representations of
  Lie groups

By the end of the curriculum, you'll have used Lie groups in three
different ways: as symmetries generating conservation laws, as
structure groups of bundles, and as the algebraic foundation of
equivariant learning. The theory runs deeper — see "Branches not
taken" for where to continue.

### Month 15: Lagrangian mechanics and motivation

**Primary text: Arnold, *Mathematical Methods of Classical Mechanics*,
Chapters 1-6**

Before symplectic manifolds, you must understand the physics they
describe. Arnold's first six chapters provide the essential physical
grounding.

**Week 1-2: Newtonian and Lagrangian mechanics (Arnold Ch. 1-3)**

- Configuration space Q, generalised coordinates q^i
- The Lagrangian L(q, q̇, t) = T - V
- Euler-Lagrange equations: d/dt(∂L/∂q̇^i) - ∂L/∂q^i = 0
- The action principle: δ∫L dt = 0

**Week 3-4: The Legendre transform and Hamiltonian mechanics (Arnold

Ch. 4-6)**
- From Lagrangian to Hamiltonian: H(q, p) = p·q̇ - L(q, q̇)
- Canonical momentum: p_i = ∂L/∂q̇^i
- Hamilton's equations: q̇^i = ∂H/∂p_i, ṗ_i = -∂H/∂q^i
- Phase space T*Q as the natural arena

**Key insight**: The passage from Lagrangian to Hamiltonian mechanics is
a change of perspective: from velocities (q, q̇) ∈ TQ to momenta (q, p)
∈ T*Q. The Legendre transform encodes this. The cotangent bundle T*Q
is the natural home of Hamiltonian mechanics — and it carries a
canonical symplectic structure.

**A moment for reflection:**

Sit with the action principle. The universe appears to "know" the
entire path and choose the one that extremises the action. Particles
don't just respond to local forces — they follow paths that make a
global integral stationary. How? Why?

This question doesn't have a clean answer. It haunted Feynman, who
showed that in quantum mechanics, particles *do* explore all paths,
weighted by e^{iS/ℏ}. The classical path extremises action because
it's the path of stationary phase.

Write 500 words on what the action principle means to you. This isn't
a technical exercise — it's philosophical reflection.

**Common sticking point**: The Legendre transform can feel like
algebraic manipulation. Understand it geometrically: you're trading
the velocity variable for its "conjugate" momentum, which is dual to
velocity in a precise sense.

### Month 16: Symplectic linear algebra

**Primary text: Cannas da Silva, *Lectures on Symplectic Geometry*,
Part I** Available free at:
https://people.math.ethz.ch/~acannas/Papers/lsg.pdf

**Companion: Arnold, Chapter 7**

Before manifolds, understand the linear algebra.

**Week 1-2: Symplectic vector spaces**

- A symplectic form on a vector space V is a bilinear form ω: V × V →
  ℝ that is:
  - Skew-symmetric: ω(v, w) = -ω(w, v)
  - Non-degenerate: ω(v, w) = 0 for all w implies v = 0
- Standard example: V = ℝ²ⁿ with ω((q,p), (q',p')) = p'·q - p·q'
- Symplectic basis, Darboux coordinates

**Week 3-4: Symplectic linear maps and Lagrangian subspaces**

- The symplectic group Sp(2n, ℝ) = {A : Aᵀ J A = J} where J = [[0, I],
  [-I, 0]]
- Lagrangian subspaces: n-dimensional subspaces L ⊂ V with ω|_L = 0
- Symplectic geometry is 2n-dimensional geometry with "area
  preservation"

**Key insight**: A symplectic form pairs "position directions" with
"momentum directions." Non-degeneracy means every position has a
conjugate momentum. Skew-symmetry means ω(v, v) = 0 — no
"self-pairing." This is the linear algebra underlying Hamiltonian
mechanics.

**Definition (symplectic manifold)**: A symplectic manifold is a pair
(M, ω) where M is a smooth manifold and ω ∈ Ω²(M) is a 2-form that is:

- Closed: dω = 0
- Non-degenerate: ω_p: T_pM × T_pM → ℝ is non-degenerate at each p

The closedness condition dω = 0 is crucial — it ensures that the
symplectic structure is preserved under Hamiltonian flow. This is the
deep geometric reason behind conservation laws.

**Computation**: Implement symplectic linear algebra.

```python
import jax.numpy as jnp

def standard_symplectic_matrix(n):
    """
    Standard symplectic form on ℝ^{2n} as a matrix.

    J = [[0, I], [-I, 0]]

    so ω(v, w) = vᵀ J w

    Convention: coordinates ordered (q_1, ..., q_n, p_1, ..., p_n)
    """
    O = jnp.zeros((n, n))
    I = jnp.eye(n)
    return jnp.block([[O, I], [-I, O]])


def is_symplectic_matrix(A, J=None):
    """Check if A ∈ Sp(2n, ℝ), i.e., AᵀJA = J."""
    n = A.shape[0] // 2
    if J is None:
        J = standard_symplectic_matrix(n)
    return jnp.allclose(A.T @ J @ A, J)


def is_lagrangian_subspace(basis_vectors, J):
    """
    Check if the subspace spanned by basis_vectors is Lagrangian.

    Lagrangian means: dim = n and ω(v, w) = 0 for all v, w in subspace.
    """
    k = basis_vectors.shape[1]
    n = J.shape[0] // 2

    if k != n:
        return False

    # Check ω(v_i, v_j) = 0 for all basis vectors
    restriction = basis_vectors.T @ J @ basis_vectors
    return jnp.allclose(restriction, 0)


# Example: ℝ⁴ with standard symplectic structure
J4 = standard_symplectic_matrix(2)
print("Standard symplectic matrix J on ℝ⁴:")
print(J4)

# The position subspace span{∂/∂q¹, ∂/∂q²} is Lagrangian
position_subspace = jnp.array([
    [1, 0],   # q¹ direction
    [0, 1],   # q² direction
    [0, 0],   # p₁ component
    [0, 0]    # p₂ component
])
print(f"\nPosition subspace is Lagrangian: {is_lagrangian_subspace(position_subspace, J4)}")
```

### Month 17: Symplectic manifolds and Darboux

**Primary text: Cannas da Silva, Parts II-III**

**Week 1-2: Symplectic manifolds**

- Definition: (M, ω) with ω closed and non-degenerate
- Primary example: T*Q with canonical symplectic form ω = -dθ where θ
  = p_i dq^i
- Other examples: Kähler manifolds, coadjoint orbits

**Week 3-4: Darboux's theorem**

- **Statement**: Around any point of a symplectic manifold, there
  exist local coordinates (q¹,...,qⁿ,p₁,...,pₙ) such that ω = Σ dq^i ∧
  dp_i
- **Consequence**: All symplectic manifolds of the same dimension are
  locally identical
- **Contrast with Riemannian**: Riemannian manifolds have local
  invariants (curvature). Symplectic manifolds have none.
- Moser's trick (proof technique)

**Key insight**: Darboux's theorem is remarkable — it says symplectic
geometry has no local structure. Every 2n-dimensional symplectic
manifold looks locally like T*ℝⁿ. All the interesting structure is
global.

**Visual anchor**: In the (q, p) phase plane, draw a blob of initial
conditions — say, a small disk. Now evolve every point in the blob
under Hamiltonian flow for some time t. The blob deforms: it might
stretch, rotate, or develop tendrils. But its area remains exactly
constant. This is Liouville's theorem, and it's a consequence of the
symplectic structure. For the harmonic oscillator, the blob rotates
rigidly. For more complex systems, it stretches dramatically while
preserving area — this is the visual origin of chaos. Sketch phase
portraits for H = ½(p² + q²) and H = ½p² - ¼q⁴ to see the contrast.

**Exercises**: Cannas da Silva exercises for Parts II-III.

### Month 18: Hamiltonian mechanics as geometry

**Primary text: Cannas da Silva, Part IV** **Companion: Arnold,
Chapters 8-9**

Hamilton's equations are not physics imposed on geometry — they *are*
geometry.

**Week 1-2: Hamiltonian vector fields**

- Given H: M → ℝ, define X_H by: ω(X_H, ·) = dH
- In coordinates: X_H = (∂H/∂p_i)∂/∂q^i - (∂H/∂q^i)∂/∂p_i
- Hamilton's equations are: γ̇ = X_H(γ)
- Energy conservation: H is constant along integral curves of X_H

**Week 3-4: Poisson brackets**

- Definition: {f, g} = ω(X_f, X_g) = X_f(g) = -X_g(f)
- In coordinates: {f, g} = Σ(∂f/∂q^i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q^i)
- Jacobi identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
- Poisson bracket as Lie bracket: X_{f,g} = [X_f, X_g]

**Key insight**: The symplectic form ω is a machine that converts
differentials (1-forms like dH) into vector fields (like X_H).
Hamilton's equations simply say: "Flow in the direction that ω pairs
with dH." The Poisson bracket makes C^∞(M) into a Lie algebra —
functions can "act" on each other.

**Computation**: Implement Hamiltonian mechanics.

```python
import jax
import jax.numpy as jnp
import diffrax

def hamiltonian_vector_field(H, n):
    """
    Given Hamiltonian H: ℝ^{2n} → ℝ, return the Hamiltonian vector field X_H.

    In canonical coordinates (q, p), X_H = (∂H/∂p, -∂H/∂q).
    """
    def X_H(z):
        dH = jax.grad(H)(z)
        dH_dq = dH[:n]
        dH_dp = dH[n:]
        return jnp.concatenate([dH_dp, -dH_dq])  # (q̇, ṗ) = (∂H/∂p, -∂H/∂q)
    return X_H


def poisson_bracket(f, g, n):
    """
    Compute {f, g} = Σ(∂f/∂q^i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q^i).

    Returns a function z ↦ {f, g}(z).
    """
    def bracket(z):
        df = jax.grad(f)(z)
        dg = jax.grad(g)(z)

        df_dq, df_dp = df[:n], df[n:]
        dg_dq, dg_dp = dg[:n], dg[n:]

        return jnp.dot(df_dq, dg_dp) - jnp.dot(df_dp, dg_dq)
    return bracket


def hamiltonian_flow(H, z0, t_span, n, dt=0.01):
    """Compute the Hamiltonian flow φ_t^H(z0)."""
    X_H = hamiltonian_vector_field(H, n)

    def dynamics(t, z, args):
        return X_H(z)

    t0, t1 = t_span
    ts = jnp.linspace(t0, t1, int((t1 - t0) / dt))

    term = diffrax.ODETerm(dynamics)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=ts)

    solution = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=dt, y0=z0, saveat=saveat
    )
    return solution.ts, solution.ys


# Example: Simple harmonic oscillator H = (p² + q²)/2
def H_sho(z):
    q, p = z[0], z[1]
    return 0.5 * (p**2 + q**2)

n = 1  # 1 degree of freedom
z0 = jnp.array([1.0, 0.0])  # Start at (q, p) = (1, 0)

ts, zs = hamiltonian_flow(H_sho, z0, (0, 10), n)

print("Simple harmonic oscillator:")
print(f"Initial: q={z0[0]:.4f}, p={z0[1]:.4f}, H={H_sho(z0):.4f}")
print(f"Final:   q={zs[-1,0]:.4f}, p={zs[-1,1]:.4f}, H={H_sho(zs[-1]):.4f}")
print(f"Energy conserved: {jnp.abs(H_sho(zs[-1]) - H_sho(z0)) < 0.01}")

# Verify canonical Poisson brackets
# Note: poisson_bracket returns a function, which we evaluate at z_test
q_func = lambda z: z[0]
p_func = lambda z: z[1]

z_test = jnp.array([2.0, 3.0])
print(f"\nCanonical Poisson brackets at z = {z_test}:")
print(f"{{q, p}} = {poisson_bracket(q_func, p_func, n)(z_test):.4f} (expected: 1)")
print(f"{{q, q}} = {poisson_bracket(q_func, q_func, n)(z_test):.4f} (expected: 0)")
print(f"{{p, p}} = {poisson_bracket(p_func, p_func, n)(z_test):.4f} (expected: 0)")
```

### Month 19: Symmetry and momentum maps

**Primary text: Cannas da Silva, Parts V-VI**
**Essential companion: Marsden & Ratiu, *Introduction to Mechanics and Symmetry*, Chapters 1-4**

Noether's theorem geometrised: continuous symmetries correspond to
conserved quantities.

**Week 1-2: Group actions and symmetry**

- Lie group G acting on symplectic manifold (M, ω)
- Symplectic action: g*ω = ω for all g ∈ G
- Infinitesimal action: Lie algebra 𝔤 → vector fields on M

**Week 3-4: The momentum map**

- Definition: μ: M → 𝔤* is a momentum map if:
  - For each ξ ∈ 𝔤, the function μ_ξ = ⟨μ, ξ⟩ satisfies X_{μ_ξ} = ξ_M
  - μ is equivariant: μ(g·x) = Ad*_g(μ(x))
- Examples: linear momentum (translation), angular momentum (rotation)
- **Noether's theorem**: If H is G-invariant, then μ is constant along
  Hamiltonian flow

**Key insight**: The momentum map packages all conserved quantities
arising from symmetry into a single geometric object. Conservation of
μ along trajectories is Noether's theorem in its natural habitat.

Noether's theorem is often stated as "symmetry implies conservation."
But the deeper statement is that symmetry and conservation are *the
same thing* viewed from different angles. A symmetry *is* a conserved
quantity; a conserved quantity *is* a symmetry. This is not metaphor —
it is mathematical identity via the momentum map.

**Common sticking point**: The definition of momentum map involves
several pieces (the dual 𝔤*, equivariance, the generating property).
Work through the examples carefully: see how linear momentum, angular
momentum, and the moment of inertia tensor all fit this framework.

### Month 20: Symplectic integrators

**Primary text: Hairer, Lubich & Wanner, *Geometric Numerical
Integration*, Chapters 1-6**

Standard ODE solvers don't preserve symplectic structure. Over long
times, they introduce spurious drift. Symplectic integrators preserve
the geometry.

**Week 1-2: Why symplectic integration?**

- Standard integrators (Runge-Kutta, Adams) don't preserve ω
- Long-time energy drift: non-symplectic methods show secular growth
  in energy error
- Symplectic integrators: Aᵀ J A = J for the discrete flow A

**Week 3-4: Symplectic methods**


For separable Hamiltonians H(q, p) = T(p) + V(q):

**Symplectic Euler**:

- p_{n+1} = p_n - Δt · ∇V(q_n)
- q_{n+1} = q_n + Δt · ∇T(p_{n+1})

**Leapfrog (Störmer-Verlet)**:

- p_{n+1/2} = p_n - (Δt/2) · ∇V(q_n)
- q_{n+1} = q_n + Δt · ∇T(p_{n+1/2})
- p_{n+1} = p_{n+1/2} - (Δt/2) · ∇V(q_{n+1})

**Note on separability**: The methods above assume H = T(p) + V(q).
For non-separable Hamiltonians, you need implicit methods like
implicit midpoint. These require solving nonlinear equations at each
step.

**Key insight**: Symplectic integrators don't conserve H exactly. But
they conserve a "shadow Hamiltonian" H̃ = H + O(Δt²) *exactly*. This is
backward error analysis — the integrator solves a slightly different
but nearby Hamiltonian system exactly. That's why energy errors don't
drift.

**Computation**: Implement symplectic integrators and compare.

```python
import jax
import jax.numpy as jnp

def symplectic_euler(grad_V, grad_T, q0, p0, dt, num_steps):
    """
    Symplectic Euler for separable H = T(p) + V(q).
    """
    q, p = q0, p0
    qs, ps = [q], [p]

    for _ in range(num_steps):
        p = p - dt * grad_V(q)    # Kick
        q = q + dt * grad_T(p)    # Drift
        qs.append(q)
        ps.append(p)

    return jnp.stack(qs), jnp.stack(ps)


def leapfrog(grad_V, grad_T, q0, p0, dt, num_steps):
    """
    Leapfrog (Störmer-Verlet) — 2nd order symplectic.
    """
    q, p = q0, p0
    qs, ps = [q], [p]

    for _ in range(num_steps):
        p_half = p - 0.5 * dt * grad_V(q)
        q = q + dt * grad_T(p_half)
        p = p_half - 0.5 * dt * grad_V(q)
        qs.append(q)
        ps.append(p)

    return jnp.stack(qs), jnp.stack(ps)


def rk4_step(f, y, dt):
    """Single step of 4th-order Runge-Kutta (NOT symplectic)."""
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def rk4(H, z0, dt, num_steps, n):
    """RK4 for Hamiltonian system (NOT symplectic)."""
    X_H = hamiltonian_vector_field(H, n)

    z = z0
    zs = [z]

    for _ in range(num_steps):
        z = rk4_step(X_H, z, dt)
        zs.append(z)

    return jnp.stack(zs)


# Compare on Kepler problem: H = |p|²/2 - 1/|q|
def kepler_H(z):
    q, p = z[:2], z[2:]
    return 0.5 * jnp.sum(p**2) - 1.0 / jnp.linalg.norm(q)

def kepler_grad_V(q):
    r = jnp.linalg.norm(q)
    return -q / r**3

def kepler_grad_T(p):
    return p

# Initial conditions: circular orbit
q0 = jnp.array([1.0, 0.0])
p0 = jnp.array([0.0, 1.0])
z0 = jnp.concatenate([q0, p0])

# Integrate for many orbits
dt = 0.01
num_steps = 10000  # About 16 orbits

# Symplectic (leapfrog)
qs_symp, ps_symp = leapfrog(kepler_grad_V, kepler_grad_T, q0, p0, dt, num_steps)
zs_symp = jnp.concatenate([qs_symp, ps_symp], axis=1)

# Non-symplectic (RK4)
zs_rk4 = rk4(kepler_H, z0, dt, num_steps, n=2)

# Energy error over time
H0 = kepler_H(z0)
energy_error_symp = jnp.array([jnp.abs(kepler_H(z) - H0) for z in zs_symp])
energy_error_rk4 = jnp.array([jnp.abs(kepler_H(z) - H0) for z in zs_rk4])

print("Long-time energy error (Kepler problem, 10000 steps):")
print(f"  Leapfrog (symplectic): max error = {jnp.max(energy_error_symp):.2e}")
print(f"  RK4 (non-symplectic):  max error = {jnp.max(energy_error_rk4):.2e}")
print(f"\nRK4 is higher-order but shows secular drift.")
print("Leapfrog errors oscillate but don't grow systematically.")
```

### Celebrating Phase 4

This was your destination. You now understand that classical mechanics
is not "described by" symplectic geometry — it *is* symplectic
geometry. Phase space is a symplectic manifold. Hamilton's equations
are the flow of a symplectic vector field. Conservation laws arise
from symmetries via momentum maps.

You've climbed a long way. Take a few days to rest and reflect.

### Spiral Return 4 (end of Month 20, ~2 weeks)

**The grand unification:**

1. **Geodesic flow is Hamiltonian**: On a Riemannian manifold (M, g), the geodesic flow on T*M is generated by the Hamiltonian H(q, p) = ½g^{ij}(q)p_ip_j.

2. **Arnold's insight**: Many classical systems are geodesic flows in disguise:
   - Free rigid body rotation = geodesic flow on SO(3) with left-invariant metric
   - Ideal fluid motion = geodesic flow on the group of volume-preserving diffeomorphisms

3. **The two geometries are linked**: Riemannian geometry lives on configuration space Q. Symplectic geometry lives on phase space T*Q. The Legendre transform connects them.

**Concrete exercise**: Take the free particle on a Riemannian manifold. Write the Lagrangian L = ½g_{ij}q̇^iq̇^j. Compute the Legendre transform to get H. Write Hamilton's equations. Verify they're equivalent to the geodesic equation.

**Write**: A substantial essay (4000+ words) titled "The Unity of Geometry and Mechanics." This should be of publishable quality — suitable for a blog with high standards or a First Principles Club position paper. Argue that classical mechanics is not merely described by symplectic geometry; it *is* symplectic geometry.

**Content retrospective**: What was most surprising in this phase? What connections did you not expect? How has your view of physics changed?

**Phase 4 Capstone: The sphere as phase space**

The deepest view so far:

- The cotangent bundle T*S² is 4-dimensional and symplectic
- The Hamiltonian H(q,p) = ½g^{ij}p_i p_j generates geodesic flow
- Verify Hamilton's equations reproduce the geodesic equation
- The sphere has SO(3) symmetry (rotations). Find the momentum map μ:
  T*S² → so(3)*
- This is angular momentum. Verify it's conserved along geodesics
- Noether: rotation symmetry → angular momentum conservation

*You see*: geodesics as Hamiltonian flow, symmetry as conservation.

### Phase 4 Content Creation

**Primary deliverable: Talk — "Mechanics IS Geometry" (45 minutes)**

*Time budget: 4 weeks*

A signature talk for conferences, meetups, or First Principles Club. Structure:

- 0:00-5:00 — Hook: Why do planets move in ellipses? (Not because of inverse-square — because of symmetry)
- 5:00-12:00 — From Newton to Lagrange to Hamilton: the evolution of mechanics
- 12:00-22:00 — Phase space as symplectic manifold: the geometry behind the equations
- 22:00-30:00 — Hamilton's equations as symplectic geometry: X_H from ω and dH
- 30:00-38:00 — Noether's theorem: symmetry → conservation via momentum maps
- 38:00-42:00 — Why this matters for computation: symplectic integrators
- 42:00-45:00 — Conclusion: mechanics is not physics + geometry, it's just geometry

Polish until you could deliver it at a conference.

**Secondary deliverable: Video series — "Building Symplectic Integrators" (4 videos, 15-20 min each)**

*Time budget: 4-5 weeks*

1. "Why Standard Integrators Fail" — demonstrate energy drift with RK4
2. "Symplectic Euler and Leapfrog" — derivation, implementation, comparison
3. "Higher-Order Methods and Composition" — Yoshida's 4th-order method
4. "Applications: Planets, Molecules, and Beyond" — real-world examples

**Tertiary deliverable: First Principles Club workshop — "Hands-on Hamiltonian Mechanics" (2 hours)**

*Time budget: 2 weeks*

Participants bring laptops:

- 0:00-20:00 — Brief theory: Hamiltonian, vector field, Poisson brackets
- 20:00-50:00 — Implement Hamilton's equations for simple systems
- 50:00-80:00 — Implement symplectic Euler and leapfrog
- 80:00-110:00 — Compare: symplectic vs RK4 on Kepler problem
- 110:00-120:00 — Discussion: when to use symplectic methods

### Phase 4 Mastery Checkpoints

Before proceeding, verify you can:

- [ ] State the Euler-Lagrange equations; derive them from the action principle
- [ ] Perform the Legendre transform from L(q, q̇) to H(q, p)
- [ ] Define symplectic manifold; state both conditions (closed, non-degenerate)
- [ ] State Darboux's theorem precisely; explain why it's remarkable
- [ ] Given H, derive X_H from ω(X_H, ·) = dH
- [ ] Write Hamilton's equations in Darboux coordinates
- [ ] Define Poisson bracket; verify Jacobi identity for a specific example
- [ ] Define momentum map; give examples (linear and angular momentum)
- [ ] State Noether's theorem in symplectic language
- [ ] Implement symplectic Euler, leapfrog, and explain why they're structure-preserving
- [ ] Demonstrate the long-time advantage of symplectic methods
- [ ] Explain how geodesic flow is Hamiltonian

---

## Phase 5: Fiber bundles and connections

**Duration: 4 months (Months 21-24)**

*Goal: Master the structural framework underlying modern geometry and gauge theory.*

### Primary text for this phase

**John Baez & Javier Muniain, *Gauge Fields, Knots and Gravity***
(World Scientific, 1994)

This book was written for exactly your profile: physicists who want
mathematical precision, mathematicians who want physical intuition,
and anyone who wants to compute. It builds bundles and connections
from the ground up, with Maxwell and Yang-Mills as the guiding
examples. The exposition is clear, the examples are concrete, and the
perspective is modern.

Use **Husemoller, *Fibre Bundles*** as a reference for formal
definitions and proofs when you need more rigor. But learn from Baez &
Muniain.

### Month 21: Vector bundles and the need for connections

**Primary text: Baez & Muniain, Part I (Electromagnetism), Chapters 1-3**
**Companion: Schuller Lectures 21-22**

Begin with what you know: electromagnetism. See how the familiar
Maxwell equations point toward bundle structure.

**Week 1-2: Review and reframe**

- Maxwell's equations in differential form language
- The vector potential A and gauge transformations A → A + dλ
- Why gauge freedom suggests bundle structure

**Week 3-4: Vector bundles**

- Vector bundle: (E, π, M) where π: E → M with vector space fibers
- Sections: smooth maps s: M → E with π ∘ s = id_M
- The tangent and cotangent bundles as examples
- Transition functions and how bundles can be "twisted"

**Key insight**: The gauge freedom in electromagnetism isn't a bug —
it's a feature. It tells us the electromagnetic potential isn't a
function but a section of a bundle. Different gauge choices are
different local trivialisations of the same geometric object.

**Visual anchor**: Picture the base manifold M as a circle lying flat.
Above each point p of the circle, imagine a vertical line (a
1-dimensional vector space). The total space E is the union of all
these lines — a surface sitting above the circle. If you can "comb"
all the lines consistently, E looks like a cylinder (the trivial
bundle). But you might not be able to: if the fibers "twist" as you go
around, E becomes a Möbius strip. Same base, same fibers, different
global structure. The tangent bundle TS² is non-trivial for a deeper
reason: you can't comb a hairy ball flat (every vector field on S² has
a zero). Baez & Muniain, Chapter 1 has excellent illustrations. Sketch
both the cylinder and the Möbius strip as bundles over S¹.

**Reference**: Husemoller, Chapters 1-3 for formal definitions.

**Transcribe**: Schuller Lectures 21-22 (Fiber bundles).

### Month 22: Principal bundles and connections

**Primary text: Baez & Muniain, Part I (Electromagnetism), Chapters 4-5**
**Companion: Schuller Lectures 23-25**

**Week 1-2: Principal bundles**

- Principal G-bundle: P → M with free, transitive G-action on fibers
- The frame bundle F(M) as the prototype
- Associated bundles: how vector bundles arise from principal bundles
- Gauge transformations as bundle automorphisms

**Week 3-4: Connections**

- The problem: how to compare vectors (or group elements) at different
  points
- Connections as "horizontal subspaces" — a rule for lifting paths
- Connection 1-forms
- Parallel transport in bundles

**Key insight**: A connection tells you how to move through the total
space "horizontally" — staying parallel to the base. The
electromagnetic potential A is a connection on a U(1) bundle. The
Christoffel symbols from Riemannian geometry are a connection on the
frame bundle. Same concept, different bundles.

**Visual anchor**: Picture a fiber bundle as a surface E hovering
above a curve M. A path γ in the base wants to "lift" to a path in E,
but there are infinitely many lifts — you could wander up and down
within each fiber. A connection picks out a unique "horizontal" lift:
at each point of E, it defines which directions are "horizontal"
(parallel to the base) versus "vertical" (along the fiber). The
horizontal lift of γ is the unique path in E that projects to γ and
always moves horizontally. For the frame bundle, horizontal lift is
parallel transport: you're carrying a frame along γ without rotating
it relative to the connection. Schuller's Lecture 24 visualises this
beautifully; sketch a path on the base of a cylinder and its
horizontal lift on the surface.

**Reference**: Husemoller, Chapters 4-5 for formal treatment.

**Transcribe**: Schuller Lectures 23-25.

### Month 23: Curvature and gauge theory

**Primary text: Baez & Muniain, Part I (Electromagnetism), Chapters
6-7 and Part III (Gravity), selected**
**Companion: Frankel, Chapters 15-17**

**Week 1-2: Curvature of connections**

- Curvature as the failure of parallel transport to close around loops
- Curvature 2-form: Ω = dω + ½[ω, ω]
- Flat connections: Ω = 0
- Holonomy: the group element acquired by transport around a loop

**Week 3-4: Electromagnetism as geometry**

- The electromagnetic field F = dA is curvature of a U(1) connection
- Maxwell's equations: dF = 0 (Bianchi identity), d*F = J (field
  equation)
- Gauge transformations as changes of local trivialisation
- The Aharonov-Bohm effect: holonomy has physical consequences

**Key insight**: Maxwell's equations are not physics imposed on geometry
— they *are* geometry. dF = 0 is the Bianchi identity, true for any
curvature. d*F = J is the dynamical equation. Electromagnetism is the
simplest gauge theory.

**Visual anchor**: On a sphere, parallel transport a tangent vector
around a closed triangle (three geodesic segments). When you return to
the starting point, the vector has rotated — even though you never
"twisted" it locally. This rotation angle is the holonomy, and it
equals the integral of curvature over the enclosed region. For a
sphere with K = 1, a triangle enclosing area A gives holonomy angle A
(in radians). This is the geometric phase: global information (you
went around a loop) manifests as local change (the vector rotated). In
electromagnetism, the Aharonov-Bohm effect is exactly this — an
electron's phase shifts when its path encloses magnetic flux, even if
it never touches the field. Your parallel transport visualiser from
Phase 3 already shows this; revisit it now with bundle language.

**Reference**: Husemoller, Chapter 9 for connections in full
generality.

### Month 24: Non-abelian gauge theory and synthesis

**Primary text: Baez & Muniain, Part II (Gauge Fields)**
**Companion: Frankel, Chapters 17-18**

**Week 1-2: Yang-Mills theory**

- Non-abelian gauge groups: SU(2), SU(3)
- The Yang-Mills equations
- The Standard Model as a gauge theory (overview)

**Week 3-4: Characteristic classes (overview) and synthesis**

- Chern-Weil theory: building topological invariants from curvature
- Chern-Simons theory: the bridge to knot theory (optional preview)
- Chern classes, Pontryagin classes
- The Euler class and Gauss-Bonnet revisited
- How Riemannian geometry fits: the Levi-Civita connection on the
  frame bundle

**Key insight**: The Levi-Civita connection from Phase 3 is a connection
on the frame bundle F(M). Its curvature is the Riemann curvature
tensor. Riemannian geometry is gauge theory with structure group GL(n)
reduced to O(n). Everything connects.

**Note**: Characteristic classes deserve deeper study than this
overview provides. Flagged for post-curriculum exploration.

**Reference**: Husemoller, Chapters 6-8 for principal bundles, Chapter
9 for connections.

### Celebrating Phase 5

You now see the full structural picture: bundles, connections,
curvature. This is the language of modern geometry and theoretical
physics.

### Spiral Return 5 (end of Month 24, ~1 week)

**The full structural picture:**

- TM is an associated bundle of the frame bundle F(M)
- The Levi-Civita connection lives on F(M)
- Riemann curvature is its curvature
- Riemannian geometry is a special case of connection theory

Everything connects.

**Content retrospective**: How does seeing Riemannian geometry as
bundle theory change your understanding? What questions remain?

**Phase 5 Capstone: The sphere with structure group**

The full picture:

- The frame bundle F(S²) is a principal SO(2)-bundle over S²
- The Levi-Civita connection is a connection on F(S²)
- Its curvature 2-form encodes Gaussian curvature
- Parallel transport of frames around a loop: the holonomy is a
  rotation by the enclosed curvature
- See how Riemannian geometry is "SO(2) gauge theory" on the frame
  bundle

*You see*: geometry as gauge theory.

### Phase 5 Content Creation

**Primary deliverable: Blog series — "Fiber Bundles for Working Programmers" (5 posts)**

*Time budget: 5-6 weeks*

1. **"Why Neural Networks Need Fiber Bundles"** — Equivariance, symmetry, motivation
2. **"Vector Bundles: Spaces That Vary"** — From tangent spaces to general bundles
3. **"Principal Bundles: Symmetry Made Structural"** — Structure groups, gauge transformations
4. **"Connections: How to Transport Information"** — Parallel transport, covariant derivatives
5. **"Curvature: When Transport Fails to Close"** — Holonomy, gauge fields

**Secondary deliverable: First Principles Club session — "From Maxwell to Yang-Mills"**

*Time budget: 2 weeks*

**Session structure (90 minutes):**

- 0:00-5:00 — Welcome, what are gauge theories?
- 5:00-25:00 — Maxwell's equations in form language
- 25:00-45:00 — The gauge principle: U(1) bundles and connections
- 45:00-65:00 — Curvature as field strength: F = dA
- 65:00-80:00 — Non-abelian: SU(2), SU(3), the Standard Model
- 80:00-90:00 — Discussion

### Phase 5 Mastery Checkpoints

Before proceeding, verify you can:

- [ ] Define vector bundle; give three examples beyond TM
- [ ] Explain what a section of a bundle is
- [ ] Define principal G-bundle; explain the structure group's role
- [ ] Explain how TM arises as an associated bundle
- [ ] Define connection on a principal bundle
- [ ] Define curvature; relate to Riemann curvature for Levi-Civita
- [ ] Explain electromagnetism as U(1) gauge theory
- [ ] State Maxwell's equations in form language

---

### Prelude to Phase 6: From bundles to learning

**Duration: 1 week**

Before entering computational geometry and geometric ML, pause to see
why everything you've learned matters for modern machine learning.

**The core insight**: A neural network layer is a map between feature
spaces. If those feature spaces have symmetry — if rotating the input
should rotate the output in a predictable way — then the layer must
respect that structure. This is equivariance, and it's fiber bundle
thinking in disguise.

**The translation dictionary**:

| Differential geometry | Geometric deep learning |
|-----------------------|------------------------|
| Lie group G | Symmetry group (rotations, translations, permutations) |
| Representation of G on V | How G acts on feature vectors |
| Principal bundle P → M | Frame bundle over input space |
| Associated bundle P ×_G V | Feature bundle (features that transform under G) |
| Section of bundle | Feature map (assignment of features to points) |
| Connection | Way of comparing features at different points |
| Equivariant map | Layer that respects the symmetry |

The Lie groups from the Phase 4 prelude — SO(3), SE(3), SU(2) — are
exactly the symmetry groups that matter for physical systems. The
representation theory you glimpsed (adjoint, coadjoint) is what
determines how features can transform.

**Why this matters for Contravariant Systems**: Structure-preserving
computation requires knowing what structure to preserve. The bundle
language gives you a vocabulary for specifying symmetries, and
equivariant architectures are the computational implementation of that
specification.

**The discretization question**: Differential geometry lives on smooth
manifolds. Computation lives on finite meshes and discrete graphs.
Discrete exterior calculus bridges this gap — it's the theory of how
differential-geometric structures survive discretization. This is why
Month 29 exists.

You don't need to master this prelude. You need to see that Phase 6
isn't a departure from Phases 1-5 — it's their computational
fulfillment.

---

## Phase 6: Computational geometry and geometric ML

**Duration: 6 months (Months 25-30)**

*Goal: Connect everything to modern computation and machine learning.*

### Months 25-26: Geometric deep learning

**Primary text: Bronstein et al., *Geometric Deep Learning* (free at geometricdeeplearning.com)**

Neural networks encode geometric assumptions. Understanding these enables better design.

**Topics:**

- Symmetry, invariance, equivariance in ML
- Group theory for ML: representations, Schur's lemma
- Equivariant neural networks
- Graph neural networks as discrete geometry
- The geometric deep learning blueprint

**Computation**: Implement equivariant layers for SO(3).

### Months 27-28: Physics-informed architectures

**Topics:**

- Hamiltonian Neural Networks (Greydanus et al., 2019)
- Lagrangian Neural Networks (Cranmer et al., 2020)
- Neural ODEs with geometric structure
- Symplectic neural networks

**Computation**: Implement and train HNNs and LNNs. Compare with black-box neural ODEs.

### Month 29: Discrete exterior calculus

**Primary reading: Desbrun, Hirani, Leok, Marsden — "Discrete Exterior Calculus"**

DEC is differential geometry for meshes.

**Topics:**

- Simplicial complexes and chains
- Discrete forms as cochains
- Discrete exterior derivative
- Discrete Hodge star
- Applications: fluids, electromagnetism on meshes

### Month 30: Final project

**Time budget: 6-8 weeks**

Choose one substantial project:

**Option A: Geometric Mechanics Library**

Build a JAX library:

- Symplectic manifolds (cotangent bundles, coadjoint orbits)
- Hamiltonian systems with autodiff
- Momentum maps and reduction
- Symplectic integrators

*Deliverable*: Open-source library on PyPI.

**Option B: Structure-Preserving Neural Dynamics**

Research project:

- Novel architecture with geometric priors
- Comparison across physical systems
- Analysis of preserved structures

*Deliverable*: Technical report of workshop paper quality.

**Option C: Contravariant Systems Prototype**

Build toward executable science:

- Symbolic geometric structures
- Automatic equation derivation
- JAX integration
- Structure-preserving numerics

*Deliverable*: Working prototype with documentation.

**Phase 6 Capstone: The sphere in computation**

Bring it full circle:

- Implement geodesic regression on S²: given points, find the best-fit
  geodesic
- Use a Hamiltonian neural network to learn the geodesic flow
- Compare: does the learned system conserve angular momentum?
- Reflect: what does "structure-preserving learning" mean for this
  object you now know so well?

*You see*: geometry informing computation.

### Phase 6 Content Creation

**Primary deliverable: Project writeup (publishable quality)**

- Technical report (8-15 pages)
- Blog post for general audience
- Tutorial notebook

**Secondary deliverable: Tutorial — "Geometric Deep Learning from First Principles"**

Accessible introduction covering:

- Why geometry matters for ML
- Key architectures
- Hands-on implementations

---

## Branches not taken

Every path involves choices. Here are the roads not taken on this
journey — any of which you might explore later, from the high ground
you'll have reached.

**Algebraic topology**: Homotopy theory, singular homology, spectral
sequences. De Rham cohomology is the differential-geometric shadow of
richer structures. Reference: Hatcher, *Algebraic Topology*.

**Complex geometry**: Complex manifolds, Kähler metrics, Hodge theory.
Many symplectic manifolds are Kähler. Reference: Huybrechts, *Complex
Geometry*.

**Lie theory in depth**: The Lie groups on-ramp in Phase 4 gives the
minimum needed for momentum maps. But Lie theory is vast: the
classification of simple Lie algebras, representation theory, the
structure of compact groups, symmetric spaces. If you pursue geometric
deep learning seriously, you'll need representation theory (how groups
act on vector spaces). If you pursue geometric mechanics, you'll need
coadjoint orbits and the Kirillov correspondence. The standard
reference is Knapp, *Lie Groups Beyond an Introduction*. For
representation theory specifically: Fulton & Harris, *Representation
Theory: A First Course*.

**Characteristic classes**: The curriculum gives only an overview in
Month 24. This is a significant gap. Characteristic classes are
topological invariants built from curvature — they detect when bundles
are "twisted" in ways that can't be untwisted. The Euler class
generalizes Gauss-Bonnet; the Chern classes are essential for complex
geometry; the Pontryagin classes matter for real bundles and appear in
physics (anomalies, index theorems). Chern-Weil theory shows how to
compute these classes from connections — the same connections you
studied in Phase 5. If you pursue gauge theory, string theory, or
topological data analysis, you will need this material. Reference:
Milnor & Stasheff, *Characteristic Classes* — dense but definitive.
For a gentler path: Bott & Tu, *Differential Forms in Algebraic
Topology*, Part III.

**Index theory**: The Atiyah-Singer index theorem — analysis,
geometry, topology unified.

**Infinite-dimensional geometry**: Loop spaces, diffeomorphism groups,
field theory.

**Contact geometry**: Odd-dimensional partner of symplectic. Optics,
thermodynamics, control. Reference: Geiges, *An Introduction to
Contact Topology*.

**Geometric quantization**: From classical (symplectic) to quantum.
Where Poisson brackets become commutators.

These are not failures but choices. You can return to any of them.

---

## Timeline summary

| Phase | Duration | Months | Focus |
|-------|----------|--------|-------|
| 0 | 2-3 weeks | 0 | Calibration |
| 1 | 4 months | 1-4 | Curves and surfaces |
| 2 | 5 months | 5-9 | Smooth manifolds |
| 3 | 5 months | 10-14 | Riemannian geometry |
| 4 | 6 months | 15-20 | Symplectic geometry |
| 5 | 4 months | 21-24 | Fiber bundles |
| 6 | 6 months | 25-30 | Computational geometry |

**Total: 30 months of serious work.**

---

## Final words

This curriculum asks much: thirty months, six projects, content
created at every phase. But you're not just learning differential
geometry. You're building the foundation for Contravariant Systems,
for First Principles Club, for a body of work that unites mathematics,
physics, and computation.

This mathematics is beautiful. The structures are deep and surprising.
Understanding them is not a means to an end — it is an end in itself,
a source of meaning and wonder.

There will be difficult stretches. Plateaus where nothing seems to
click. Evenings when the notation swims. This is the nature of
learning something real. The structure of this curriculum — the
spirals, the projects, the explanations — is designed to carry you
through.

The goal is not to finish the reading list. The goal is to reach the
point where these ideas are *yours* — where you think in manifolds and
forms and connections as naturally as you now think in functions and
derivatives.

That's what it means to understand something cold.

The geometry awaits.
