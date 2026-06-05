---
title: Convex Conjugate (Legendre–Fenchel Transform)
type: concept
tags: [optimization, convex-optimization, duality]
difficulty: advanced
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Convex Conjugate (Legendre–Fenchel Transform)

> **One-liner:** Describe a bowl-shaped curve not by its points but by all the slanted lines that just touch it from below — then "slope" becomes the new variable.

## 🟢 Intuition (Dummy)
A convex curve (a bowl) can be drawn in two equivalent ways: by listing its points, or by
listing every straight line that grazes it from underneath without crossing — its
**tangent lines**. Each tangent has a slope and an intercept, and there's exactly one
tangent for each slope. So instead of asking "what's the height at this $x$?" you can ask
"for a line of this slope, how low does it touch?" Re-describing the curve in terms of
*slope* instead of *position* is the **Legendre–Fenchel transform** (a.k.a. **convex
conjugate**). It's the engine for turning one optimization problem into its dual.

## 🟡 Mechanics (Practitioner)
**Definition 7.4.** The convex conjugate of $f:\mathbb{R}^D\to\mathbb{R}$ is
$$ f^*(\mathbf{s}) = \sup_{\mathbf{x}\in\mathbb{R}^D}\big(\langle\mathbf{s},\mathbf{x}\rangle - f(\mathbf{x})\big), $$
a function of the new "slope" variable $\mathbf{s}$ (the tangent direction
$\mathbf{s}=\nabla_{\mathbf{x}}f(\mathbf{x})$). It needs $f$ neither convex nor
differentiable. Using the standard dot product, $\langle\mathbf{s},\mathbf{x}\rangle=\mathbf{s}^\top\mathbf{x}$.

**Geometric reading (1-D, $f(x)=x^2$):** a line $y=sx+c$ with fixed slope $s$ "just
touches" the curve at the smallest intercept $c=\inf_{x_0}(-sx_0+f(x_0))$; the conjugate
is defined as the negative of this. For a smooth convex $f$ there's no need for the
$\sup$ and a one-to-one $f\leftrightarrow f^*$ correspondence.

**Worked example (Ex. 7.7):** for $f(\mathbf{y})=\tfrac{\lambda}{2}\mathbf{y}^\top K^{-1}\mathbf{y}$
with $K\succ0$, set $\nabla_{\mathbf{y}}(\langle\mathbf{y},\boldsymbol\alpha\rangle-f(\mathbf{y}))=0$
to get $\mathbf{y}=\tfrac{1}{\lambda}K\boldsymbol\alpha$, giving
$$ f^*(\boldsymbol\alpha) = \tfrac{1}{2\lambda}\boldsymbol\alpha^\top K\boldsymbol\alpha. $$

**Conjugate of a sum (Ex. 7.8):** separable functions have separable conjugates,
$L(\mathbf{t})=\sum_i\ell_i(t_i)\;\Rightarrow\;L^*(\mathbf{z})=\sum_i\ell_i^*(z_i)$.

**ML usage:** for convex losses applied independently per example, the conjugate loss is a
clean route to a **dual problem** — used in SVMs and regularized risk minimization. It
gives a duality framework alternative to [[lagrange-multipliers|Lagrangian duality]].

## 🔴 Theory (Expert)
**Why it works.** A convex set is the intersection of its **supporting hyperplanes**; a
convex function's epigraph is a convex set, so the function is equivalently described by
its supporting hyperplanes — which are its tangents. The Legendre–Fenchel transform makes
"function of position" into "function of gradient." Names: classical **Legendre
transform** ([[adrien-marie-legendre]], on convex differentiable functions) generalized
to nonconvex/non-differentiable by **Fenchel** ([[werner-fenchel]]) via the supremum.

- **Involution on convex functions:** $f^{**}=f$ when $f$ is convex and closed —
  applying the transform twice returns the original. The slope of $f$ is $s$; the slope of
  $f^*$ is $x$ (a symmetric gradient relation).
- **Dual derivation (Ex. 7.9):** for convex $f,g$ and $A\mathbf{x}=\mathbf{y}$,
  $$ \min_{\mathbf{x}} f(A\mathbf{x})+g(\mathbf{x}) = \max_{\mathbf{u}} -f^*(\mathbf{u}) - g^*(-A^\top\mathbf{u}), $$
  derived by introducing a Lagrange multiplier $\mathbf{u}$, swapping min/max (valid by
  convexity), and recognizing two conjugates. (For general inner products $A^\top$ is
  replaced by the adjoint $A^*$.)
- For convex differentiable $f$ the $\sup$ is attained uniquely, linking conjugacy to the
  invertibility of the gradient map $\mathbf{x}\mapsto\nabla f(\mathbf{x})$.
- Physics connection: relates the Lagrangian and Hamiltonian in classical mechanics.

## Connections
- Builds on: [[convex-optimization]], [[gradient]], [[lagrange-multipliers]]
- Leads to: dual formulations of regularized ML losses, SVMs (Ch. 12)
- Used by: convex-loss dual problems, [[inner-product]]
- People: [[adrien-marie-legendre]], [[werner-fenchel]]

## Sources
- `raw/mml-book.pdf` §7.3.3 (Def. 7.4, Eqs. 7.53–7.68, Examples 7.7–7.9), printed pp. 242–246 (PDF pp. 247–251).
