---
title: Linear Programming
type: concept
tags: [optimization, convex-optimization, linear-programming]
difficulty: intermediate
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Linear Programming

> **One-liner:** Push a straight cost as low as possible while staying inside a region carved out by straight walls — the best answer always sits at a corner.

## 🟢 Intuition (Dummy)
You're mixing ingredients to make something as cheaply as possible, but each ingredient
has limits ("at most this much sugar," "at least that much flour"). Both the cost and
every limit are *straight-line* rules — nothing curvy. Those limits fence off an allowed
region (a polygon). Because cost increases steadily in one direction, the cheapest legal
point is always a corner of that fenced region. A **linear program** is exactly this:
straight objective, straight constraints, answer at a corner.

## 🟡 Mechanics (Practitioner)
The standard form (7.39):
$$ \min_{\mathbf{x}\in\mathbb{R}^d} \mathbf{c}^\top\mathbf{x}\quad\text{subject to}\quad A\mathbf{x}\le\mathbf{b}, $$
with $A\in\mathbb{R}^{m\times d}$, $\mathbf{b}\in\mathbb{R}^m$: $d$ variables, $m$ linear
constraints. The objective and all constraints are linear, so this is a
[[convex-optimization|convex]] program (linear contour lines, polyhedral feasible set).

**The dual via [[lagrange-multipliers|Lagrange multipliers]].** The Lagrangian is
$$ L(\mathbf{x},\boldsymbol\lambda)=\mathbf{c}^\top\mathbf{x}+\boldsymbol\lambda^\top(A\mathbf{x}-\mathbf{b})=(\mathbf{c}+A^\top\boldsymbol\lambda)^\top\mathbf{x}-\boldsymbol\lambda^\top\mathbf{b}. $$
Setting $\partial L/\partial\mathbf{x}=0$ gives $\mathbf{c}+A^\top\boldsymbol\lambda=0$,
so $\mathcal{D}(\boldsymbol\lambda)=-\boldsymbol\lambda^\top\mathbf{b}$. The **dual LP**:
$$ \max_{\boldsymbol\lambda\in\mathbb{R}^m} -\mathbf{b}^\top\boldsymbol\lambda\quad\text{subject to}\quad \mathbf{c}+A^\top\boldsymbol\lambda=0,\ \boldsymbol\lambda\ge0. $$
This is *also* a linear program, but with $m$ variables instead of $d$. **Solve whichever
is smaller** — primal if $d<m$, dual if $m<d$.

**Worked example (Ex. 7.5):** $\min_{\mathbf{x}\in\mathbb{R}^2} -\big[5,3\big]\mathbf{x}$
subject to five linear inequalities; the linear objective gives linear contours and the
optimum sits at a vertex of the feasible polygon (Fig. 7.9).

**ML usage:** "one of the most widely used approaches in industry" — resource allocation,
$\ell_1$-regularized formulations, and as a subroutine in larger ML pipelines.

## 🔴 Theory (Expert)
- **Strong duality** holds (the program is convex), so primal optimum $=$ dual optimum;
  the dual gives certificates of optimality and infeasibility.
- The feasible set $\{\mathbf{x}: A\mathbf{x}\le\mathbf{b}\}$ is a **polyhedron**; if the
  optimum is finite it is attained at a **vertex** (extreme point) — the basis of the
  simplex method. Interior-point methods solve LPs in polynomial time.
- LP is the linear special case of the general convex program (7.38); the quadratic
  generalization is [[quadratic-programming]].
- Symmetry of primal/dual variable counts ($d\leftrightarrow m$) is a recurring duality
  theme echoing [[lagrange-multipliers|Lagrangian duality]].

## Connections
- Builds on: [[lagrange-multipliers]], [[convex-optimization]], [[system-of-linear-equations]]
- Leads to: [[quadratic-programming]]
- Used by: industrial optimization, $\ell_1$ methods in ML
- People: [[joseph-louis-lagrange]]

## Sources
- `raw/mml-book.pdf` §7.3.1 (Eqs. 7.39–7.44, Example 7.5), printed pp. 239–240 (PDF pp. 244–245).
