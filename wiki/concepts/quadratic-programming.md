---
title: Quadratic Programming
type: concept
tags: [optimization, convex-optimization, quadratic-programming]
difficulty: intermediate
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Quadratic Programming

> **One-liner:** Minimize a bowl-shaped cost while staying inside straight walls — the same as a linear program, but now the cost itself curves like a bowl.

## 🟢 Intuition (Dummy)
A [[linear-programming|linear program]] has a flat, tilted cost; a **quadratic program**
has a cost shaped like a bowl. You still must stay inside a fenced region with straight
walls, but now the cheapest point might be *inside* the region (at the bottom of the
bowl) rather than always at a corner. Because the bowl has a single lowest point, the
problem stays well-behaved and has one best answer.

## 🟡 Mechanics (Practitioner)
The standard form (7.45):
$$ \min_{\mathbf{x}\in\mathbb{R}^d} \tfrac12\mathbf{x}^\top Q\mathbf{x} + \mathbf{c}^\top\mathbf{x}\quad\text{subject to}\quad A\mathbf{x}\le\mathbf{b}, $$
with $A\in\mathbb{R}^{m\times d}$, $\mathbf{b}\in\mathbb{R}^m$, $\mathbf{c}\in\mathbb{R}^d$,
and $Q\in\mathbb{R}^{d\times d}$ **symmetric positive definite**. Positive-definite $Q$
makes the objective [[convex-optimization|convex]] (elliptical contour lines), so the
problem is a convex program with $d$ variables and $m$ linear constraints.

**The dual via [[lagrange-multipliers|Lagrange multipliers]].** The Lagrangian is
$$ L(\mathbf{x},\boldsymbol\lambda)=\tfrac12\mathbf{x}^\top Q\mathbf{x} + (\mathbf{c}+A^\top\boldsymbol\lambda)^\top\mathbf{x} - \boldsymbol\lambda^\top\mathbf{b}. $$
Setting $\partial L/\partial\mathbf{x}=0$ gives $Q\mathbf{x}+(\mathbf{c}+A^\top\boldsymbol\lambda)=0$.
Because $Q$ is positive definite it is **invertible**, so
$$ \mathbf{x} = -Q^{-1}(\mathbf{c}+A^\top\boldsymbol\lambda). $$
Substituting back yields the **dual QP**:
$$ \max_{\boldsymbol\lambda\in\mathbb{R}^m} -\tfrac12(\mathbf{c}+A^\top\boldsymbol\lambda)^\top Q^{-1}(\mathbf{c}+A^\top\boldsymbol\lambda) - \boldsymbol\lambda^\top\mathbf{b}\quad\text{subject to}\quad \boldsymbol\lambda\ge0. $$

**Worked example (Ex. 7.6):** $Q=\big[\begin{smallmatrix}2&1\\1&4\end{smallmatrix}\big]$,
$\mathbf{c}=\big[5,3\big]^\top$, with box constraints $-1\le x_1,x_2\le 1$. Elliptical
contours; optimum at the star inside/on the feasible box (Fig. 7.4).

**ML usage:** the **support vector machine** (Ch. 12) is the canonical QP in ML — the
margin-maximization objective is quadratic and the dual QP exposes support vectors and
kernels. Also least-squares with linear constraints.

## 🔴 Theory (Expert)
- The closed-form unconstrained minimizer $\mathbf{x}=-Q^{-1}\mathbf{c}$ exists exactly
  because $Q\succ0$; constraints then truncate this to the feasible polyhedron.
- The dual objective $-\tfrac12(\cdot)^\top Q^{-1}(\cdot)-\boldsymbol\lambda^\top\mathbf{b}$
  is **concave** in $\boldsymbol\lambda$ (negative of a positive-definite quadratic form
  plus a linear term), so the dual is a concave maximization with simple sign constraints.
- **Strong duality** holds. QP generalizes [[linear-programming|LP]] (set $Q=0$, though
  then convexity is only semidefinite) and specializes general convex programming (7.38).
- If $Q$ is only positive *semi*definite the objective is still convex but the minimizer
  may be non-unique; the book assumes $Q\succ0$ for a clean closed form.

## Connections
- Builds on: [[lagrange-multipliers]], [[convex-optimization]], [[matrix]]
- Leads to: support vector machines (Ch. 12)
- Used by: SVMs, constrained least squares
- People: [[joseph-louis-lagrange]]

## Sources
- `raw/mml-book.pdf` §7.3.2 (Eqs. 7.45–7.52, Example 7.6), printed pp. 241–242 (PDF pp. 246–247).
