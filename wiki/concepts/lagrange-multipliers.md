---
title: Lagrange Multipliers and Lagrangian Duality
type: concept
tags: [optimization, constrained-optimization, duality]
difficulty: advanced
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Lagrange Multipliers and Lagrangian Duality

> **One-liner:** When you must minimize something while obeying rules, attach a price tag to breaking each rule and fold those prices into the objective — then let the prices themselves become a new, easier problem to solve.

## 🟢 Intuition (Dummy)
Suppose you want to spend as little money as possible (minimize cost) but you're *not
allowed* to leave a certain neighborhood (a constraint). The clumsy approach is an
infinite fine: step outside and you owe $\infty$ — but that's impossible to optimize
smoothly. The clever approach: instead of an infinite fine, charge a *finite price* per
unit you stray over the line. That price is the **Lagrange multiplier**. Crank the price
up just enough and the cheapest plan naturally stays in-bounds. Even better, you can
flip the whole thing around: pick the prices to make the situation as expensive as
possible for the optimizer — a "dual" game that's often much easier to solve.

## 🟡 Mechanics (Practitioner)
The **constrained problem** (the *primal*):
$$ \min_{\mathbf{x}} f(\mathbf{x}) \quad\text{subject to}\quad g_i(\mathbf{x})\le 0,\ i=1,\dots,m. $$
Introduce **Lagrange multipliers** $\lambda_i\ge 0$, one per inequality, and form the
**Lagrangian**:
$$ L(\mathbf{x},\boldsymbol\lambda) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) = f(\mathbf{x}) + \boldsymbol\lambda^\top g(\mathbf{x}). $$
This replaces the brutal indicator penalty $\mathbf{1}(z)=\{0 \text{ if } z\le0,\ \infty \text{ else}\}$ with a linear one. The **dual function** minimizes out $\mathbf{x}$:
$$ \mathcal{D}(\boldsymbol\lambda) = \min_{\mathbf{x}\in\mathbb{R}^d} L(\mathbf{x},\boldsymbol\lambda), $$
and the **Lagrangian dual problem** maximizes it:
$$ \max_{\boldsymbol\lambda\in\mathbb{R}^m} \mathcal{D}(\boldsymbol\lambda)\quad\text{subject to}\quad \boldsymbol\lambda\ge 0. $$

**Recipe:** differentiate $L$ w.r.t. $\mathbf{x}$, set to zero, solve for $\mathbf{x}^*$,
substitute back to get $\mathcal{D}(\boldsymbol\lambda)$, then maximize over
$\boldsymbol\lambda\ge 0$.

**Equality constraints** $h_j(\mathbf{x})=0$: rewrite each as $h_j\le 0$ **and**
$-h_j\le 0$; the net effect is that the multiplier for an equality constraint is
**unconstrained** (any sign), while inequality multipliers stay $\ge 0$.

**ML usage:** the engine behind the **support vector machine** dual (Ch. 12), and behind
[[linear-programming]] and [[quadratic-programming]] duals. Solving the dual is often
cheaper and exposes structure (e.g. support vectors, kernels).

## 🔴 Theory (Expert)
**Why is the dual nice?** $L(\mathbf{x},\boldsymbol\lambda)$ is **affine in
$\boldsymbol\lambda$**, so $\mathcal{D}(\boldsymbol\lambda)=\min_{\mathbf{x}}L$ is a
pointwise minimum of affine functions, hence **concave** — *even if $f$ and $g_i$ are
nonconvex*. So the dual is always a concave maximization (efficiently solvable).

**Minimax inequality (7.23):** for any $\varphi(\mathbf{x},\mathbf{y})$,
$$ \max_{\mathbf{y}}\min_{\mathbf{x}} \varphi(\mathbf{x},\mathbf{y}) \;\le\; \min_{\mathbf{x}}\max_{\mathbf{y}} \varphi(\mathbf{x},\mathbf{y}). $$
Proof: $\min_x\varphi\le\max_y\varphi$ for all $x,y$; take $\max_y$ of the LHS and
$\min_x$ of the RHS preserving the inequality.

**Weak duality (7.27):** since $J(\mathbf{x})=\max_{\boldsymbol\lambda\ge0}L(\mathbf{x},\boldsymbol\lambda)$
recovers the indicator-penalized objective, the original problem is
$\min_{\mathbf{x}}\max_{\boldsymbol\lambda\ge0}L$. Applying the minimax inequality:
$$ \underbrace{\min_{\mathbf{x}}\max_{\boldsymbol\lambda\ge0}L}_{\text{primal}} \;\ge\; \underbrace{\max_{\boldsymbol\lambda\ge0}\min_{\mathbf{x}}L}_{\text{dual}}. $$
So the **dual value is always a lower bound on the primal value.** When $f,g_i$ are
[[convex-optimization|convex]], this becomes **strong duality** (equality). This is one
of two duality frameworks in the chapter; the other is Legendre–Fenchel
([[convex-conjugate]]).

## Connections
- Builds on: [[gradient]], [[affine-space]]
- Leads to: [[convex-optimization]], [[linear-programming]], [[quadratic-programming]], [[convex-conjugate]]
- Used by: support vector machines (Ch. 12), constrained ML objectives
- People: [[joseph-louis-lagrange]]

## Sources
- `raw/mml-book.pdf` §7.2 (Def. 7.1, Eqs. 7.16–7.28), printed pp. 233–236 (PDF pp. 238–241).
