---
title: "MML Ch. 7 — Continuous Optimization"
type: source
tags: [optimization, gradient-descent, convex-optimization, machine-learning]
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch. 7 — Continuous Optimization

- **Raw file:** `raw/mml-book.pdf` §7, printed pp. 225–248 (PDF pp. 230–253)
- **Type:** book chapter
- **Ingested:** 2026-06-04

Training an ML model usually reduces to minimizing a differentiable **objective
function** $f:\mathbb{R}^D\to\mathbb{R}$ over its parameters. This chapter is the
numerical-optimization toolbox for "find a good set of parameters." It splits into
two branches (Fig. 7.1): **unconstrained** optimization (gradient descent and its
variants) and **constrained** optimization (Lagrange multipliers, duality), plus a
special well-behaved class — **convex** optimization — where local optima are global.

## Section coverage

| Section | Title | Wiki page |
|---|---|---|
| 7 intro | Stationary points, global vs. local minima, descent intuition | (covered across pages below) |
| 7.1 | Optimization Using Gradient Descent | [[gradient-descent]] |
| 7.1.1 | Step-size (learning rate), adaptive heuristics | [[gradient-descent]] |
| 7.1.2 | Gradient Descent With Momentum | [[gradient-descent]] |
| 7.1.3 | Stochastic Gradient Descent | [[stochastic-gradient-descent]] |
| 7.2 | Constrained Optimization and Lagrange Multipliers; Lagrangian, primal/dual, minimax inequality, weak duality | [[lagrange-multipliers]] |
| 7.3 | Convex Optimization; convex set, convex function, Jensen, strong duality | [[convex-optimization]] |
| 7.3.1 | Linear Programming | [[linear-programming]] |
| 7.3.2 | Quadratic Programming | [[quadratic-programming]] |
| 7.3.3 | Legendre–Fenchel Transform and Convex Conjugate | [[convex-conjugate]] |
| 7.4 | Further Reading (Newton/quasi-Newton, conjugate gradient, subgradient, mirror descent) | — |

## TL;DR (5 bullets)
- **Gradient descent** moves parameters opposite the gradient, $\mathbf{x}_{i+1}=\mathbf{x}_i-\gamma_i\big((\nabla f)(\mathbf{x}_i)\big)^\top$; the **step-size** $\gamma$ (learning rate) is the critical knob, and **momentum** and **SGD** are the two key practical refinements.
- **Stochastic gradient descent** replaces the full-data sum gradient with a cheap unbiased mini-batch estimate — the workhorse of large-scale ML.
- **Constrained** problems $\min_x f(x)$ s.t. $g_i(x)\le 0$ are handled with **Lagrange multipliers**: build the Lagrangian $L(\mathbf{x},\boldsymbol\lambda)=f(\mathbf{x})+\boldsymbol\lambda^\top g(\mathbf{x})$ and pass to the **dual** problem $\max_{\boldsymbol\lambda\ge 0} \mathcal{D}(\boldsymbol\lambda)$.
- **Weak duality** (dual $\le$ primal) always holds via the minimax inequality; for **convex** problems we get **strong duality** (dual $=$ primal) and every local minimum is global.
- Two canonical convex programs — **linear programs** ($\min c^\top x$ s.t. $Ax\le b$) and **quadratic programs** ($\min \tfrac12 x^\top Q x + c^\top x$ s.t. $Ax\le b$) — and the **Legendre–Fenchel transform / convex conjugate** $f^*(s)=\sup_x(\langle s,x\rangle-f(x))$ for deriving duals.

## Key concepts introduced
- [[gradient-descent]] — first-order descent; step-size, momentum
- [[stochastic-gradient-descent]] — mini-batch unbiased gradient estimates
- [[lagrange-multipliers]] — constrained optimization, Lagrangian, Lagrangian duality
- [[convex-optimization]] — convex sets, convex functions, Jensen's inequality, strong duality
- [[linear-programming]] — linear objective + linear inequality constraints
- [[quadratic-programming]] — convex quadratic objective + affine constraints
- [[convex-conjugate]] — Legendre–Fenchel transform, dual via conjugates

## Notable results / theorems
- **Descent guarantee:** for small $\gamma\ge 0$, $f(\mathbf{x}_1)\le f(\mathbf{x}_0)$ when stepping along $-\nabla f$ (Eq. 7.5).
- **Minimax inequality (7.23):** $\max_y\min_x \varphi(x,y)\le \min_x\max_y \varphi(x,y)$.
- **Weak duality (7.27):** primal value $\ge$ dual value, always.
- **Strong duality:** for convex $f$ and convex constraint sets, dual optimum $=$ primal optimum.
- **First-order convexity (7.31):** $f(y)\ge f(x)+\nabla f(x)^\top(y-x)$; **second-order:** $\nabla^2 f\succeq 0$.
- **Jensen's inequality (7.30):** $f(\theta x+(1-\theta)y)\le \theta f(x)+(1-\theta)f(y)$.
- **Conjugate of a sum (7.63):** separable losses have separable conjugates, $L^*(z)=\sum_i \ell_i^*(z_i)$.
- Cited side-result: **Abel–Ruffini theorem** (no algebraic solution for degree $\ge 5$ polynomials) motivates iterative methods.

## What this changes in the wiki
- First chapter from **Part II setup** in the wiki: introduces the optimization machinery that PCA (Ch. 10), density estimation (Ch. 11), and classification/SVM (Ch. 12) all rely on.
- Builds directly on **gradients/Hessian** from Ch. 5 ([[gradient]], [[hessian]] — TODO pages) and **singular values / condition number** from Ch. 4 ([[singular-value-decomposition]] — TODO).
- No contradictions with existing pages; extends the Ch. 2 idea of **closure** (vector spaces) to **convexity-preserving operations**.
