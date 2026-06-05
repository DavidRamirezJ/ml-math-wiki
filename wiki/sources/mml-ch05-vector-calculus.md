---
title: "MML Ch. 5 — Vector Calculus"
type: source
tags: [calculus, differentiation, optimization, source]
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch. 5 — Vector Calculus

- **Raw file:** `raw/mml-book.pdf` §5, printed pp. 139–171 (PDF pp. 145–177)
- **Type:** book chapter
- **Ingested:** 2026-06-04

This chapter develops the differential calculus needed to train ML models:
how to differentiate functions of one variable, of many variables, and of
vectors/matrices; how to package those derivatives (gradient, Jacobian,
Hessian); how to approximate functions locally (Taylor series); and how to
compute gradients efficiently at scale (backpropagation / automatic
differentiation). The unifying engine throughout is the **chain rule**.

## Section coverage

| § | Title | Wiki page(s) |
|---|---|---|
| 5.1 | Differentiation of Univariate Functions (difference quotient, derivative, differentiation rules) | [[derivative]] |
| 5.1.1 | Taylor Series | [[taylor-series]] |
| 5.1.2 | Differentiation Rules (product, quotient, sum, chain) | [[derivative]], [[chain-rule]] |
| 5.2 | Partial Differentiation and Gradients | [[gradient]], [[partial-derivative]] |
| 5.2.1 | Basic Rules of Partial Differentiation | [[chain-rule]] |
| 5.2.2 | Chain Rule (multivariate) | [[chain-rule]] |
| 5.3 | Gradients of Vector-Valued Functions (the Jacobian) | [[jacobian]] |
| 5.4 | Gradients of Matrices (gradient tensors) | [[jacobian]] (gradient-tensor section) |
| 5.5 | Useful Identities for Computing Gradients | [[gradient]] (identities table) |
| 5.6 | Backpropagation and Automatic Differentiation | [[backpropagation]], [[automatic-differentiation]] |
| 5.7 | Higher-Order Derivatives (the Hessian) | [[hessian]] |
| 5.8 | Linearization and Multivariate Taylor Series | [[taylor-series]], [[hessian]] |
| 5.9 | Further Reading | — |

## TL;DR (5 bullets)
- The **[[derivative]]** is the limit of the difference quotient; it points in the direction of steepest ascent and is the local linearizer of a function.
- For multivariate $f:\mathbb R^n\to\mathbb R$ the derivative becomes the **[[gradient]]**, a row vector of **[[partial-derivative|partial derivatives]]**; for vector-valued $f:\mathbb R^n\to\mathbb R^m$ it becomes the **[[jacobian]]**, an $m\times n$ matrix.
- The **[[chain-rule]]** generalizes to a product of Jacobians; writing gradients as row vectors makes it literal matrix multiplication and makes dimensions "cancel."
- **[[taylor-series]]** approximate a smooth function by a polynomial built from its derivatives; the multivariate version uses the gradient (1st order) and the **[[hessian]]** (2nd order).
- **[[backpropagation]]** is reverse-mode **[[automatic-differentiation]]**: it computes gradients of a deep function composition (a neural network) by sweeping the chain rule backward through a computation graph, reusing intermediate results — gradient cost ≈ function cost.

## Key concepts introduced
- [[derivative]] — univariate differentiation, difference quotient, differentiation rules
- [[taylor-series]] — polynomial approximation from derivatives (uni- and multivariate)
- [[partial-derivative]] — derivative w.r.t. one variable, others held fixed
- [[gradient]] — row vector collecting all first-order partials of a scalar function
- [[jacobian]] — matrix collecting all first-order partials of a vector-valued function; its determinant rescales volumes
- [[chain-rule]] — differentiating compositions; multivariate form as matrix multiplication
- [[hessian]] — symmetric matrix of second-order partials; measures curvature
- [[backpropagation]] — efficient reverse-mode gradient computation in deep networks
- [[automatic-differentiation]] — evaluating exact gradients of programs via the chain rule on a computation graph

## Notable results / theorems
- **Definition 5.2 (Derivative):** $\dfrac{df}{dx}=\lim_{h\to0}\dfrac{f(x+h)-f(x)}{h}$.
- **Differentiation rules (5.29–5.32):** product, quotient, sum, and chain rules.
- **Definition 5.4 (Taylor series):** $T_\infty(x)=\sum_{k=0}^{\infty}\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k$; $x_0=0$ gives the Maclaurin series.
- **Definition 5.5 (Partial derivative) / (5.40) Gradient** as a $1\times n$ row vector.
- **Definition 5.6 (Jacobian):** $J=\dfrac{d\mathbf f}{d\mathbf x}\in\mathbb R^{m\times n}$, $J(i,j)=\partial f_i/\partial x_j$ (numerator layout). $|\det J|$ is the local volume-scaling factor (used in §6.7 change of variables).
- **Multivariate chain rule (5.48, 5.53):** $\frac{\partial}{\partial\mathbf x}(g\circ f)=\frac{\partial g}{\partial f}\frac{\partial f}{\partial\mathbf x}$ — a product of Jacobians.
- **Useful gradient identities (5.99–5.108):** e.g. $\frac{\partial \mathbf x^\top\mathbf a}{\partial \mathbf x}=\mathbf a^\top$, $\frac{\partial \mathbf x^\top B\mathbf x}{\partial \mathbf x}=\mathbf x^\top(B+B^\top)$, $\frac{\partial}{\partial X}\operatorname{tr}(f(X))=\operatorname{tr}(\partial f/\partial X)$.
- **Schwarz / Clairaut symmetry (5.146):** for twice-continuously-differentiable $f$, $\frac{\partial^2 f}{\partial x\partial y}=\frac{\partial^2 f}{\partial y\partial x}$, so the **Hessian (5.147)** is symmetric.
- **Backpropagation (5.115–5.118, 5.143–5.145):** recursive backward application of the chain rule over a computation graph.

## What this changes in the wiki
- First calculus chapter; introduces the differentiation machinery that Part II ML methods rely on. Connects forward to [[system-of-linear-equations|least-squares]] (Example 5.11), linear regression (Ch. 9), and optimization / gradient descent (Ch. 7).
- Reuses linear-algebra pages without redefining them: [[vector]], [[matrix]], [[linear-mapping]], [[basis-and-rank]], and the determinant (Ch. 4) — the Jacobian determinant is the determinant of a basis-change matrix when the transformation is linear (Example, §5.3).
- No contradictions with existing pages.
