---
title: Jacobian
type: concept
tags: [calculus, differentiation, multivariate, linear-algebra]
difficulty: intermediate
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# Jacobian

> **One-liner:** The Jacobian is the full grid of partial slopes for a function that outputs many numbers — the matrix version of a derivative.

## 🟢 Intuition (Dummy)
The [[gradient]] handles functions that spit out a *single* number. But many
functions output a whole list — a coordinate change turns $(x_1,x_2)$ into
$(y_1,y_2)$, a network layer turns one vector into another. To describe how *all*
the outputs respond to *all* the inputs, you need a whole table of slopes: how
much does output 1 change when I nudge input 1, when I nudge input 2, and so on
for every output. That table is the **Jacobian**. A neat bonus: its
"size-changing power" (the determinant) tells you how much it stretches or
shrinks areas and volumes.

## 🟡 Mechanics (Practitioner)
For a vector-valued $\mathbf f:\mathbb R^n\to\mathbb R^m$, the **Jacobian**
(Def 5.6) is the $m\times n$ matrix of all first-order partials:
$$
J=\nabla_{\mathbf x}\mathbf f=\frac{d\mathbf f(\mathbf x)}{d\mathbf x}
=\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n}\\
\vdots & & \vdots\\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}\in\mathbb R^{m\times n},\qquad J(i,j)=\frac{\partial f_i}{\partial x_j}.
$$
(Numerator layout: outputs index rows, inputs index columns.) Special cases:
$f:\mathbb R^n\to\mathbb R$ gives a $1\times n$ row ([[gradient]]); $f:\mathbb R\to\mathbb R^m$ gives an $m\times1$ column; $f:\mathbb R\to\mathbb R$ gives a scalar.

**Worked example (linear map).** For $\mathbf f(\mathbf x)=A\mathbf x$ with
$A\in\mathbb R^{m\times n}$, since $f_i=\sum_j A_{ij}x_j$ we get
$\partial f_i/\partial x_j = A_{ij}$, so
$$\frac{d\mathbf f}{d\mathbf x}=A.$$
The Jacobian of a [[linear-mapping]] is just its matrix.

**Jacobian determinant — volume scaling (§5.3).** Transforming variables by
$y_1=-2x_1+x_2,\ y_2=x_1+x_2$ gives
$$J=\begin{bmatrix}-2&1\\[2pt]1&1\end{bmatrix},\qquad |\det J| = |-3| = 3,$$
so the transformation scales areas by a factor of $3$. When the map is linear,
$J$ equals the basis-change matrix (a [[linear-mapping]] result); when nonlinear,
$J$ is the *local* linear approximation and $|\det J|$ the *local* volume factor.

**ML usage:** $|\det J|$ is the correction term in the change-of-variables
formula for probability densities (used in normalizing flows and the
reparametrization trick, §6.7). Layer-wise Jacobians are the factors multiplied
in [[backpropagation]].

## 🔴 Theory (Expert)
- **Gradient tensors (§5.4).** Differentiating a *matrix* $A\in\mathbb R^{m\times n}$
  w.r.t. another matrix $B\in\mathbb R^{p\times q}$ yields a 4-D tensor
  $J_{ijkl}=\partial A_{ij}/\partial B_{kl}$. Using the [[linear-mapping|isomorphism]]
  $\mathbb R^{m\times n}\cong\mathbb R^{mn}$ you may **flatten** to a plain
  $mn\times pq$ Jacobian matrix so the [[chain-rule]] reduces to matrix
  multiplication; otherwise you must track which tensor dimensions to sum over.
- The [[chain-rule]] for compositions is the **product of Jacobians**
  (5.48): $\frac{d}{d\mathbf x}(g\circ f)=\frac{\partial g}{\partial f}\frac{\partial f}{\partial\mathbf x}$ — dimensions match exactly because the inner Jacobian's rows are the outer's columns.
- The numerator layout (book convention) vs. denominator layout (transpose) is a
  pure bookkeeping choice; numerator layout keeps the chain rule left-to-right.

## Connections
- Builds on: [[partial-derivative]], [[gradient]], [[matrix]], [[linear-mapping]]
- Leads to: [[chain-rule]], [[hessian]], change of variables for densities (Ch. 6)
- Used by: [[backpropagation]], normalizing flows, the reparametrization trick
- People: [[carl-gustav-jacob-jacobi]]

## Sources
- `raw/mml-book.pdf` §5.3 (Def 5.6), §5.4, printed pp. 149–158 (PDF pp. 155–164).
