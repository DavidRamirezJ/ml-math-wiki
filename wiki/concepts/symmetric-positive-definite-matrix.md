---
title: Symmetric, Positive Definite Matrix
type: concept
tags: [analytic-geometry, inner-product, linear-algebra, matrix-decomposition]
difficulty: intermediate
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Symmetric, Positive Definite Matrix

> **One-liner:** A symmetric positive-definite matrix is a square matrix that always returns a positive number when you sandwich any nonzero vector around it — it is exactly what an inner product looks like in coordinates.

## 🟢 Intuition (Dummy)
Imagine a machine that takes a vector, runs it through a matrix, and reports back a single number
($x^\top A x$). A "positive definite" machine has a nice property: no matter which nonzero vector
you feed it, the answer is always positive — never zero, never negative. "Symmetric" means the
machine treats the two slots the same way. Such machines are the well-behaved ones: they bend
space without flipping it inside out, they have a meaningful notion of "length," and they are
exactly the matrices that encode a valid way of measuring angles and distances.

## 🟡 Mechanics (Practitioner)
**Definition 3.4.** A symmetric matrix $A\in\mathbb R^{n\times n}$ (i.e. $A=A^\top$) is
**positive definite** if
$$ x^\top A x > 0 \quad \text{for all } x\in\mathbb R^n\setminus\{\mathbf 0\}. $$
If only $x^\top A x \ge 0$ holds, $A$ is **positive semidefinite**.

**Link to inner products (Thm 3.5).** $\langle x,y\rangle = \hat x^\top A\,\hat y$ defines an
[[inner-product]] **iff** $A$ is symmetric positive definite. So SPD matrices and inner products
are two views of the same object.

**Worked example (§3.2).**
$$ A_1 = \begin{bmatrix}9 & 6\\ 6 & 5\end{bmatrix},\qquad A_2 = \begin{bmatrix}9 & 6\\ 6 & 3\end{bmatrix}. $$
For $A_1$: $x^\top A_1 x = 9x_1^2 + 12x_1x_2 + 5x_2^2 = (3x_1+2x_2)^2 + x_2^2 > 0$ — **positive definite**.
For $A_2$: $x^\top A_2 x = (3x_1+2x_2)^2 - x_2^2$, which is $<0$ at $x=\big[2,-3\big]^\top$ — **not** positive definite.

**Two quick consequences (if $A$ is SPD):**
- $\ker(A) = \{\mathbf 0\}$, so $A$ is invertible ($Ax\ne\mathbf 0$ for $x\ne\mathbf 0$).
- Every diagonal entry is positive: $a_{ii} = e_i^\top A e_i > 0$.

**ML usage.** Covariance matrices are positive semidefinite (positive definite when full-rank).
Kernel/Gram matrices must be positive semidefinite to be valid (§12.4). SPD structure enables the
**Cholesky decomposition** (Ch. 4) for fast, stable solves and sampling from Gaussians.

## 🔴 Theory (Expert)
- **Equivalent characterizations:** $A$ symmetric is positive definite $\iff$ all eigenvalues
  $>0$ $\iff$ all leading principal minors $>0$ (Sylvester's criterion) $\iff$ $A=LL^\top$ for a
  unique lower-triangular $L$ with positive diagonal (Cholesky).
- In a projection context, $B^\top B$ is positive definite exactly when $B$ has full column rank;
  this is why the **normal equation** in [[orthogonal-projection]] is solvable and the pseudo-inverse
  $(B^\top B)^{-1}B^\top$ exists. A "jitter"/ridge term $\epsilon I$ is added in practice to force
  positive definiteness and numerical stability.
- Returned to in §4.3 (matrix decompositions); positive *semidefiniteness* is the defining property
  of kernels (§12.4).

## Connections
- Builds on: [[matrix]], [[inner-product]]
- Leads to: [[orthogonal-projection]] (normal equation), Cholesky decomposition (Ch. 4), kernels (Ch. 12)
- Used by: covariance matrices, Gaussian distributions, kernel methods
- People: —

## Sources
- `raw/mml-book.pdf` §3.2.3 (Def 3.4, Thm 3.5, Example 3.4), printed pp. 73–74 (PDF pp. 78–79).
