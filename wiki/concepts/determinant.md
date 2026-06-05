---
title: Determinant
type: concept
tags: [linear-algebra, matrix-decomposition]
difficulty: intermediate
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Determinant

> **One-liner:** The determinant is a single number that tells you how much a matrix stretches or shrinks volume — and if it's zero, the matrix squashes space flat and can't be undone.

## 🟢 Intuition (Dummy)
Take a unit square and feed its corners through a [[matrix]]. The square turns into a parallelogram.
The **determinant** is the *area* of that parallelogram (with a sign telling you if it got flipped
over). In 3D it's the *volume* of the resulting box. If the matrix flattens the square into a line —
zero area — the determinant is 0, and that flattening can never be reversed (the matrix has no
inverse). So the determinant is a one-number health check: nonzero means "reversible and full-bodied,"
zero means "collapsed."

## 🟡 Mechanics (Practitioner)
The determinant $\det(A)$ (also written $|A|$, not to be confused with absolute value) maps a square
matrix $A\in\mathbb R^{n\times n}$ to a real number.

**Small cases:**
$$\det(a_{11})=a_{11},\qquad \det\!\begin{pmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{pmatrix}=a_{11}a_{22}-a_{12}a_{21}.$$

**Sarrus' rule ($3\times 3$):**
$$\det(A)=a_{11}a_{22}a_{33}+a_{21}a_{32}a_{13}+a_{31}a_{12}a_{23}-a_{31}a_{22}a_{13}-a_{11}a_{32}a_{23}-a_{21}a_{12}a_{33}.$$

**Triangular matrices:** $\det(T)=\prod_{i=1}^n T_{ii}$ (product of the diagonal). This is the key to
computing determinants fast: use [[gaussian-elimination]] to reach row-echelon (triangular) form,
then multiply the diagonal.

**Laplace expansion (Thm 4.2)** — recursive cofactor expansion along column $j$:
$$\det(A)=\sum_{k=1}^n(-1)^{k+j}\,a_{kj}\,\det(A_{k,j}),$$
where $A_{k,j}$ is $A$ with row $k$ and column $j$ deleted. $\det(A_{k,j})$ is a *minor*;
$(-1)^{k+j}\det(A_{k,j})$ is a *cofactor*.

**Worked example (Laplace along first row):**
$$A=\begin{pmatrix}1&2&3\\3&1&2\\0&0&1\end{pmatrix},\quad \det(A)=1(1-0)-2(3-0)+3(0-0)=-5.$$

**Key properties:** $\det(AB)=\det(A)\det(B)$; $\det(A^\top)=\det(A)$; $\det(A^{-1})=1/\det(A)$;
$\det(\lambda A)=\lambda^n\det(A)$; swapping two rows flips the sign; adding a multiple of one row to
another leaves it unchanged; [[linear-mapping|similar matrices]] share a determinant (so it is
basis-invariant).

**ML usage:** appears in the log-likelihood of a multivariate Gaussian (the $\log\det\Sigma$ term),
in change-of-variables for normalizing flows, and as a theoretical test of invertibility — though in
practice numerical methods ([[gaussian-elimination]], factorizations) have largely superseded explicit
determinant computation.

## 🔴 Theory (Expert)
- **Thm 4.1:** $A\in\mathbb R^{n\times n}$ is invertible $\iff \det(A)\neq 0$.
- **Thm 4.3:** $\det(A)\neq 0 \iff \operatorname{rk}(A)=n$ (full rank). These three notions —
  invertibility, nonzero determinant, full rank — coincide for square matrices.
- **Geometric meaning (Examples 4.1–4.2):** $|\det([\mathbf a_1,\dots,\mathbf a_n])|$ is the
  $n$-dimensional volume of the parallelepiped spanned by the columns; the *sign* encodes the
  orientation relative to the standard basis. Linearly dependent columns ⇒ degenerate (zero-volume)
  shape ⇒ $\det=0$.
- **Spectral link (Thm 4.16):** $\det(A)=\prod_{i=1}^n\lambda_i$, the product of the
  [[eigenvalues-and-eigenvectors|eigenvalues]] (with multiplicity, in $\mathbb C$). Via the
  [[eigendecomposition]] $A=PDP^{-1}$: $\det(A)=\det(P)\det(D)\det(P^{-1})=\det(D)=\prod_i d_{ii}$.
- The determinant is the unique alternating multilinear function of the columns that is $1$ on the
  identity. The [[characteristic-polynomial]] $p_A(\lambda)=\det(A-\lambda I)$ is built directly from
  it.

## Connections
- Builds on: [[matrix]], [[gaussian-elimination]], [[basis-and-rank]], [[linear-independence]]
- Leads to: [[characteristic-polynomial]], [[eigenvalues-and-eigenvectors]], [[trace]]
- Used by: [[cholesky-decomposition]], [[eigendecomposition]], multivariate Gaussian likelihood
- People: [[pierre-simon-laplace]] (Laplace expansion)

## Sources
- `raw/mml-book.pdf` §4.1, Defs/Thms 4.1–4.3, Examples 4.1–4.3, printed pp. 99–103 (PDF pp. 104–108).
