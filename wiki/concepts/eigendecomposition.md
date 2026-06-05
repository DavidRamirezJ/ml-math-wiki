---
title: Eigendecomposition and Diagonalization
type: concept
tags: [linear-algebra, matrix-decomposition]
difficulty: advanced
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Eigendecomposition and Diagonalization

> **One-liner:** If you describe a matrix using its own special stretch-directions as the coordinate axes, the matrix becomes a simple diagonal "just multiply each axis by a number."

## 🟢 Intuition (Dummy)
A general [[matrix]] mixes up coordinates in a messy way. But if you switch to a coordinate system
built from the matrix's own [[eigenvalues-and-eigenvectors|eigenvectors]] (its natural axes), the
matrix becomes trivially simple: it just stretches each axis by its eigenvalue. **Diagonalization**
is exactly this change of viewpoint. The recipe $A=PDP^{-1}$ says: *step into the eigenbasis*
($P^{-1}$), *stretch along each axis* ($D$, the diagonal of eigenvalues), *step back out* ($P$).
This makes hard operations like raising a matrix to the 100th power almost free.

## 🟡 Mechanics (Practitioner)
**Definition (Def 4.19):** $A\in\mathbb R^{n\times n}$ is **diagonalizable** if it is
[[linear-mapping|similar]] to a diagonal matrix — there is an invertible $P$ with $D=P^{-1}AP$.

**Theorem 4.20 (Eigendecomposition):**
$$A=PDP^{-1}$$
exists **if and only if** the eigenvectors of $A$ form a basis of $\mathbb R^n$. Then the columns of
$P=[\mathbf p_1,\dots,\mathbf p_n]$ are the eigenvectors and $D=\operatorname{diag}(\lambda_1,\dots,\lambda_n)$
holds the matching eigenvalues. (This follows from $AP=PD$, i.e. $A\mathbf p_i=\lambda_i\mathbf p_i$ for each column.)

**Symmetric case:** by the [[spectral-theorem]], a symmetric $A$ is *always* diagonalizable with an
**orthogonal** $P$, so $A=PDP^\top$ (Thm 4.21) — the inverse is just the transpose.

**Worked example (Example 4.11):** $A=\tfrac12\begin{pmatrix}5&-2\\-2&5\end{pmatrix}$ has eigenvalues
$\lambda_1=\tfrac72,\ \lambda_2=\tfrac32$ with orthonormal eigenvectors
$\mathbf p_1=\tfrac1{\sqrt2}(1,-1)^\top,\ \mathbf p_2=\tfrac1{\sqrt2}(1,1)^\top$. Then
$$P=\tfrac1{\sqrt2}\begin{pmatrix}1&1\\-1&1\end{pmatrix},\quad P^{-1}AP=\begin{pmatrix}\tfrac72&0\\0&\tfrac32\end{pmatrix}=D,$$
and since $P$ is orthogonal, $P^{-1}=P^\top$.

**Why it's useful — cheap matrix powers and determinants:**
$$A^k=(PDP^{-1})^k=PD^kP^{-1},\qquad D^k=\operatorname{diag}(\lambda_1^k,\dots,\lambda_n^k),$$
$$\det(A)=\det(D)=\prod_i d_{ii}.$$

**ML usage:** matrix powers appear in Markov chains and graph diffusion; eigendecomposition of
covariance matrices is exactly [[principal-component-analysis]]; it powers spectral clustering and
the analysis of recurrent dynamics. The geometric reading (Fig. 4.7): $P^{-1}$ rotates into the
eigenbasis, $D$ scales, $P$ rotates back.

## 🔴 Theory (Expert)
- **Existence condition:** $P$ must be invertible, i.e. the $n$ eigenvectors must be linearly
  independent — equivalently $A$ is **non-defective** (geometric multiplicity = algebraic multiplicity
  for every eigenvalue). Distinct eigenvalues suffice (Thm 4.12) but are not necessary.
- **Defective matrices cannot be diagonalized** over the reals (e.g. $\begin{pmatrix}2&1\\0&2\end{pmatrix}$).
  The **Jordan normal form** generalizes the decomposition to such cases (beyond this chapter).
- **Diagonalizability vs. invertibility are independent:** $\operatorname{diag}(1,0)$ is diagonalizable
  but singular; a real rotation is invertible but not diagonalizable over $\mathbb R$ (complex
  eigenvalues). See the matrix phylogeny (Fig. 4.13).
- Contrast with the [[singular-value-decomposition]]: eigendecomposition uses one (possibly
  non-orthogonal) basis applied and undone *within one space* and requires a square, non-defective
  matrix; the SVD uses two *orthonormal* bases across domain and codomain and exists for every matrix.
  For symmetric matrices the two coincide.

## Connections
- Builds on: [[eigenvalues-and-eigenvectors]], [[characteristic-polynomial]], [[spectral-theorem]], [[basis-and-rank]], [[linear-mapping]]
- Leads to: [[singular-value-decomposition]], [[low-rank-approximation]]
- Used by: [[principal-component-analysis]], spectral clustering, matrix-power computations, Markov chains
- People: —

## Sources
- `raw/mml-book.pdf` §4.4, Def 4.19, Thms 4.20–4.21, Example 4.11, printed pp. 115–119 (PDF pp. 120–124).
