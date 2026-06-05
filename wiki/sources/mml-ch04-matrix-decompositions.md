---
title: "MML Ch.4 — Matrix Decompositions"
type: source
tags: [linear-algebra, matrix-decomposition, source]
sources: []
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch.4 — Matrix Decompositions

- **Raw file:** `raw/mml-book.pdf`, Chapter 4, printed pp. 98–138 (PDF pp. 103–143).
- **Type:** book chapter
- **Ingested:** 2026-06-04

This chapter covers three aspects of matrices: how to **summarize** them (determinant, trace,
eigenvalues), how to **decompose** them (Cholesky, eigendecomposition, SVD), and how to use those
decompositions for **approximation** (low-rank / Eckart–Young). The unifying analogy: matrix
decomposition is like factoring an integer ($21 = 7\cdot 3$) into interpretable pieces.

## Section coverage

| § | Title | Printed pp. | PDF pp. | Wiki page(s) |
|---|---|---|---|---|
| 4.1 | Determinant and Trace | 99–104 | 104–109 | [[determinant]], [[trace]], [[characteristic-polynomial]] |
| 4.2 | Eigenvalues and Eigenvectors | 105–114 | 110–119 | [[eigenvalues-and-eigenvectors]], [[characteristic-polynomial]], [[spectral-theorem]] |
| 4.3 | Cholesky Decomposition | 114–115 | 119–120 | [[cholesky-decomposition]] |
| 4.4 | Eigendecomposition and Diagonalization | 115–119 | 120–124 | [[eigendecomposition]] |
| 4.5 | Singular Value Decomposition | 119–128 | 124–133 | [[singular-value-decomposition]] |
| 4.6 | Matrix Approximation | 129–133 | 134–138 | [[low-rank-approximation]], [[singular-value-decomposition]] |
| 4.7 | Matrix Phylogeny | 134–135 | 139–140 | [[singular-value-decomposition]], [[eigendecomposition]] (taxonomy) |
| 4.8 | Further Reading | 135–136 | 140–141 | — |

## TL;DR (5 bullets)
- The **[[determinant]]** is a single number that measures the signed volume scaling of a square
  matrix; $\det(A)\neq 0$ iff $A$ is invertible iff it has full rank. The **[[trace]]** is the sum
  of diagonal entries and is basis-invariant.
- **[[eigenvalues-and-eigenvectors]]** ($A\mathbf x = \lambda\mathbf x$) reveal the directions a
  linear map only stretches; eigenvalues are the roots of the **[[characteristic-polynomial]]**
  $p_A(\lambda)=\det(A-\lambda I)$. $\det A=\prod\lambda_i$ and $\operatorname{tr}A=\sum\lambda_i$.
- The **[[cholesky-decomposition]]** $A=LL^\top$ is a matrix "square root" for symmetric positive
  definite matrices; **[[eigendecomposition]]** $A=PDP^{-1}$ diagonalizes a matrix in its eigenbasis
  (when one exists), with $A=PDP^\top$ for symmetric $A$ via the **[[spectral-theorem]]**.
- The **[[singular-value-decomposition]]** $A=U\Sigma V^\top$ exists for *every* matrix (square or
  not) — two orthonormal bases linked by non-negative singular values; it is the "fundamental
  theorem of linear algebra."
- The **[[low-rank-approximation]]** built from the top-$k$ singular triples is, by the
  **Eckart–Young theorem**, the *best* rank-$k$ approximation in spectral norm — the engine behind
  compression, PCA, and dimensionality reduction.

## Key concepts introduced
- [[determinant]] — signed volume, invertibility test, Laplace expansion, Sarrus' rule.
- [[trace]] — sum of diagonal, cyclic invariance, basis-invariant.
- [[characteristic-polynomial]] — $\det(A-\lambda I)$, whose roots are the eigenvalues.
- [[eigenvalues-and-eigenvectors]] — eigenvalue equation, eigenspaces, algebraic vs geometric multiplicity, defective matrices.
- [[spectral-theorem]] — symmetric matrices have an orthonormal eigenbasis with real eigenvalues.
- [[cholesky-decomposition]] — $A=LL^\top$ for SPD matrices.
- [[eigendecomposition]] — diagonalization $A=PDP^{-1}$ via the eigenbasis.
- [[singular-value-decomposition]] — $A=U\Sigma V^\top$ for any matrix.
- [[low-rank-approximation]] — truncated SVD and the Eckart–Young optimality theorem.

## Notable results / theorems
- **Thm 4.1 / 4.3:** $A$ invertible $\iff \det(A)\neq 0 \iff \operatorname{rk}(A)=n$.
- **Thm 4.2 (Laplace expansion):** recursive cofactor formula for the determinant.
- **Thm 4.8:** $\lambda$ is an eigenvalue $\iff$ it is a root of $p_A(\lambda)$.
- **Thm 4.12:** $n$ distinct eigenvalues $\Rightarrow$ linearly independent eigenvectors (a basis).
- **Thm 4.14:** $A^\top A$ is symmetric positive semidefinite for any $A$ (positive definite if $\operatorname{rk}A=n$).
- **Thm 4.15 (Spectral theorem):** symmetric $A$ has an ONB of real-eigenvalue eigenvectors $\Rightarrow A=PDP^\top$.
- **Thm 4.16 / 4.17:** $\det A=\prod_i\lambda_i$, $\operatorname{tr}A=\sum_i\lambda_i$.
- **Thm 4.18 (Cholesky):** unique $A=LL^\top$ for SPD $A$.
- **Thm 4.20 (Eigendecomposition):** $A=PDP^{-1}$ exists $\iff$ eigenvectors form a basis.
- **Thm 4.22 (SVD theorem):** $A=U\Sigma V^\top$ exists for every $A\in\mathbb R^{m\times n}$.
- **Thm 4.24:** $\|A\|_2=\sigma_1$ (spectral norm = largest singular value).
- **Thm 4.25 (Eckart–Young):** truncated SVD is the optimal rank-$k$ approximation; the error is $\sigma_{k+1}$.

## What this changes in the wiki
- This is the first chapter ingested after Ch.2; it builds heavily on [[matrix]], [[vector-space]],
  [[linear-mapping]], [[linear-independence]], [[basis-and-rank]], [[gaussian-elimination]],
  [[system-of-linear-equations]]. No contradictions with existing pages.
- It depends on **inner product / norm**, **orthogonal/orthonormal bases**, and **positive
  definiteness** from Ch.3 — [[inner-product]], [[norm]], [[orthogonal-matrix]],
  [[symmetric-positive-definite-matrix|positive definite matrix]],
  [[gram-schmidt-orthogonalization|Gram-Schmidt]] — all ingested in the same batch, so these resolve.
- Forward links into Ch.10 ([[principal-component-analysis]]) and Ch.6 ([[gaussian-distribution]]) now resolve too.
