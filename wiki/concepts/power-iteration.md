---
title: Power Iteration
type: concept
tags: [linear-algebra, eigenvalues, numerical-methods, algorithms]
difficulty: intermediate
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
---

# Power Iteration

> **One-liner:** A dead-simple loop — multiply by the matrix, normalize, repeat — that homes in on the single most dominant direction of a matrix.

## 🟢 Intuition (Dummy)
Pick any starting arrow. Hit it with the matrix; it gets stretched most in the matrix's "strongest"
direction, a bit in the others. Rescale it back to length 1 so it doesn't blow up. Hit it again. Each
round, the strongest direction wins a bigger share, and the weaker ones fade. After enough rounds the
arrow points almost exactly along the matrix's top direction. That's it — no fancy algebra, just
multiply-and-normalize until it settles. Google's original PageRank used this idea to rank web pages.

## 🟡 Mechanics (Practitioner)
To find the eigenvector of a matrix $S$ with the **largest** eigenvalue, start from a random
$\mathbf x_0$ (not in the null space of $S$) and iterate:
$$ \mathbf x_{k+1}=\frac{S\,\mathbf x_k}{\|S\,\mathbf x_k\|},\qquad k=0,1,2,\dots $$
Each step keeps $\|\mathbf x_k\|=1$. The sequence converges to the dominant eigenvector of $S$; the
associated eigenvalue is recovered as $\lambda\approx\mathbf x_k^\top S\mathbf x_k$ (Rayleigh quotient).

**Why use it:** in [[principal-component-analysis|PCA]] you often need only the top few eigenvectors of
the [[data-covariance-matrix]], not the full decomposition. Computing all $D$ eigenvectors and throwing
most away is wasteful; iterating directly for the top ones is far cheaper. In the extreme of needing only
the first PC, power iteration is the tool. Deflation (subtract the found component) then yields the next.

**ML usage:** top-PC extraction, PageRank, dominant-mode analysis, spectral methods where only the
leading eigenvector matters.

## 🔴 Theory (Expert)
- **Convergence.** Write $\mathbf x_0=\sum_i c_i\mathbf v_i$ in the eigenbasis with eigenvalues
  $|\lambda_1|>|\lambda_2|\ge\dots$. Then $S^k\mathbf x_0=\sum_i c_i\lambda_i^k\mathbf v_i$, and after
  normalization the $\mathbf v_1$ term dominates because $(\lambda_i/\lambda_1)^k\to0$ for $i\ge2$.
  Convergence is **linear** with rate $|\lambda_2/\lambda_1|$ — slow when the top two eigenvalues are close.
- **Requirements:** a strictly dominant eigenvalue and $c_1\neq0$ (generic for random $\mathbf x_0$). If $S$
  is invertible it suffices that $\mathbf x_0\neq\mathbf 0$.
- **Why iterate at all.** Eigenvalues are roots of the characteristic polynomial, but by the
  **Abel–Ruffini theorem** (Ruffini 1799; Abel 1826) there is no algebraic root formula for polynomials of
  degree $\ge 5$, so matrices larger than $4\times4$ *require* iterative numerical methods. Modern libraries
  (`np.linalg.eigh`, `np.linalg.svd`) use such methods under the hood.
- **Relatives:** inverse iteration (top eigenvector of $S^{-1}$ → smallest of $S$), shifted/Rayleigh-quotient
  iteration (faster, cubic near convergence), Lanczos/Arnoldi (Krylov subspace methods for several eigenpairs).

## Connections
- Builds on: [[eigenvalues-and-eigenvectors]], [[matrix]], [[data-covariance-matrix]]
- Leads to: Krylov methods, inverse/shifted iteration
- Used by: [[principal-component-analysis]], PageRank, spectral methods
- People: —

## Sources
- `raw/mml-book.pdf` §10.4.2 (Eq. 10.52), printed pp. 334–335 (PDF pp. 340–341).
