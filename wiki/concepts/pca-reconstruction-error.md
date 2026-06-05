---
title: PCA — Projection / Reconstruction-Error Perspective
type: concept
tags: [pca, optimization, projection, dimensionality-reduction]
difficulty: advanced
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
---

# PCA — Projection / Reconstruction-Error Perspective

> **One-liner:** A second way to derive PCA: find the low-dimensional subspace whose shadows of the data points are, on average, as close to the originals as possible.

## 🟢 Intuition (Dummy)
Shine a light on a cloud of 3-D points so they cast shadows onto a flat sheet of paper. Each point
moves a little to land on the paper — that movement is the "error." Now tilt the paper until the
*total* movement is as small as possible. The paper you end up with is the principal subspace, and the
shadows are the compressed data. PCA from this angle is just: choose the flat surface that the data
falls onto with the least drop. The best drop is always **straight down** (perpendicular) — an
orthogonal projection.

## 🟡 Mechanics (Practitioner)
With an orthonormal basis $(\mathbf b_1,\dots,\mathbf b_M)$ of the principal subspace $U$, the
reconstruction is $\tilde{\mathbf x}_n=\sum_{m=1}^M z_{mn}\mathbf b_m=B\mathbf z_n$. The objective is the
**average squared reconstruction error**:
$$ J_M=\frac1N\sum_{n=1}^N\|\mathbf x_n-\tilde{\mathbf x}_n\|^2 . $$

**Step 1 — optimal coordinates.** For fixed basis, $\partial J_M/\partial z_{in}=0$ gives
$$ z_{in}=\mathbf b_i^\top\mathbf x_n , $$
i.e. the coordinates are the **orthogonal projection** of $\mathbf x_n$ onto $\mathbf b_i$. So
$\tilde{\mathbf x}_n=\big(\sum_{m\le M}\mathbf b_m\mathbf b_m^\top\big)\mathbf x_n=BB^\top\mathbf x_n$
(using $B^\top B=I$). The optimal linear projection is the orthogonal one.

**Step 2 — optimal basis.** The displacement lies entirely in the orthogonal complement:
$$ \mathbf x_n-\tilde{\mathbf x}_n=\sum_{j=M+1}^D(\mathbf b_j^\top\mathbf x_n)\,\mathbf b_j . $$
Plugging in and using the ONB property, the loss becomes (with $S$ the [[data-covariance-matrix]])
$$ J_M=\sum_{j=M+1}^D \mathbf b_j^\top S\,\mathbf b_j=\operatorname{tr}\!\Big(\Big(\textstyle\sum_{j>M}\mathbf b_j\mathbf b_j^\top\Big)S\Big). $$
This is the **variance of the data in the discarded directions**. Minimizing it ⇒ assign the smallest
$D-M$ eigenvalues to the complement, so the **principal subspace is spanned by the top-$M$ eigenvectors of $S$**, giving
$$ J_M=\sum_{j=M+1}^D\lambda_j . $$

**Equivalently:** minimizing reconstruction error = maximizing retained variance ([[pca-maximum-variance]]) — same eigenvector solution.

**ML usage:** identifies PCA as the **optimal linear auto-encoder**: encoder $B^\top$, decoder $B$,
loss $\frac1N\sum_n\|\mathbf x_n-BB^\top\mathbf x_n\|^2$. See [[principal-component-analysis]], [[auto-encoder]].

## 🔴 Theory (Expert)
- **Why orthogonal is optimal.** Because $U^\perp=\operatorname{span}[\mathbf b_{M+1},\dots,\mathbf b_D]$
  is the orthogonal complement, the coordinates $\zeta_m$ of $\mathbf x$ and $z_m$ of $\tilde{\mathbf x}$
  coincide for $m\le M$; the residual lives purely in $U^\perp$ (Fig. 10.9). The orthogonal projector is
  $\tilde{\mathbf x}=B(B^\top B)^{-1}B^\top\mathbf x=BB^\top\mathbf x$ when $B$ is orthonormal.
- **Low-rank approximation of the identity.** $J_M=\frac1N\sum_n\|(I-BB^\top)\mathbf x_n\|^2$, so PCA finds
  the best rank-$M$ projector $BB^\top\approx I$ (in the data metric). $BB^\top$ is symmetric, rank $M$,
  built from rank-one pieces $\mathbf b_m\mathbf b_m^\top$; the complement projector $\sum_{j>M}\mathbf b_j\mathbf b_j^\top$
  has rank $D-M$. Connects to Eckart–Young / truncated SVD.
- **Trace formulation.** Using cyclicity and linearity of $\operatorname{tr}$, $J_M=\operatorname{tr}((I-BB^\top)S)$,
  cleanly exposing the role of the discarded spectrum.
- This perspective is due to Pearson (1901), who introduced PCA via the reconstruction-error (best-fit line/plane) formulation.

## Connections
- Builds on: [[orthogonal-projection]], [[data-covariance-matrix]], [[basis-and-rank]], [[singular-value-decomposition]]
- Leads to: [[principal-component-analysis]], [[auto-encoder]]
- Used by: [[principal-component-analysis]], [[dimensionality-reduction]]
- People: [[karl-pearson]]

## Sources
- `raw/mml-book.pdf` §10.3 (Eqs. 10.26–10.44), printed pp. 325–332 (PDF pp. 331–338).
