---
title: PCA — Maximum Variance Perspective
type: concept
tags: [pca, optimization, eigenvalues, dimensionality-reduction]
difficulty: advanced
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
---

# PCA — Maximum Variance Perspective

> **One-liner:** One way to derive PCA: pick the directions that make the compressed data spread out as much as possible, because spread = information.

## 🟢 Intuition (Dummy)
If you must summarize a cloud of points with a single number per point — their position along *one*
line — which line should you choose? Choose the line along which the points are most spread out.
A line where everyone clusters tightly would throw away the differences between points; a line where
they're spread far apart preserves who's-far-from-whom. "Most spread" means "most information kept."
PCA's first idea is exactly this: find the direction of maximum spread, then the next one
perpendicular to it, and so on.

## 🟡 Mechanics (Practitioner)
Project centered data onto a unit direction $\mathbf b_1$ (so $\|\mathbf b_1\|=1$). The first
coordinate is $z_{1n}=\mathbf b_1^\top\mathbf x_n$, and its variance is
$$ V_1=\frac1N\sum_{n=1}^N z_{1n}^2=\frac1N\sum_n \mathbf b_1^\top\mathbf x_n\mathbf x_n^\top\mathbf b_1=\mathbf b_1^\top S\mathbf b_1, $$
where $S$ is the [[data-covariance-matrix]]. We must constrain $\|\mathbf b_1\|=1$ (otherwise just
scaling $\mathbf b_1$ inflates $V_1$). So:
$$ \max_{\mathbf b_1}\ \mathbf b_1^\top S\mathbf b_1 \quad\text{subject to}\quad \|\mathbf b_1\|^2=1. $$

**Solve with a Lagrangian** (see [[lagrange-multipliers]]):
$$ \mathcal L(\mathbf b_1,\lambda_1)=\mathbf b_1^\top S\mathbf b_1+\lambda_1\!\left(1-\mathbf b_1^\top\mathbf b_1\right). $$
Setting $\partial\mathcal L/\partial\mathbf b_1=0$ and $\partial\mathcal L/\partial\lambda_1=0$:
$$ S\mathbf b_1=\lambda_1\mathbf b_1,\qquad \mathbf b_1^\top\mathbf b_1=1. $$
This is an **eigenproblem**: $\mathbf b_1$ is an eigenvector of $S$ and $\lambda_1$ its eigenvalue.
The objective at the optimum is
$$ V_1=\mathbf b_1^\top S\mathbf b_1=\lambda_1\,\mathbf b_1^\top\mathbf b_1=\lambda_1. $$
**Variance captured = the eigenvalue.** So pick the eigenvector with the **largest** eigenvalue — the
**first principal component**. ($\sqrt{\lambda_1}$ is the *loading*, the std accounted for.)

**Going to $M$ dimensions.** Find the $m$th PC by removing the effect of the first $m-1$ from the data
$\hat X=X-\sum_{i<m}\mathbf b_i\mathbf b_i^\top X$, then maximizing $\mathbf b_m^\top\hat S\mathbf b_m$.
The optimal $\mathbf b_m$ is the top eigenvector of $\hat S$ — which turns out to be the eigenvector of
$S$ with the $m$th largest eigenvalue. Total variance captured:
$$ V_M=\sum_{m=1}^M\lambda_m,\qquad \text{variance lost}=J_M=\sum_{j=M+1}^D\lambda_j=V_D-V_M. $$

**ML usage:** this is the derivation that explains *why* PCA picks top eigenvectors — used whenever
PCA is applied for compression or visualization (see [[principal-component-analysis]]).

## 🔴 Theory (Expert)
- **Deflation shares eigenvectors.** With $B_{m-1}=\sum_{i<m}\mathbf b_i\mathbf b_i^\top$ the projector
  onto the first $m-1$ PCs, $\hat S=(S-SB_{m-1}-B_{m-1}S+B_{m-1}SB_{m-1})$. For an eigenvector $\mathbf b_i$ of $S$:
  $B_{m-1}\mathbf b_i=\mathbf b_i$ if $i<m$, else $0$. Hence $\hat S\mathbf b_i=\lambda_i\mathbf b_i$ for $i\ge m$
  and $\hat S\mathbf b_i=0$ for $i<m$. So **every eigenvector of $S$ is an eigenvector of $\hat S$**, with
  the already-extracted directions pushed to eigenvalue $0$ (they span $\ker\hat S$). This justifies the
  sequential greedy procedure being globally optimal.
- **Rayleigh–Ritz.** $\max_{\|\mathbf b\|=1}\mathbf b^\top S\mathbf b=\lambda_{\max}(S)$; the constrained
  max over a subspace orthogonal to the top $m-1$ eigenvectors gives the next eigenvalue — the Courant–Fischer
  characterization. Symmetry of $S$ (spectral theorem, Thm 4.15) guarantees a real orthonormal eigenbasis.
- **Equivalence.** This max-variance objective is identical to the [[pca-reconstruction-error|minimum-reconstruction-error]]
  objective (§10.3): minimizing variance in the discarded orthogonal complement = maximizing retained variance.
- Hotelling (1933) framed PCA precisely as variance maximization.

## Connections
- Builds on: [[data-covariance-matrix]], [[eigenvalues-and-eigenvectors]], [[lagrange-multipliers]], [[orthogonal-projection]]
- Leads to: [[principal-component-analysis]], [[pca-reconstruction-error]]
- Used by: [[principal-component-analysis]], [[dimensionality-reduction]]
- People: [[harold-hotelling]]

## Sources
- `raw/mml-book.pdf` §10.2 (Eqs. 10.7–10.25), printed pp. 320–324 (PDF pp. 326–330).
