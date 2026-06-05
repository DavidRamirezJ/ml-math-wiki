---
title: Principal Component Analysis (PCA)
type: concept
tags: [dimensionality-reduction, pca, unsupervised-learning, matrix-decomposition]
difficulty: intermediate
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
---

# Principal Component Analysis (PCA)

> **One-liner:** PCA finds the few directions in which your data spreads out the most and describes every data point using only its position along those directions.

## 🟢 Intuition (Dummy)
Imagine a swarm of points shaped like a flattened, tilted football (an ellipsoid). It's long in one
direction, medium in another, and almost flat in a third. PCA looks at the swarm and finds those
natural axes, longest first. To compress the data you keep only the longest few axes and record how
far along each one a point sits — you ignore the squashed directions because nobody moves much there
anyway.

The longest axis is the **first principal component**: the single direction that captures the most
"spread." The next axis (at right angles to the first) is the second, and so on. Keep $M$ of them and
you've turned $D$ numbers per point into $M$ numbers, with the smallest possible loss.

## 🟡 Mechanics (Practitioner)
**Setup.** Centered i.i.d. data $\mathbf x_n\in\mathbb R^D$ with [[data-covariance-matrix|data covariance matrix]]
$S=\frac1N\sum_n \mathbf x_n\mathbf x_n^\top=\frac1N XX^\top$ (here $X=[\mathbf x_1,\dots,\mathbf x_N]\in\mathbb R^{D\times N}$).
Encode with orthonormal $B=[\mathbf b_1,\dots,\mathbf b_M]$: $\mathbf z=B^\top\mathbf x$, decode $\tilde{\mathbf x}=BB^\top\mathbf x$.

**The objective (two equivalent forms):**
$$ \max_{B:\,B^\top B=I}\ \operatorname{tr}(B^\top S B) \qquad\Longleftrightarrow\qquad \min_{B:\,B^\top B=I}\ \tfrac1N\sum_{n}\|\mathbf x_n-BB^\top\mathbf x_n\|^2 . $$
Maximizing captured variance ([[pca-maximum-variance]]) equals minimizing reconstruction error ([[pca-reconstruction-error]]).

**The solution.** Both give: **the columns of $B$ are the $M$ eigenvectors of $S$ with the $M$ largest
eigenvalues** $\lambda_1\ge\dots\ge\lambda_M$. Then
$$ V_M=\sum_{m=1}^M\lambda_m\ \ (\text{captured variance}),\qquad J_M=\sum_{j=M+1}^D\lambda_j\ \ (\text{reconstruction error/lost variance}). $$
The relative captured variance is $V_M/V_D$.

**The practical recipe (§10.6):**
1. **Mean-subtract:** $\mathbf x_n\leftarrow\mathbf x_n-\boldsymbol\mu$.
2. **Standardize:** divide each dimension $d$ by its std $\sigma_d$ (unit-free, variance 1 per axis).
3. **Eigendecompose** $S$ (or take the [[singular-value-decomposition|SVD]] of $X$); collect the top-$M$ eigenvectors as columns of $B$.
4. **Project** a (standardized) point: $\mathbf z_*=B^\top\mathbf x_*$, reconstruct $\tilde{\mathbf x}_*=BB^\top\mathbf x_*$. PCA returns the **coordinates** $\mathbf z_*$.
5. **Undo standardization** to map back to data space: $\tilde x_*^{(d)}\leftarrow \tilde x_*^{(d)}\sigma_d+\mu_d$.

**Worked sketch (MNIST "8", §10.6 example):** 5,389 images, each $\mathbf x\in\mathbb R^{784}$. With
$M=1$ the reconstruction is a blurry generic 8; $M=10,100,500$ get progressively sharper; $\approx 550$
PCs reconstruct essentially perfectly. The reconstruction-error curve mirrors the eigenvalue scree
plot — a handful of components carry most of the variance.

**Computation choices:**
- **Eigendecomposition** of $S$ directly, or **SVD** of $X$ — eigenvalues relate to singular values by $\lambda_d=\sigma_d^2/N$, and the columns of $U$ in $X=U\Sigma V^\top$ are the eigenvectors of $S$.
- **Best rank-$M$ approximation** (Eckart–Young, Thm 4.25): $\tilde X_M=U_M\Sigma_M V_M^\top$ — truncate the SVD at the top $M$ singular values.
- **$N\ll D$ trick (§10.5):** eigendecompose the $N\times N$ matrix $\frac1N X^\top X$ (cheap), get its eigenvectors $\mathbf c_m=X^\top\mathbf b_m$, then recover $\mathbf b_m\propto X\mathbf c_m$ and renormalize.
- **Top vector only:** [[power-iteration]].

## 🔴 Theory (Expert)
- **Why eigenvectors.** The first PC solves $\max_{\|\mathbf b_1\|=1}\mathbf b_1^\top S\mathbf b_1$. The Lagrangian $\mathcal L=\mathbf b_1^\top S\mathbf b_1+\lambda_1(1-\mathbf b_1^\top\mathbf b_1)$ has stationarity $S\mathbf b_1=\lambda_1\mathbf b_1$ — an **eigenproblem** — and the objective at the optimum is $\mathbf b_1^\top S\mathbf b_1=\lambda_1$. So variance along $\mathbf b_m$ *equals* its eigenvalue. Subsequent PCs repeat on the deflated covariance $\hat S$, which shares $S$'s eigenvectors (eigenvalue $0$ on the already-extracted subspace).
- **Spectral theorem (Thm 4.15).** $S$ is symmetric positive semidefinite, so it has a real orthonormal eigenbasis — the principal axes are guaranteed orthogonal and real.
- **Best rank-$M$ approximation of the identity.** $J_M=\frac1N\sum_n\|(I-BB^\top)\mathbf x_n\|^2$; finding the orthonormal $B$ is equivalent to finding the best rank-$M$ approximation $BB^\top$ of $I$ (in the metric induced by $S$). $BB^\top$ is the orthogonal projector onto the principal subspace; $\sum_{j>M}\mathbf b_j\mathbf b_j^\top$ projects onto its orthogonal complement, where the displacement vectors $\mathbf x_n-\tilde{\mathbf x}_n$ live.
- **Karhunen–Loève transform** is PCA's name in signal processing.
- **Rotational ambiguity.** PCA identifies the optimal *subspace* but is invariant to rotations within it (and to the overall sign of each PC) — it cannot recover the underlying *sources*; ICA addresses this with non-Gaussian priors.
- **Probabilistic view.** [[probabilistic-pca]] recovers PCA as the noise-free ($\sigma\to0$) MLE of a Gaussian latent-variable model, and exposes PCA as a covariance-matrix decomposition $S\to BB^\top$.
- **Limits of direct computation.** Eigenvalues are roots of the characteristic polynomial, but by **Abel–Ruffini** there is no algebraic root formula for degree $\ge 5$, so $>4\times4$ matrices need iterative solvers (`np.linalg.eigh`, `np.linalg.svd`).

## Connections
- Builds on: [[data-covariance-matrix]], [[basis-and-rank]], [[linear-mapping]], [[singular-value-decomposition]], [[eigenvalues-and-eigenvectors]], [[orthogonal-projection]], [[lagrange-multipliers]]
- Leads to: [[probabilistic-pca]], [[auto-encoder]], kernel PCA, GP-LVM
- Used by: [[dimensionality-reduction]], data visualization, compression, denoising, [[power-iteration]] (PageRank)
- People: [[karl-pearson]], [[harold-hotelling]]

## Sources
- `raw/mml-book.pdf` §10.1–10.6, printed pp. 318–338 (PDF pp. 324–344). Key equations 10.1–10.62.
