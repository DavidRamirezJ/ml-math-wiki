---
title: Data Covariance Matrix
type: concept
tags: [statistics, linear-algebra, pca]
difficulty: intermediate
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
---

# Data Covariance Matrix

> **One-liner:** A square table that records, for a dataset, how much each pair of features spreads and moves together — the object whose biggest directions PCA hunts for.

## 🟢 Intuition (Dummy)
Suppose every data point has height and weight. The data covariance matrix answers two kinds of
questions at once: "how much do heights vary?" (and weights), and "when height goes up, does weight
tend to go up too?" The diagonal entries are the individual spreads (variances); the off-diagonal
entries are the "move-together" amounts (covariances). Lay a cloud of points down and the covariance
matrix is essentially the shape of the ellipse hugging that cloud — fat where the data spreads, thin
where it doesn't, tilted if features are correlated.

## 🟡 Mechanics (Practitioner)
For a **centered** dataset $\mathbf x_1,\dots,\mathbf x_N\in\mathbb R^D$ (mean $\mathbf 0$),
arranged as columns of $X=[\mathbf x_1,\dots,\mathbf x_N]\in\mathbb R^{D\times N}$, the data
covariance matrix is
$$ S=\frac1N\sum_{n=1}^N \mathbf x_n\mathbf x_n^\top \;=\; \frac1N XX^\top\ \in\mathbb R^{D\times D}. $$
> ⚠️ Note the book's convention: data points are **columns** of $X$ (so $X$ is $D\times N$, the
> transpose of the usual $N\times D$ data matrix). This keeps $S=\frac1N XX^\top$ clean.

**Key properties:**
- **Symmetric** and **positive semidefinite** — real, non-negative eigenvalues; an orthonormal eigenbasis (spectral theorem).
- Diagonal $S_{dd}$ = variance of feature $d$; off-diagonal $S_{ij}$ = covariance of features $i,j$.
- If the data is not centered, subtract the mean first; variance is mean-invariant: $\mathbb V[B^\top(\mathbf x-\boldsymbol\mu)]=\mathbb V[B^\top\mathbf x]$, so centering is "without loss of generality."

**Worked example.** For $\mathbf x_1=[2,0]^\top,\mathbf x_2=[-2,0]^\top$ (already centered),
$S=\frac12\big(\mathbf x_1\mathbf x_1^\top+\mathbf x_2\mathbf x_2^\top\big)=\begin{bmatrix}4&0\\0&0\end{bmatrix}$ —
all the variance is along $x_1$, none along $x_2$, so PCA would keep $\mathbf b_1=[1,0]^\top$ with $\lambda_1=4$.

**ML usage:** the central object of [[principal-component-analysis]] — its top eigenvectors are the
principal components, and their eigenvalues are the variances captured. Also appears in whitening,
Gaussian models, and Mahalanobis distance.

## 🔴 Theory (Expert)
- The variance of the data projected onto a unit direction $\mathbf b$ is the **Rayleigh quotient**
  $\mathbf b^\top S\mathbf b$ (with $\|\mathbf b\|=1$). Maximizing it over $\mathbf b$ gives the top
  eigenvector/eigenvalue of $S$ — this is exactly the PCA first-component derivation.
- **SVD link.** With $X=U\Sigma V^\top$, $\;S=\frac1N XX^\top=\frac1N U\Sigma\Sigma^\top U^\top$. The
  columns of $U$ are the eigenvectors of $S$ and the eigenvalues are $\lambda_d=\sigma_d^2/N$, tying
  the spectrum of $S$ to the singular values of $X$.
- **Rank.** If $N<D$ (and no duplicate points), $\operatorname{rank}(S)=N$, so $S$ has $D-N$ zero
  eigenvalues — the basis of the $N\ll D$ shortcut (work with the $N\times N$ matrix $\frac1N X^\top X$,
  which shares all nonzero eigenvalues).
- This is the **empirical/sample** covariance (a maximum-likelihood estimate using $1/N$, not the
  unbiased $1/(N-1)$). In PPCA the model covariance is $BB^\top+\sigma^2 I$, which equals $S$ as $\sigma\to0$.

## Connections
- Builds on: [[matrix]], [[vector]], covariance/variance (Ch. 6)
- Leads to: [[principal-component-analysis]], [[pca-maximum-variance]], [[singular-value-decomposition]]
- Used by: [[principal-component-analysis]], [[probabilistic-pca]], whitening, Gaussian models
- People: —

## Sources
- `raw/mml-book.pdf` §10.1 (Eq. 10.1), §10.2 remark on centering, §10.4–10.5, printed pp. 318–336 (PDF pp. 324–342).
