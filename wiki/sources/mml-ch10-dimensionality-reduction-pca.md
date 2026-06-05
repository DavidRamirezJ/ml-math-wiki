---
title: "MML Ch.10 — Dimensionality Reduction with Principal Component Analysis"
type: source
tags: [dimensionality-reduction, pca, unsupervised-learning, linear-algebra]
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch.10 — Dimensionality Reduction with Principal Component Analysis

- **Raw file:** `raw/mml-book.pdf` §10, printed pp. 317–347 (PDF pp. 323–353)
- **Type:** book chapter
- **Ingested:** 2026-06-04

This is a **Part II** chapter: it assembles the linear-algebra and probability machinery of
Part I into a concrete ML method — **principal component analysis (PCA)**, the workhorse linear
dimensionality-reduction algorithm. PCA is derived three ways (max-variance, min-reconstruction,
latent-variable) that all land on the **same** eigenvector solution.

## TL;DR (5 bullets)
- **The problem:** compress data $\mathbf x_n\in\mathbb R^D$ to a code $\mathbf z_n\in\mathbb R^M$ ($M<D$) with an orthonormal projection $\mathbf z=B^\top\mathbf x$, then reconstruct $\tilde{\mathbf x}=BB^\top\mathbf x$, losing as little as possible.
- **Two equivalent objectives:** **maximize the variance** captured in the code, or **minimize the average squared reconstruction error** $\frac1N\sum\|\mathbf x_n-\tilde{\mathbf x}_n\|^2$. Both yield identical answers.
- **The solution:** the principal subspace is spanned by the $M$ eigenvectors of the **data covariance matrix** $S=\frac1N XX^\top$ associated with its $M$ **largest eigenvalues**. The captured variance is $\sum_{m=1}^M\lambda_m$; the lost variance (= reconstruction error) is $\sum_{j=M+1}^D\lambda_j$.
- **Computation:** eigendecomposition of $S$, or the **SVD** of $X$ (eigenvalues $\lambda_d=\sigma_d^2/N$); the **Eckart–Young** theorem makes truncated SVD the best rank-$M$ approximation. For $N\ll D$, work with the $N\times N$ matrix $X^\top X$ instead. For just the top vector, **power iteration**.
- **Probabilistic view:** **probabilistic PCA (PPCA)** is a Gaussian latent-variable model $\mathbf x=B\mathbf z+\boldsymbol\mu+\boldsymbol\epsilon$; classical PCA is its maximum-likelihood solution in the noise-free limit $\sigma\to 0$. Generalizes to factor analysis, ICA, kernel PCA, and deep auto-encoders.

## Section coverage

| § | Title | Wiki page(s) |
|---|---|---|
| 10 intro | Motivation (compression, overcomplete/correlated data) | [[dimensionality-reduction]] |
| 10.1 | Problem Setting (codes, projection matrix, encoder/decoder) | [[principal-component-analysis]], [[data-covariance-matrix]] |
| 10.2 | Maximum Variance Perspective | [[pca-maximum-variance]], [[principal-component-analysis]] |
| 10.2.1 | Direction with maximal variance (first PC) | [[pca-maximum-variance]] |
| 10.2.2 | $M$-dimensional subspace with maximal variance | [[pca-maximum-variance]] |
| 10.3 | Projection Perspective | [[pca-reconstruction-error]] |
| 10.3.1 | Setting and objective (reconstruction error) | [[pca-reconstruction-error]] |
| 10.3.2 | Finding optimal coordinates (orthogonal projection) | [[pca-reconstruction-error]] |
| 10.3.3 | Finding the basis of the principal subspace | [[pca-reconstruction-error]], [[principal-component-analysis]] |
| 10.4 | Eigenvector Computation and Low-Rank Approximations | [[principal-component-analysis]] |
| 10.4.1 | PCA using low-rank matrix approximations (SVD, Eckart–Young) | [[principal-component-analysis]] |
| 10.4.2 | Practical aspects (Abel–Ruffini, power iteration) | [[power-iteration]] |
| 10.5 | PCA in High Dimensions ($N\ll D$ trick) | [[principal-component-analysis]] |
| 10.6 | Key Steps of PCA in Practice (standardize → eig → project → undo) | [[principal-component-analysis]] |
| 10.7 | Latent Variable Perspective | [[probabilistic-pca]] |
| 10.7.1 | Generative process and probabilistic model | [[probabilistic-pca]] |
| 10.7.2 | Likelihood and joint distribution | [[probabilistic-pca]] |
| 10.7.3 | Posterior distribution | [[probabilistic-pca]] |
| 10.8 | Further Reading (auto-encoder, FA, ICA, kernel PCA, GP-LVM) | [[principal-component-analysis]], [[probabilistic-pca]] |

## Key concepts introduced
- [[dimensionality-reduction]] — exploiting low-dimensional structure in high-dimensional data.
- [[principal-component-analysis]] — the full PCA method: objective, eigenvector solution, and practical recipe.
- [[data-covariance-matrix]] — $S=\frac1N XX^\top$, whose spectrum drives PCA.
- [[pca-maximum-variance]] — deriving PCA by maximizing projected variance (Lagrangian → eigenproblem).
- [[pca-reconstruction-error]] — deriving PCA by minimizing average squared reconstruction error.
- [[probabilistic-pca]] — the Gaussian latent-variable model with PCA as its noise-free MLE.
- [[power-iteration]] — iterative method for the dominant eigenvector.

## Notable results / theorems
- **Variance = eigenvalue (Eq. 10.15, 10.23):** $V_m=\mathbf b_m^\top S\mathbf b_m=\lambda_m$ for a unit eigenvector. The first PC is the top eigenvector of $S$.
- **Reconstruction error (Eq. 10.44):** $J_M=\sum_{j=M+1}^D\lambda_j$ — the sum of the discarded eigenvalues.
- **Variance–reconstruction equivalence (§10.3.3):** minimizing $\frac1N\sum\|\mathbf x_n-\tilde{\mathbf x}_n\|^2$ = maximizing retained variance; same eigenvector solution.
- **SVD link (Eq. 10.48–10.49):** $S=\frac1N U\Sigma\Sigma^\top U^\top$, so columns of $U$ are eigenvectors of $S$ and $\lambda_d=\sigma_d^2/N$.
- **Eckart–Young (Thm 4.25, applied here):** truncated SVD gives the best rank-$M$ approximation $\tilde X_M=U_M\Sigma_M V_M^\top$ of $X$.
- **PPCA marginal (Eq. 10.70b):** $p(\mathbf x)=\mathcal N(\boldsymbol\mu,\,BB^\top+\sigma^2 I)$; **posterior** (Eq. 10.73–10.75) by Gaussian conditioning.
- **PPCA MLE (Eq. 10.77–10.81):** $\boldsymbol\mu_{ML}$ = sample mean, $\sigma^2_{ML}=\frac{1}{D-M}\sum_{j>M}\lambda_j$, and $B_{ML}=T(\Lambda-\sigma^2 I)^{1/2}R$ — unique up to an orthogonal $R$; classical PCA at $\sigma\to 0$.

## What this changes in the wiki
- First **Part II / ML-method** chapter ingested: introduces unsupervised learning, the encoder/decoder framing, and a latent-variable model.
- Heavily consumes Part-I concepts not yet written (eigenvalues, SVD, orthogonal projection, spectral theorem, Gaussian distribution, Lagrange multipliers, trace, covariance). These are linked as TODO markers and belong to Chapters 3, 4, 6, 7 — **not** created here.
- Connects to [[linear-mapping]] (encoder/decoder $B^\top, B$) and [[basis-and-rank]] (orthonormal basis of the principal subspace; rank-$M$ approximation). No contradictions with existing pages.
