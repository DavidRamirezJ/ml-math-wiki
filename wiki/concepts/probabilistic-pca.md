---
title: Probabilistic PCA (PPCA)
type: concept
tags: [pca, latent-variable-model, gaussian, generative-model, dimensionality-reduction]
difficulty: advanced
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
---

# Probabilistic PCA (PPCA)

> **One-liner:** PCA rebuilt as a probability story — each data point is born from a few hidden random numbers, stretched by a matrix and sprinkled with noise — so we can sample, score, and reason about uncertainty.

## 🟢 Intuition (Dummy)
Plain PCA gives you compressed codes but no sense of "how sure am I?" or "what would a *new* fake data
point look like?" PPCA tells a generative story: every observation starts as a tiny random seed
$\mathbf z$ (a few numbers drawn from a bell curve), gets stretched and rotated by a matrix $B$, shifted
to the right place by $\boldsymbol\mu$, and finally gets a dusting of random noise. Run the story
forward and you can *generate* new data (e.g. new handwritten 8s). Run it backward (Bayes' rule) and
you can ask: given this observation, what hidden seed most likely produced it — and how confident am I?

## 🟡 Mechanics (Practitioner)
**Generative model** (Tipping & Bishop, 1999) with latent $\mathbf z\in\mathbb R^M$:
$$ p(\mathbf z)=\mathcal N(\mathbf 0,\,I),\qquad \mathbf x=B\mathbf z+\boldsymbol\mu+\boldsymbol\epsilon,\quad \boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\,\sigma^2 I), $$
so $p(\mathbf x\mid\mathbf z,B,\boldsymbol\mu,\sigma^2)=\mathcal N(\mathbf x\mid B\mathbf z+\boldsymbol\mu,\,\sigma^2 I)$.
Here $B\in\mathbb R^{D\times M}$, $\boldsymbol\mu\in\mathbb R^D$.

**Sampling (ancestral):** draw $\mathbf z_n\sim p(\mathbf z)$, then $\mathbf x_n\sim p(\mathbf x\mid\mathbf z_n)$.

**Marginal likelihood** (integrate out $\mathbf z$):
$$ p(\mathbf x\mid B,\boldsymbol\mu,\sigma^2)=\int p(\mathbf x\mid\mathbf z)\,p(\mathbf z)\,d\mathbf z=\mathcal N\big(\mathbf x\mid\boldsymbol\mu,\ BB^\top+\sigma^2 I\big). $$
(Mean $\mathbb E[\mathbf x]=\boldsymbol\mu$; covariance $\mathbb V[\mathbf x]=BB^\top+\sigma^2 I$.) This — not the conditional — is what you maximize for ML/MAP estimation.

**Joint and posterior.** $\mathbf x,\mathbf z$ are jointly Gaussian with cross-covariance $\operatorname{Cov}[\mathbf x,\mathbf z]=B$:
$$ p(\mathbf x,\mathbf z)=\mathcal N\!\left(\begin{bmatrix}\boldsymbol\mu\\\mathbf 0\end{bmatrix},\ \begin{bmatrix}BB^\top+\sigma^2 I & B\\ B^\top & I\end{bmatrix}\right). $$
Gaussian conditioning gives the **posterior over the code**:
$$ p(\mathbf z\mid\mathbf x)=\mathcal N(\mathbf m,\,C),\quad \mathbf m=B^\top(BB^\top+\sigma^2 I)^{-1}(\mathbf x-\boldsymbol\mu),\quad C=I-B^\top(BB^\top+\sigma^2 I)^{-1}B. $$
$C$ is data-independent; a small $\det C$ means a confident embedding, a large one flags a possible outlier.

**ML usage:** generative modeling, handling missing data via Bayes, novelty/outlier detection, mixtures
of PCA, EM for streaming data, and as the gateway to factor analysis and Bayesian PCA. See [[principal-component-analysis]].

## 🔴 Theory (Expert)
- **Maximum likelihood solution** (Tipping & Bishop, 1999):
  $$ \boldsymbol\mu_{ML}=\tfrac1N\sum_n\mathbf x_n,\qquad \sigma^2_{ML}=\tfrac1{D-M}\sum_{j=M+1}^D\lambda_j,\qquad B_{ML}=T(\Lambda-\sigma^2 I)^{1/2}R, $$
  where $T$ holds the top $M$ eigenvectors of $S$, $\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_M)$,
  and $R$ is *any* orthogonal matrix. $\sigma^2_{ML}$ = average leftover variance in the discarded directions
  (the "noise"). $\Lambda-\sigma^2 I\succeq0$ since $\sigma^2\le\lambda_M$.
- **PCA as a special case.** As $\sigma\to0$, $BB^\top+\sigma^2I\to BB^\top$ must equal $S=T\Lambda T^{-1}$,
  giving $B=T\Lambda^{1/2}R$ — exactly classical PCA. So (P)PCA *decomposes the data covariance matrix*.
- **Rotational non-identifiability.** $B_{ML}$ is unique only up to the orthogonal $R$; (10.78) is effectively
  an SVD. PCA finds the subspace, not the axes — the same ambiguity ICA resolves with non-Gaussian priors.
- **All stationary points of the PPCA likelihood are global optima** (a notable property lost by factor analysis).
- **Generalizations.** Per-dimension noise $\sigma_d^2$ → **factor analysis** (no closed form; needs EM; rotation-sensitive).
  Non-Gaussian $p(\mathbf z)$ → **ICA** (blind-source separation). Prior on $(B,\boldsymbol\mu,\sigma^2)$ → **Bayesian PCA**
  (Bishop, 1999), with automatic $M$ selection via MCMC/variational inference. Kernelize inner products → **kernel PCA**;
  GP latent map → **GP-LVM**.

## Connections
- Builds on: [[principal-component-analysis]], [[gaussian-distribution]], [[bayes-theorem]], [[linear-mapping]], [[data-covariance-matrix]]
- Leads to: factor analysis, ICA, Bayesian PCA, kernel PCA, GP-LVM, [[auto-encoder]]
- Used by: generative modeling, missing-data imputation, outlier detection
- People: [[karl-pearson]], [[harold-hotelling]]

## Sources
- `raw/mml-book.pdf` §10.7–10.8 (Eqs. 10.63–10.81), printed pp. 339–346 (PDF pp. 345–352).
