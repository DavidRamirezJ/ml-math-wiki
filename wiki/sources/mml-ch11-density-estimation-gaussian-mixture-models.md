---
title: "MML Ch. 11 — Density Estimation with Gaussian Mixture Models"
type: source
tags: [density-estimation, gaussian-mixture-model, em-algorithm, latent-variable-models, machine-learning]
sources: []
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch. 11 — Density Estimation with Gaussian Mixture Models

- **Raw file:** `raw/mml-book.pdf` Chapter 11, printed pp. 348–369 (PDF pp. 354–375).
- **Type:** book chapter
- **Ingested:** 2026-06-04

This is the third "pillar" application chapter of Part II (after regression Ch. 9 and dimensionality
reduction Ch. 10): **density estimation** — compactly representing a dataset by a probability density
from a parametric family. The headline method is the **Gaussian mixture model (GMM)**, fit by
**maximum likelihood** via the **expectation–maximization (EM) algorithm**.

## Section coverage

| § | Title | Wiki page(s) |
|---|---|---|
| 11.0 | Intro: density estimation, mixture models | [[mixture-model]] |
| 11.1 | Gaussian Mixture Model | [[gaussian-mixture-model]] |
| 11.2 | Parameter Learning via Maximum Likelihood | [[gaussian-mixture-model]] (likelihood), [[responsibility]] |
| 11.2.1 | Responsibilities | [[responsibility]] |
| 11.2.2 | Updating the Means | [[gaussian-mixture-model]] (Thm 11.1) |
| 11.2.3 | Updating the Covariances | [[gaussian-mixture-model]] (Thm 11.2) |
| 11.2.4 | Updating the Mixture Weights | [[gaussian-mixture-model]] (Thm 11.3) |
| 11.3 | EM Algorithm | [[expectation-maximization]] |
| 11.4 | Latent-Variable Perspective | [[latent-variable-model]] |
| 11.4.1 | Generative Process and Probabilistic Model | [[latent-variable-model]] |
| 11.4.2 | Likelihood (marginalization) | [[latent-variable-model]] |
| 11.4.3 | Posterior Distribution | [[responsibility]] (as posterior) |
| 11.4.4 | Extension to a Full Dataset | [[latent-variable-model]] |
| 11.4.5 | EM Algorithm Revisited | [[expectation-maximization]] |
| 11.5 | Further Reading: histograms, KDE | [[kernel-density-estimation]], [[histogram]] |

## TL;DR (5 bullets)

- A **mixture model** writes a density as a convex combination $p(x)=\sum_k \pi_k p_k(x)$ of simple base
  densities; with Gaussian components it is a **GMM**, which can model multimodal data a single Gaussian cannot.
- Fitting a GMM by **maximum likelihood** gives a log-likelihood with a log-of-a-sum that *cannot* be
  maximized in closed form — unlike linear regression or PCA.
- The **responsibility** $r_{nk}$ — the posterior probability that component $k$ generated point $x_n$ —
  is the central quantity; the optimal-parameter equations (Thms 11.1–11.3) are all responsibility-weighted averages.
- Because the updates and the responsibilities are mutually dependent, you **iterate**: the **EM algorithm**
  alternates an E-step (recompute responsibilities) and an M-step (re-estimate $\mu_k,\Sigma_k,\pi_k$),
  monotonically increasing the likelihood to a local optimum.
- The **latent-variable perspective** (one-hot indicator $z$ with prior $\pi$) justifies all of this:
  responsibilities are exact Bayesian posteriors $p(z_k=1\mid x)$, and EM falls out as maximizing the
  expected complete-data log-likelihood.

## Key concepts introduced

- [[mixture-model]] — convex combination of base distributions.
- [[gaussian-mixture-model]] — the mixture model with Gaussian components, plus its MLE update theorems.
- [[responsibility]] — soft assignment / posterior probability of component membership.
- [[expectation-maximization]] — the iterative E-step/M-step fitting algorithm.
- [[latent-variable-model]] — the generative one-hot indicator view of a GMM.
- [[kernel-density-estimation]] — nonparametric density estimation by placing a kernel on each point.
- [[histogram]] — the simplest nonparametric density estimator (binning + counting).

## Notable results / theorems

- **Thm 11.1 (mean update):** $\mu_k^{\text{new}} = \frac{1}{N_k}\sum_n r_{nk}\,x_n$, with $N_k=\sum_n r_{nk}$.
- **Thm 11.2 (covariance update):** $\Sigma_k^{\text{new}} = \frac{1}{N_k}\sum_n r_{nk}(x_n-\mu_k)(x_n-\mu_k)^\top$.
- **Thm 11.3 (weight update):** $\pi_k^{\text{new}} = N_k/N$ (derived via a [[lagrange-multipliers|Lagrange multiplier]] for $\sum_k\pi_k=1$).
- **Responsibility as posterior (Eq. 11.69/11.72):** $p(z_k=1\mid x) = \frac{\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\,\mathcal N(x\mid\mu_j,\Sigma_j)} = r_{nk}$ via [[bayes-theorem|Bayes' theorem]].
- **EM monotonic improvement:** every EM iteration increases the log-likelihood (Neal & Hinton, 1999); converges to a *local* optimum (Dempster et al., 1977).

## What this changes in the wiki

- First chapter ingested from Part II (ML methods); introduces the **probabilistic / generative-model**
  thread (priors, posteriors, marginalization, latent variables) that connects to Bayesian inference and
  to PCA's continuous latent-variable view (Ch. 10).
- Establishes **EM** as a reusable optimization tool for latent-variable models beyond GMMs.
- Connects to clustering: GMM is the "soft" probabilistic generalization of K-means (set $\Sigma_k=I$,
  ignore covariances → K-means hard assignment).
- No contradictions with existing linear-algebra pages; depends on Gaussian distribution, Bayes' theorem,
  MLE, and matrix calculus from Chs. 5–8 (linked as TODO stubs where pages don't yet exist).
