---
title: Gaussian Mixture Model (GMM)
type: concept
tags: [density-estimation, gaussian-mixture-model, maximum-likelihood, clustering, machine-learning]
difficulty: intermediate
sources: [mml-ch11-density-estimation-gaussian-mixture-models]
created: 2026-06-04
updated: 2026-06-04
---

# Gaussian Mixture Model (GMM)

> **One-liner:** A Gaussian mixture model fits data by stacking several bell curves on top of each other, and learns where each bell goes by repeatedly asking "which bell most likely produced each point?"

## 🟢 Intuition (Dummy)
A single bell curve (Gaussian) can only describe data that clumps around one center. Real data often has
**several** clumps. A GMM uses several bell curves at once — each with its own center, width, and
importance — and blends them. To fit it, you can't just solve one tidy equation like for a single bell.
Instead you guess, then refine: figure out how much each bell is "responsible" for each data point, nudge
each bell toward the points it owns, and repeat until things stop moving. That loop is the EM algorithm.

## 🟡 Mechanics (Practitioner)
A GMM is a [[mixture-model]] whose components are Gaussians:
$$ p(x \mid \theta) = \sum_{k=1}^{K} \pi_k\, \mathcal N\!\big(x \mid \mu_k, \Sigma_k\big),\qquad
0 \le \pi_k \le 1,\ \ \sum_{k=1}^{K}\pi_k = 1, $$
with parameters $\theta=\{\mu_k,\Sigma_k,\pi_k : k=1,\dots,K\}$. ($K=1$ recovers a single Gaussian.)

**Objective — maximum likelihood.** For i.i.d. data $X=\{x_1,\dots,x_N\}$ the log-likelihood is
$$ \mathcal L = \log p(X\mid\theta) = \sum_{n=1}^{N}\log \sum_{k=1}^{K}\pi_k\,\mathcal N\!\big(x_n\mid\mu_k,\Sigma_k\big). $$
The **log of a sum** cannot be simplified, so setting $\nabla_\theta\mathcal L=0$ gives no closed-form
solution (unlike linear regression or PCA). The stationarity conditions instead yield coupled
fixed-point equations in terms of the [[responsibility|responsibilities]]
$r_{nk}=\dfrac{\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\mathcal N(x_n\mid\mu_j,\Sigma_j)}$,
with $N_k:=\sum_{n=1}^N r_{nk}$ the total responsibility of component $k$:

| Parameter | Update | Reference |
|---|---|---|
| Means | $\displaystyle \mu_k^{\text{new}} = \frac{1}{N_k}\sum_{n=1}^N r_{nk}\,x_n$ | **Thm 11.1** |
| Covariances | $\displaystyle \Sigma_k^{\text{new}} = \frac{1}{N_k}\sum_{n=1}^N r_{nk}(x_n-\mu_k)(x_n-\mu_k)^\top$ | **Thm 11.2** |
| Weights | $\displaystyle \pi_k^{\text{new}} = \frac{N_k}{N}$ | **Thm 11.3** |

Each update is a **responsibility-weighted average**: the mean is pulled toward the points a component is
responsible for; the covariance is the weighted spread of those points; the weight is the fraction of
total responsibility. Because $r_{nk}$ itself depends on all parameters, you must **iterate** (see
[[expectation-maximization]]).

**Worked example (book's running example).** Data $X=\{-3,-2.5,-1,0,2,4,5\}$, $K=3$, equal initial
weights $\pi_k=\tfrac13$. One full update cycle moves $\mu_1:-4\to-2.7$, $\mu_3:8\to3.7$, shrinks
$\sigma_1^2:1\to0.14$, and raises the log-likelihood from $-28.3$ to $-14.4$. After 5 EM iterations it
converges to $p(x)=0.29\,\mathcal N(x\mid-2.75,0.06)+0.28\,\mathcal N(x\mid-0.5,0.25)+0.43\,\mathcal N(x\mid3.64,1.63)$.

**ML usage.** Soft clustering, density estimation, generative sampling (ancestral sampling: pick a
component by $\boldsymbol\pi$, then sample from it), speaker/image segmentation, anomaly detection.

## 🔴 Theory (Expert)
- **Proof sketch (Thm 11.1).** Using $\partial\log p(x_n\mid\theta)/\partial\theta = \frac{1}{p(x_n\mid\theta)}\partial p(x_n\mid\theta)/\partial\theta$ and
  $\partial\mathcal N(x_n\mid\mu_k,\Sigma_k)/\partial\mu_k = (x_n-\mu_k)^\top\Sigma_k^{-1}\mathcal N(\cdot)$,
  the gradient collapses to $\sum_n r_{nk}(x_n-\mu_k)^\top\Sigma_k^{-1}=0$, giving the weighted mean.
- **Proof sketch (Thm 11.2).** Apply matrix-calculus identities $\partial\det(\Sigma)^{-1/2}/\partial\Sigma = -\tfrac12\det(\Sigma)^{-1/2}\Sigma^{-1}$ and
  $\partial[(x-\mu)^\top\Sigma^{-1}(x-\mu)]/\partial\Sigma = -\Sigma^{-1}(x-\mu)(x-\mu)^\top\Sigma^{-1}$; setting the gradient to $0$ yields $N_k I=\big(\sum_n r_{nk}(x_n-\mu_k)(x_n-\mu_k)^\top\big)\Sigma_k^{-1}$.
- **Proof sketch (Thm 11.3).** The constraint $\sum_k\pi_k=1$ is handled with a [[lagrange-multipliers|Lagrange multiplier]] $\lambda$; stationarity gives $\pi_k=-N_k/\lambda$, and summing forces $\lambda=-N$, hence $\pi_k=N_k/N$.
- These are *necessary* (stationarity) conditions only; the responsibilities couple all parameters, so the
  system is solved iteratively and converges to a **local** optimum.
- **Degenerate/overfitting pathology:** if a component collapses onto a single data point ($\mu_k=x_n$,
  $\Sigma_k\to 0$), the likelihood diverges to $\infty$ — a fundamental flaw of ML for GMMs; priors
  (MAP / Bayesian inference, variational methods) regularize this.
- **Relation to K-means:** fix $\Sigma_k=I$ and harden the soft responsibilities to a single winner →
  K-means. GMM = "soft" probabilistic generalization (MacKay, 2003).

## Connections
- Builds on: [[mixture-model]], [[gaussian-distribution]], [[maximum-likelihood-estimation]], [[responsibility]]
- Leads to: [[expectation-maximization]], [[latent-variable-model]]
- Used by: clustering, density estimation, generative modeling
- People: [[arthur-dempster]] (EM), [[thomas-bayes]] (responsibilities as posteriors)

## Sources
- `raw/mml-book.pdf` §11.1–11.2 (Eqs. 11.3–11.10, Thms 11.1–11.3), printed pp. 349–360 (PDF pp. 355–366).
