---
title: Latent-Variable Model (Discrete, for GMMs)
type: concept
tags: [latent-variable-models, generative-models, bayesian-inference, gaussian-mixture-model]
difficulty: advanced
sources: [mml-ch11-density-estimation-gaussian-mixture-models]
created: 2026-06-04
updated: 2026-06-04
---

# Latent-Variable Model (Discrete, for GMMs)

> **One-liner:** A latent-variable model imagines an unseen "which-group" switch behind each data point, and recovers the data's distribution by averaging over all the ways that switch could have been set.

## 🟢 Intuition (Dummy)
Picture a machine that makes a data point in two hidden steps: first it secretly flips a weighted die to
pick one of $K$ factories, then that factory stamps out a point with its own characteristic spread. You
only ever see the finished point — never which factory made it. The factory choice is the **latent
variable**: hidden, but real. The whole point of the model is that, even though the choice is hidden, you
can reason about it: given a finished point, you can ask "which factory probably made this?" — and that
answer is exactly a cluster responsibility.

## 🟡 Mechanics (Practitioner)
For a [[gaussian-mixture-model]] with $K$ components, introduce a **one-hot** indicator
$z=[z_1,\dots,z_K]^\top$ with exactly one entry $=1$ ($\sum_k z_k=1$, a "1-of-$K$" / multinoulli encoding):

- **Prior** on the latent: $p(z)=\boldsymbol\pi=[\pi_1,\dots,\pi_K]^\top$, i.e. $p(z_k=1)=\pi_k$.
- **Conditional** (likelihood given the chosen component): $p(x\mid z_k=1)=\mathcal N(x\mid\mu_k,\Sigma_k)$.
- **Joint**: $p(x,z_k=1)=\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k)$.

**Generative process (ancestral sampling).** (1) sample $z\sim p(z)$ to pick a component; (2) sample
$x\sim p(x\mid z)$ from it; discard $z$ to get a sample from the GMM.

**Marginal likelihood** — sum the latent out over its $K$ one-hot settings:
$$ p(x\mid\theta) = \sum_{z} p(x\mid z,\theta)\,p(z\mid\theta) = \sum_{k=1}^{K}\pi_k\,\mathcal N\!\big(x\mid\mu_k,\Sigma_k\big), $$
which is **exactly** the GMM density — so the latent-variable model and the mixture model are the same thing.

**Posterior** (per point $n$, via [[bayes-theorem|Bayes' theorem]]) is the [[responsibility]]:
$$ p(z_{nk}=1\mid x_n) = \frac{\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\mathcal N(x_n\mid\mu_j,\Sigma_j)} = r_{nk}. $$

**ML usage.** This view (i) justifies the responsibility formulas as genuine posteriors, and (ii) makes the
[[expectation-maximization|EM algorithm]] derivable in a principled way for GMMs and general
latent-variable models.

## 🔴 Theory (Expert)
- **Full dataset.** Each point $x_n$ gets its own latent $z_n$, sharing the prior $\boldsymbol\pi$; the
  conditional factorizes, $p(x_1,\dots,x_N\mid z_1,\dots,z_N)=\prod_n p(x_n\mid z_n)$ (plate notation,
  Fig. 11.12).
- **Discrete vs continuous latents.** A GMM uses a *discrete* latent (which of $K$ components); contrast
  PCA / probabilistic PCA (Ch. 10), whose latents are *continuous* in $\mathbb R^M$. EM applies to both.
- **Why this matters for inference.** Defining a model by the joint $p(x,z)$ lets us (a) marginalize for
  the likelihood, (b) condition for the posterior, and (c) bound the log-marginal with a variational
  lower bound — the scaffolding behind EM and variational inference.
- **Limitations of ML here.** Only a point estimate of $\theta$ is obtained (no parameter uncertainty), and
  there is no conjugate prior for the GMM, so a Bayesian treatment requires approximations (variational
  inference). The number of components $K$ is itself a model-selection problem (cross-validation, or model
  evidence in a Bayesian setting).

## Connections
- Builds on: [[bayes-theorem]], [[gaussian-distribution]], [[mixture-model]], marginalization
- Leads to: [[expectation-maximization]], variational inference, generative models
- Used by: [[gaussian-mixture-model]], [[responsibility]]
- People: [[thomas-bayes]]

## Sources
- `raw/mml-book.pdf` §11.4 (Eqs. 11.58–11.72), printed pp. 363–367 (PDF pp. 369–373).
