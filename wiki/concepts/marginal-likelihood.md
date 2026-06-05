---
title: Marginal Likelihood (Evidence)
type: concept
tags: [machine-learning, bayesian, model-selection, probabilistic-modeling]
difficulty: advanced
sources: [mml-ch09-linear-regression]
created: 2026-06-04
updated: 2026-06-04
---

# Marginal Likelihood (Evidence)

> **One-liner:** The probability that a whole model — averaged over every parameter setting it could take — would have produced your data; it's how Bayesians compare and rank models.

## 🟢 Intuition (Dummy)
The likelihood asks "how well does *this specific* set of parameters explain the data?" The marginal
likelihood asks a bigger question: "how well does this *model* explain the data, considering *all* the
parameter settings it allows, weighted by how plausible each was beforehand?"

Why average instead of cherry-picking the best parameters? Because a wildly flexible model can always
find *some* parameter setting that fits — but most of its settings fit terribly, so on average it's
not impressive. A simpler model that fits decently across most of its settings scores higher. This
built-in penalty for needless complexity is "Bayesian Occam's razor," and it's why the marginal
likelihood is the natural tool for **model selection**.

## 🟡 Mechanics (Practitioner)
The **marginal likelihood** (a.k.a. **evidence**) integrates the likelihood over the prior:
$$ p(Y\mid X) = \int p(Y\mid X,\boldsymbol\theta)\,p(\boldsymbol\theta)\,d\boldsymbol\theta = \mathbb E_{\boldsymbol\theta}\big[p(Y\mid X,\boldsymbol\theta)\big]. $$
It is the normalizing denominator in Bayes' theorem (it makes the posterior integrate to 1) and is
independent of $\boldsymbol\theta$.

**Closed form for [[bayesian-linear-regression]]** with prior $p(\boldsymbol\theta)=\mathcal N(\mathbf m_0,\mathbf S_0)$
and Gaussian likelihood $p(\mathbf y\mid\mathbf X,\boldsymbol\theta)=\mathcal N(\mathbf y\mid\mathbf X\boldsymbol\theta,\sigma^2\mathbf I)$:
$$ \boxed{\;p(Y\mid X) = \mathcal N\!\big(\mathbf y \;\big|\; \mathbf X\mathbf m_0,\;\; \mathbf X\mathbf S_0\mathbf X^\top + \sigma^2\mathbf I\big)\;} $$
- **Mean** $\mathbf X\mathbf m_0$: the data you'd expect under the prior mean.
- **Covariance** $\mathbf X\mathbf S_0\mathbf X^\top+\sigma^2\mathbf I$: prior parameter uncertainty propagated
  through the model, plus the noise floor.

**ML usage.** Compare two candidate models (e.g. polynomial degrees, kernels, feature sets) by their
marginal likelihoods; the larger one is preferred — no held-out validation set required. Maximizing the
log marginal likelihood w.r.t. hyperparameters (noise $\sigma^2$, prior scale) is **empirical Bayes /
type-II maximum likelihood**, the standard way to tune Gaussian-process and Bayesian-regression models.

## 🔴 Theory (Expert)
- **Why Gaussian.** $\mathbf y=\mathbf X\boldsymbol\theta+\boldsymbol\epsilon$ is an affine transform of the
  Gaussian $\boldsymbol\theta$ plus independent Gaussian noise $\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\sigma^2\mathbf I)$.
  An affine transform of a Gaussian is Gaussian, so $\mathbf y$ is Gaussian; mean and covariance follow
  from $\mathbb E[\mathbf y]=\mathbf X\mathbb E[\boldsymbol\theta]=\mathbf X\mathbf m_0$ and
  $\mathrm{Cov}[\mathbf y]=\mathbf X\,\mathrm{Cov}[\boldsymbol\theta]\,\mathbf X^\top+\sigma^2\mathbf I=\mathbf X\mathbf S_0\mathbf X^\top+\sigma^2\mathbf I$.
- **Relation to the posterior predictive.** The marginal likelihood and the posterior predictive
  distribution share the same functional form; the differences are (i) the marginal likelihood scores
  the *training* targets $\mathbf y$ (not test targets $y_*$), and (ii) it averages over the *prior*, while
  the posterior predictive averages over the *posterior*.
- **Bayesian Occam's razor.** Because $p(Y\mid X)$ is a normalized distribution over datasets, a model
  that spreads its probability mass over many possible datasets must assign less to any particular one.
  Complex models hedge across more datasets, so unless the extra complexity is genuinely needed, the
  simpler model wins automatically — the evidence penalizes overcomplexity without an explicit penalty
  term.
- **Intractability in general.** The integral is closed-form only for conjugate (here Gaussian–Gaussian)
  models. For neural networks and non-conjugate likelihoods it must be approximated (Laplace,
  variational inference, sampling).

## Connections
- Builds on: [[bayes-theorem]], [[gaussian-distribution]], [[bayesian-linear-regression]]
- Leads to: model selection, empirical Bayes, [[gaussian-process]]
- Used by: Bayesian model comparison, hyperparameter tuning
- People: [[thomas-bayes]]

## Sources
- `raw/mml-book.pdf` §9.3.5 (Eqs. 9.60–9.64), printed pp. 312–313 (PDF pp. 318–319).
