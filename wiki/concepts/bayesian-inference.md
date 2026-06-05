---
title: Bayesian Inference
type: concept
tags: [machine-learning, probabilistic-modeling, bayesian]
difficulty: advanced
sources: [mml-ch08-when-models-meet-data]
created: 2026-06-04
updated: 2026-06-04
---

# Bayesian Inference

> **One-liner:** Instead of committing to one best setting of the model, keep *every* plausible setting weighted by how well it explains the data, and average over all of them when you predict.

## 🟢 Intuition (Dummy)
Maximum likelihood and MAP hand you a single answer: "the slope is 2.3." Bayesian inference refuses
to throw away the uncertainty. It says: "given what I've seen, the slope is *probably* around 2.3,
but 2.0 and 2.6 are also believable" — a whole *distribution* of answers. When it's time to predict,
it doesn't pick one slope; it asks every plausible slope for its prediction and averages them,
weighting each by how believable it is. The payoff: predictions that honestly report "I'm not sure,"
which matters a lot when a decision depends on them.

## 🟡 Mechanics (Practitioner)
Start from a **prior** $p(\boldsymbol\theta)$ (beliefs before data) and a **likelihood**
$p(\mathcal X\mid\boldsymbol\theta)$. Bayes' theorem produces the **posterior** (MML §8.4.2, Eq. 8.22):
$$ p(\boldsymbol\theta\mid\mathcal X)=\frac{p(\mathcal X\mid\boldsymbol\theta)\,p(\boldsymbol\theta)}{p(\mathcal X)},
\qquad p(\mathcal X)=\int p(\mathcal X\mid\boldsymbol\theta)\,p(\boldsymbol\theta)\,d\boldsymbol\theta . $$
The denominator $p(\mathcal X)$ is the **marginal likelihood / evidence**, found by integrating the
parameters out. Bayes' theorem **inverts** the parameter→data relationship into a data→parameter one.

The key difference from [[maximum-likelihood-estimation|MLE]] / [[maximum-a-posteriori-estimation|MAP]]:
those return a single point $\boldsymbol\theta^*$ and predict with $p(\mathbf x\mid\boldsymbol\theta^*)$.
Bayesian inference keeps the whole posterior and forms the **predictive distribution** by
marginalizing (Eq. 8.23):
$$ p(\mathbf x)=\int p(\mathbf x\mid\boldsymbol\theta)\,p(\boldsymbol\theta)\,d\boldsymbol\theta
= \mathbb E_{\boldsymbol\theta}\big[p(\mathbf x\mid\boldsymbol\theta)\big] . $$
The prediction is an **average over all plausible parameters**, weighted by their plausibility — so
parameter uncertainty propagates into prediction uncertainty.

| | Point estimate (MLE/MAP) | Bayesian inference |
|---|---|---|
| Output | single $\boldsymbol\theta^*$ | full posterior $p(\boldsymbol\theta\mid\mathcal X)$ |
| Core computation | **optimization** | **integration** |
| Prediction | plug in $\boldsymbol\theta^*$ | average / marginalize |
| Uncertainty | none | yes |

**Latent-variable models (§8.4.3).** Models may also carry **latent variables** $\mathbf z$ (not
parameters — they describe the data-generating process). With prior $p(\mathbf z)$, the likelihood
marginalizes them out (Eq. 8.25):
$$ p(\mathbf x\mid\boldsymbol\theta)=\int p(\mathbf x\mid\mathbf z,\boldsymbol\theta)\,p(\mathbf z)\,d\mathbf z . $$
This is the structure behind [[principal-component-analysis|PCA]] (Ch. 10) and
[[gaussian-mixture-model|Gaussian mixtures]] (Ch. 11), trained via [[expectation-maximization|EM]].

**ML usage.** Bayesian linear regression, Gaussian processes, Bayesian optimization (hyperparameter
search), topic models, data-efficient reinforcement learning, online ranking (TrueSkill).

## 🔴 Theory (Expert)
- A **probabilistic model is fully specified by the joint distribution** $p(\mathbf x,\boldsymbol\theta)$
  (§8.4.1): from it the prior, likelihood (product rule), evidence (sum rule), and posterior all
  follow. This is why the joint is the central object and why [[directed-graphical-models]] draw its
  factorization.
- **Tractability.** The integrals in (8.22)/(8.23) are closed-form only for **conjugate** priors
  ([[conjugate-prior]], §6.6.1) — e.g. a Gaussian prior with a Gaussian likelihood yields a Gaussian
  posterior. Otherwise approximate: **MCMC** (stochastic), or deterministic **Laplace approximation**,
  **variational inference**, **expectation propagation**.
- For latent-variable models, marginalizing *both* $\mathbf z$ and $\boldsymbol\theta$ at once is
  generally impossible; one settles for the parameter-conditioned latent posterior
  $p(\mathbf z\mid\mathcal X,\boldsymbol\theta)=\frac{p(\mathcal X\mid\mathbf z,\boldsymbol\theta)p(\mathbf z)}{p(\mathcal X\mid\boldsymbol\theta)}$ (Eq. 8.28) — exactly the E-step of EM.
- The Bayesian/frequentist split traces back to the definition of probability (§6.1.1): MLE is
  frequentist, Bayesian inference treats parameters as random variables.

## Connections
- Builds on: [[bayes-theorem]], [[probability-distribution]], [[maximum-a-posteriori-estimation]]
- Leads to: [[bayesian-model-selection]], [[directed-graphical-models]], [[conjugate-prior]]
- Used by: [[latent-variable-model]], [[principal-component-analysis]], [[gaussian-mixture-model]], [[expectation-maximization]]
- People: [[thomas-bayes]]

## Sources
- `raw/mml-book.pdf` §8.4 (Eqs. 8.19–8.28), printed pp. 272–277 (PDF pp. 278–283).
