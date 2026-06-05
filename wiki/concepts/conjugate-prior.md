---
title: Conjugate Prior
type: concept
tags: [probability, bayesian, inference]
difficulty: intermediate
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Conjugate Prior

> **One-liner:** A conjugate prior is a prior chosen so that the posterior comes out as the same kind of distribution — so updating beliefs is just tweaking numbers.

## 🟢 Intuition (Dummy)
[[bayes-theorem|Bayes' theorem]] tells you to multiply a prior belief by the evidence to get an updated belief, but that multiplication usually produces some ugly shape you can't write down. A **conjugate prior** is a clever matching choice: pick the prior from the right "family," and the messy multiplication magically lands you back in the *same* family — just with shifted parameters. It's like a currency that stays the same currency after the exchange: you only update the amount, never convert. Each new data point becomes a simple counter increment.

## 🟡 Mechanics (Practitioner)
**Definition (Def 6.13):** a prior is **conjugate** for a likelihood if the resulting posterior is of the same form/type as the prior.

**Beta–Binomial (Ex 6.11).** Likelihood $\;p(x\mid N,\mu)=\binom{N}{x}\mu^x(1-\mu)^{N-x}$, prior $\;\mu\sim\mathrm{Beta}(\alpha,\beta)\propto\mu^{\alpha-1}(1-\mu)^{\beta-1}$. Observing $h$ heads:
$$ p(\mu\mid h,N,\alpha,\beta)\propto \mu^{h}(1-\mu)^{N-h}\cdot\mu^{\alpha-1}(1-\mu)^{\beta-1}=\mu^{h+\alpha-1}(1-\mu)^{N-h+\beta-1}, $$
$$ \Rightarrow\quad p(\mu\mid\cdot)=\mathrm{Beta}(\alpha+h,\ \beta+N-h). $$
The update is *purely* "add successes to $\alpha$, add failures to $\beta$."

**Beta–Bernoulli (Ex 6.12).** Single observation $x\in\{0,1\}$: posterior is $\mathrm{Beta}(\alpha+x,\ \beta+1-x)$.

**Standard conjugate pairs (Table 6.2):**

| Likelihood | Conjugate prior | Posterior |
|---|---|---|
| Bernoulli | Beta | Beta |
| Binomial | Beta | Beta |
| Gaussian (mean) | Gaussian | Gaussian |
| Gaussian (variance, univariate) | inverse-Gamma | inverse-Gamma |
| Gaussian (covariance, multivariate) | inverse-Wishart | inverse-Wishart |
| Multinomial | Dirichlet | Dirichlet |

**ML usage:** keeps Bayesian inference closed-form and online — sequential/streaming updates, Bayesian linear regression with Gaussian prior on weights (Ch.9), topic models (Dirichlet–Multinomial). The Beta parameters $(\alpha,\beta)$ act like pseudo-counts encoding prior belief.

## 🔴 Theory (Expert)
- **Every** [[exponential-family]] member has a conjugate prior, also in the exponential family (Eq 6.120). Deriving the canonical conjugate of the Bernoulli's exponential-family form *yields the Beta distribution* (Ex 6.15) — conjugacy is structural, not coincidental.
- Conjugacy sidesteps the intractable evidence integral $p(y)$: because prior × likelihood stays in a known family, the normalizer is read off in closed form.
- Geometrically, conjugate priors "retain the same distance structure as the likelihood" (Agarwal & Daumé III, 2010) on the statistical manifold.
- Trade-off: conjugacy buys tractability but constrains the prior's shape; modern practice often forgoes it for flexible priors plus MCMC/variational inference.

## Connections
- Builds on: [[bayes-theorem]], [[exponential-family]], [[probability-distribution]]
- Leads to: Bayesian linear regression (Ch.9), Dirichlet–Multinomial models
- Used by: online/streaming Bayesian updating, [[gaussian-distribution]] posteriors
- People: [[thomas-bayes]]

## Sources
- `raw/mml-book.pdf` §6.6.1 (Def 6.13, Ex 6.11–6.12, Table 6.2), printed pp. 208–210 (PDF pp. 214–216).
