---
title: Thomas Bayes (c. 1701–1761)
type: person
tags: [probability, bayesian-inference, people]
sources: [mml-ch06-probability-and-distributions, mml-ch11-density-estimation-gaussian-mixture-models, mml-ch09-linear-regression]
created: 2026-06-04
updated: 2026-06-04
---

# Thomas Bayes (c. 1701–1761)

> **Why they matter to ML:** His theorem turns a prior belief plus observed data into a posterior belief — the engine of probabilistic inference, and the reason GMM responsibilities are exact posterior probabilities.

## Life (brief)
Thomas Bayes was an English Presbyterian minister and mathematician. His central result on inverse
probability was published posthumously in 1763 ("An Essay towards solving a Problem in the Doctrine of
Chances"), communicated by Richard Price. Pierre-Simon Laplace later developed and popularized the idea
into the modern form used today.

## Key contributions
- **Bayes' theorem.** Relates the posterior to the likelihood and prior:
  $$ p(z\mid x) = \frac{p(x\mid z)\,p(z)}{p(x)} . $$
  In this chapter it converts the latent-variable prior $p(z_k=1)=\pi_k$ and component likelihood
  $p(x\mid z_k=1)=\mathcal N(x\mid\mu_k,\Sigma_k)$ into the [[responsibility]]
  $r_{nk}=p(z_k=1\mid x_n)$ — the posterior probability that component $k$ generated the point. Maps to →
  [[bayes-theorem]] (TODO page), used in → [[latent-variable-model]], [[gaussian-mixture-model]].

- **Bayes' theorem in linear regression (Ch. 9).** The same rule converts a Gaussian parameter prior
  $p(\boldsymbol\theta)$ and Gaussian likelihood into a Gaussian parameter posterior, powering
  [[maximum-a-posteriori-estimation]] (posterior mode), [[bayesian-linear-regression]] (full posterior),
  and the [[marginal-likelihood]] (the denominator $p(Y\mid X)$ used for model selection).

## Connections
- Concepts: [[bayes-theorem]], [[responsibility]], [[latent-variable-model]], [[maximum-a-posteriori-estimation]], [[bayesian-linear-regression]], [[marginal-likelihood]]
- Related people: [[arthur-dempster]] (EM builds on Bayesian posteriors in its E-step), [[carl-friedrich-gauss]] (Gaussian likelihoods enabling closed-form Bayesian regression)

## Sources
- `raw/mml-book.pdf` §6.3 (Bayes' theorem, sum & product rules), printed pp. 184–186 (PDF pp. 190–192).
- `raw/mml-book.pdf` §11.4.3 (Eq. 11.68, "According to Bayes' theorem…"), printed p. 365 (PDF p. 371).
- `raw/mml-book.pdf` §9.2.3 & §9.3 (Bayes' theorem applied to regression parameters), printed pp. 300–311 (PDF pp. 306–317).
