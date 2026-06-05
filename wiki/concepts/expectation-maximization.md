---
title: Expectation–Maximization (EM) Algorithm
type: concept
tags: [em-algorithm, latent-variable-models, maximum-likelihood, optimization, machine-learning]
difficulty: advanced
sources: [mml-ch11-density-estimation-gaussian-mixture-models]
created: 2026-06-04
updated: 2026-06-04
---

# Expectation–Maximization (EM) Algorithm

> **One-liner:** EM fits models with hidden groupings by guessing the hidden labels, updating the model to match, and repeating until it stops improving.

## 🟢 Intuition (Dummy)
You want to fit several bell curves to clustered data, but it's a chicken-and-egg problem: if you knew
which points belonged to which cluster, fitting each curve would be easy — and if you knew the curves,
deciding cluster membership would be easy. You know neither. EM breaks the deadlock by alternating: (1)
with the current curves, guess how much each cluster is responsible for each point; (2) with those
guesses fixed, refit each curve to the points it owns. Each pass makes the fit better (never worse), and
after a few rounds it settles. That alternation is Expectation–Maximization.

## 🟡 Mechanics (Practitioner)
EM is a general iterative scheme for maximum-likelihood (or MAP) estimation in
[[latent-variable-model|latent-variable models]] (Dempster, Laird & Rubin, 1977). For a
[[gaussian-mixture-model]]:

**Concrete EM for a GMM**
1. **Initialize** $\mu_k,\Sigma_k,\pi_k$.
2. **E-step** — evaluate [[responsibility|responsibilities]] for every point with current parameters:
   $$ r_{nk} = \frac{\pi_k\,\mathcal N(x_n\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\,\mathcal N(x_n\mid\mu_j,\Sigma_j)} . $$
3. **M-step** — re-estimate parameters from the responsibilities ($N_k=\sum_n r_{nk}$):
   $$ \mu_k = \tfrac{1}{N_k}\textstyle\sum_n r_{nk}x_n,\quad
   \Sigma_k = \tfrac{1}{N_k}\textstyle\sum_n r_{nk}(x_n-\mu_k)(x_n-\mu_k)^\top,\quad
   \pi_k = \tfrac{N_k}{N}. $$
   (Update the means first, then reuse them in the covariance update.)
4. **Repeat** E and M until the log-likelihood (or the parameters) stops changing.

**Worked example (book).** On the 7-point running example, EM converges in **5 iterations**; the negative
log-likelihood drops monotonically. On a 2-D dataset with $K=3$ it converges in ~62 iterations.

**ML usage.** GMMs and other mixture models, hidden Markov models, factor analysis, nonlinear
state-space models, missing-data problems, and (via the latent-variable view) many EM-flavored algorithms
in reinforcement learning.

## 🔴 Theory (Expert)
- **Principled derivation.** Given current parameters $\theta^{(t)}$, the **E-step** forms the expected
  complete-data log-likelihood (the "$Q$-function"):
  $$ Q\big(\theta\mid\theta^{(t)}\big) = \mathbb E_{z\mid x,\theta^{(t)}}\!\big[\log p(x,z\mid\theta)\big] = \int \log p(x,z\mid\theta)\,p\big(z\mid x,\theta^{(t)}\big)\,dz, $$
  where the expectation is over the **posterior** $p(z\mid x,\theta^{(t)})$ of the latents (for a GMM that
  posterior is exactly the responsibilities). The **M-step** sets
  $\theta^{(t+1)}=\arg\max_\theta Q(\theta\mid\theta^{(t)})$.
- **Monotonic ascent.** Every EM iteration is guaranteed not to decrease the marginal log-likelihood (Neal
  & Hinton, 1999); the mechanism is that EM maximizes a variational lower bound (the ELBO) that is tight
  after the E-step.
- **No global guarantee.** EM converges to a *local* optimum (or saddle) of the log-likelihood — not
  necessarily the global maximum. Multiple random restarts mitigate poor local optima.
- **Generality.** EM applies whenever marginalizing latents is hard but the complete-data problem is easy;
  the GMM is the canonical instance. See Bishop (2006), Barber (2012), Murphy (2012).

## Connections
- Builds on: [[latent-variable-model]], [[maximum-likelihood-estimation]], [[responsibility]], [[bayes-theorem]]
- Leads to: variational inference, [[gaussian-mixture-model]] training, K-means (hard-assignment limit)
- Used by: [[gaussian-mixture-model]], hidden Markov models, factor analysis
- People: [[arthur-dempster]]

## Sources
- `raw/mml-book.pdf` §11.3 (Eqs. 11.53–11.56) & §11.4.5 (Eqs. 11.73), printed pp. 360–362, 367 (PDF pp. 366–368, 373).
