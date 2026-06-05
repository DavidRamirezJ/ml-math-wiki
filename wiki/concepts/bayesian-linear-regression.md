---
title: Bayesian Linear Regression
type: concept
tags: [machine-learning, bayesian, probabilistic-modeling, uncertainty]
difficulty: advanced
sources: [mml-ch09-linear-regression]
created: 2026-06-04
updated: 2026-06-04
---

# Bayesian Linear Regression

> **One-liner:** Instead of choosing one best curve, keep *all* curves that are consistent with the data — weighted by how plausible each is — and average over them to predict, getting honest error bars for free.

## 🟢 Intuition (Dummy)
Maximum likelihood and MAP hand you a single curve and say "this is the answer." But with little or
noisy data, lots of different curves fit almost equally well. Picking just one throws away that
ambiguity — and pretends to be more certain than it should be.

Bayesian linear regression refuses to commit. It keeps a whole *distribution* over possible curves:
some very likely, some less so. To predict at a new point, it asks every plausible curve what it
thinks and averages the answers, weighting by plausibility. The payoff: the prediction comes with a
**confidence band** that is narrow where you have data and fans out wide where you don't — exactly the
honesty you want before making a risky decision.

## 🟡 Mechanics (Practitioner)
**Model.** Place a Gaussian prior on the parameters and keep the Gaussian likelihood:
$$ p(\boldsymbol\theta)=\mathcal N(\mathbf m_0,\mathbf S_0), \qquad p(y\mid\mathbf x,\boldsymbol\theta)=\mathcal N\!\big(y\mid\boldsymbol\phi^\top(\mathbf x)\boldsymbol\theta,\sigma^2\big). $$

**Posterior over parameters (Theorem 9.1).** By Bayes' theorem the posterior is again Gaussian:
$$ p(\boldsymbol\theta\mid X,Y)=\mathcal N(\boldsymbol\theta\mid\mathbf m_N,\mathbf S_N), $$
$$ \mathbf S_N = \big(\mathbf S_0^{-1} + \sigma^{-2}\boldsymbol\Phi^\top\boldsymbol\Phi\big)^{-1}, \qquad \mathbf m_N = \mathbf S_N\big(\mathbf S_0^{-1}\mathbf m_0 + \sigma^{-2}\boldsymbol\Phi^\top\mathbf y\big). $$
The posterior mean $\mathbf m_N$ **equals the [[maximum-a-posteriori-estimation|MAP]] estimate** (mode = mean
for a Gaussian).

**Posterior predictive distribution.** To predict at a test input $\mathbf x_*$, integrate out
$\boldsymbol\theta$:
$$ p(y_*\mid X,Y,\mathbf x_*) = \int p(y_*\mid\mathbf x_*,\boldsymbol\theta)\,p(\boldsymbol\theta\mid X,Y)\,d\boldsymbol\theta = \mathcal N\!\Big(y_*\;\big|\;\boldsymbol\phi^\top(\mathbf x_*)\mathbf m_N,\;\; \boldsymbol\phi^\top(\mathbf x_*)\mathbf S_N\boldsymbol\phi(\mathbf x_*) + \sigma^2\Big). $$
The predictive variance has two parts:
- $\boldsymbol\phi^\top(\mathbf x_*)\mathbf S_N\boldsymbol\phi(\mathbf x_*)$ — **parameter uncertainty** (grows
  far from the data),
- $\sigma^2$ — irreducible **measurement noise** (drop it for noise-free function values $f(\mathbf x_*)$).

**Prior predictions.** Even with *no data* you can predict using the prior:
$p(y_*\mid\mathbf x_*)=\mathcal N(\boldsymbol\phi^\top(\mathbf x_*)\mathbf m_0,\ \boldsymbol\phi^\top(\mathbf x_*)\mathbf S_0\boldsymbol\phi(\mathbf x_*)+\sigma^2)$.

**ML usage.** This is the workhorse for regression with calibrated uncertainty — active learning,
Bayesian optimization, safe decision-making in robotics/RL. It is the finite-feature cousin of
[[gaussian-process|Gaussian processes]]: sampling $\boldsymbol\theta_i$ from the posterior and forming
$\boldsymbol\phi^\top(\cdot)\boldsymbol\theta_i$ yields sample *functions*, so the parameter distribution
induces a **distribution over functions**.

## 🔴 Theory (Expert)
- **Proof of Thm 9.1 (completing the square).** In log-space the posterior is
  $-\tfrac12\big(\boldsymbol\theta^\top(\sigma^{-2}\boldsymbol\Phi^\top\boldsymbol\Phi+\mathbf S_0^{-1})\boldsymbol\theta
  -2(\sigma^{-2}\boldsymbol\Phi^\top\mathbf y+\mathbf S_0^{-1}\mathbf m_0)^\top\boldsymbol\theta\big)+\text{const}$.
  This is a negative quadratic form in $\boldsymbol\theta$ ⇒ the posterior is Gaussian. Matching to the
  generic form $\mathbf x^\top\mathbf A\mathbf x - 2\mathbf a^\top\mathbf x$ with $\mathbf A=\boldsymbol\Sigma^{-1}$,
  $\boldsymbol\mu=\mathbf A^{-1}\mathbf a$ reads off $\mathbf S_N^{-1}=\mathbf A$ and $\mathbf m_N=\mathbf S_N\mathbf a$.
  Conjugacy (Gaussian prior + Gaussian likelihood ⇒ Gaussian posterior) is what makes this closed-form.
- **Predictive moments by affine rules.** Because $y_*=\boldsymbol\phi^\top(\mathbf x_*)\boldsymbol\theta+\epsilon$
  is an affine function of the Gaussian $\boldsymbol\theta$ plus independent Gaussian noise, the predictive
  mean/variance follow from the standard mean/covariance rules for affine transforms of Gaussians:
  $\mathbb V[y_*]=\mathbb V_{\boldsymbol\theta}[\boldsymbol\phi^\top\boldsymbol\theta]+\mathbb V_\epsilon[\epsilon]$.
- **MAP / ML as limits.** As the prior widens ($\mathbf S_0^{-1}\to\mathbf 0$, i.e. $b^2\to\infty$),
  $\mathbf m_N\to\boldsymbol\theta_{\mathrm{ML}}$ and the parameter regularization vanishes. For finite prior,
  $\mathbf m_N=\boldsymbol\theta_{\mathrm{MAP}}=\boldsymbol\theta_{\mathrm{RLS}}$ with $\lambda=\sigma^2/b^2$. So
  ML ⊂ MAP ⊂ Bayesian is a ladder of increasing principle.
- **Why it beats overfitting.** Averaging over the posterior, rather than reporting its mode, means a
  flexible model with poorly-constrained parameters expresses its ignorance as *large predictive
  variance* instead of a single overconfident wiggly curve — see the degree-7 example (Fig. 9.11) where
  the MAP fit looks fine but the posterior reveals huge uncertainty.

## Connections
- Builds on: [[maximum-a-posteriori-estimation]], [[bayes-theorem]], [[gaussian-distribution]], [[linear-regression]], [[marginal-likelihood]]
- Leads to: [[gaussian-process]], Bayesian optimization, [[marginal-likelihood|model selection]]
- Used by: active learning, reinforcement learning, robotics (uncertainty-aware prediction)
- People: [[thomas-bayes]]

## Sources
- `raw/mml-book.pdf` §9.3 (Eqs. 9.35–9.59, Theorem 9.1), printed pp. 303–311 (PDF pp. 309–317).
