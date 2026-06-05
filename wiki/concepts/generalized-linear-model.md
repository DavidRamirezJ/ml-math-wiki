---
title: Generalized Linear Model
type: concept
tags: [machine-learning, classification, regression, neural-networks]
difficulty: intermediate
sources: [mml-ch09-linear-regression]
created: 2026-06-04
updated: 2026-06-04
---

# Generalized Linear Model

> **One-liner:** Take a linear regression and squash its output through a fixed nonlinear function, so you can predict things like probabilities or counts — and stacking these is exactly a neural network.

## 🟢 Intuition (Dummy)
Linear regression predicts a number that can be anything from $-\infty$ to $+\infty$. But sometimes the
thing you want to predict is restricted: a probability must live in $[0,1]$; a count must be a
non-negative integer. A raw line can't respect those limits.

The fix: keep the linear part to do the "combining of inputs," then pass its output through a fixed
shaping function that maps it into the allowed range. For probabilities, an S-shaped "sigmoid" squashes
any number into $[0,1]$. That combination — linear core + squashing function — is a **generalized linear
model**. And here's the kicker: a single artificial neuron is *exactly* this, and stacking many of them
gives a deep neural network.

## 🟡 Mechanics (Practitioner)
A GLM relates the linear predictor $f(\mathbf x)=\boldsymbol\theta^\top\boldsymbol\phi(\mathbf x)$ (the
[[linear-regression]] model) to the output via a smooth, invertible **activation / transfer function**
$\sigma$:
$$ y = \sigma\big(f(\mathbf x)\big), \qquad f(\mathbf x) = \boldsymbol\theta^\top\boldsymbol\phi(\mathbf x). $$
The inverse $\sigma^{-1}$ is the **canonical link function**. Ordinary linear regression is the special
case $\sigma=\text{identity}$.

**Logistic regression** (binary classification) uses the **logistic sigmoid**
$$ \sigma(f) = \frac{1}{1+\exp(-f)} \in [0,1], $$
interpreted as $p(y=1)$ of a Bernoulli outcome $y\in\{0,1\}$. Other likelihoods cover other data types:
Binomial/Poisson for counts, Gaussian (identity link) for real-valued regression.

**Connection to neural networks.** A single GLM layer is
$$ \mathbf x_{k+1} = f_k(\mathbf x_k), \qquad f_k(\mathbf x_k) = \sigma_k(\mathbf A_k\mathbf x_k + \mathbf b_k), $$
with weight matrix $\mathbf A_k$ and bias $\mathbf b_k$. Compose $K$ of them and
$f_{K-1}\circ\cdots\circ f_0$ is a **$K$-layer deep neural network**. So GLMs are literally the building
blocks of feedforward nets — each layer is an [[affine-space|affine map]] followed by a nonlinearity.

**ML usage.** Logistic regression is the canonical baseline classifier; softmax regression is its
multi-class version; GLMs underlie much of statistics (insurance, epidemiology, econometrics). The
GLM view is the conceptual bridge from linear regression to deep learning.

## 🔴 Theory (Expert)
- **Not linear in outputs.** Despite the name, $y=\sigma(f(\mathbf x))$ is *not* linear in $\boldsymbol\theta$
  once $\sigma$ is nonlinear — "linear" refers to the linear predictor $f$ inside. The exponential-family
  likelihood + link-function structure is what earns the "generalized linear" label.
- **Loss from likelihood.** As in [[maximum-likelihood-estimation]], the loss is the negative
  log-likelihood of the chosen exponential-family distribution: Bernoulli ⇒ cross-entropy/log-loss,
  Gaussian ⇒ squared error, Poisson ⇒ Poisson deviance. The canonical link makes the NLL convex in
  $\boldsymbol\theta$ for a single GLM.
- **Loss of closed form.** Unlike linear-Gaussian regression, GLM fitting generally has no closed form;
  one uses iteratively reweighted least squares or gradient descent. For *deep* compositions of GLMs the
  objective becomes **non-convex**, and full Bayesian marginalization over weights is analytically
  intractable — the gap that approximate inference (variational, Laplace, MCMC) and, in the
  infinite-width limit, [[gaussian-process|Gaussian processes]] address.
- **Beyond Gaussian priors.** The chapter's *Further Reading* also notes the **Laplace prior** ⇒ $\ell_1$
  / LASSO regularization for sparse, interpretable models in the underdetermined ($N\ll D$) regime.

## Connections
- Builds on: [[linear-regression]], [[maximum-likelihood-estimation]], [[affine-space]]
- Leads to: logistic regression, softmax classification, deep neural networks, [[gaussian-process]]
- Used by: classification, count regression, deep learning layers
- People: —

## Sources
- `raw/mml-book.pdf` §9.5 Further Reading (Eq. 9.72), printed pp. 315–316 (PDF pp. 321–322).
