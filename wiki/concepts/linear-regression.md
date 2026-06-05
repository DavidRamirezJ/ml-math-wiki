---
title: Linear Regression
type: concept
tags: [linear-regression, machine-learning, supervised-learning, probabilistic-modeling]
difficulty: beginner
sources: [mml-ch09-linear-regression]
created: 2026-06-04
updated: 2026-06-04
---

# Linear Regression

> **One-liner:** Linear regression draws the best line (or curve) through a cloud of noisy data points so you can predict the value at a new point.

## 🟢 Intuition (Dummy)
You have a scatter of dots — say, house size on the bottom and price up the side. They roughly
trend upward but don't sit on a perfect line because of randomness ("noise"). Linear regression
finds the single straight line that comes *closest* to all the dots at once, so that for a new
house size you can read off a predicted price.

"Closest" means: for each dot, measure how far the line misses it vertically, square that gap, add
up all the squared gaps, and pick the line that makes the total smallest. The squaring punishes big
misses hard and makes the math clean.

The surprising part: "linear" doesn't mean you can only fit straight lines. If you first transform
the input — square it, cube it, take its sine — and then fit a line to those transformed numbers,
you can bend the curve as much as you like. The model stays "linear" because the *knobs you tune*
(the weights) still combine in a simple, straight-line way.

## 🟡 Mechanics (Practitioner)
**The model.** We assume each observed target is a function of the input plus zero-mean Gaussian
noise:
$$ y = f(\mathbf x) + \epsilon, \qquad \epsilon \sim \mathcal N(0,\sigma^2). $$
"Linear regression" means the parameters $\boldsymbol\theta$ enter *linearly*. The general form uses a
**feature map** $\boldsymbol\phi:\mathbb R^D\to\mathbb R^K$:
$$ p(y\mid \mathbf x,\boldsymbol\theta) = \mathcal N\!\big(y \mid \boldsymbol\phi^\top(\mathbf x)\boldsymbol\theta,\; \sigma^2\big), \qquad y = \boldsymbol\phi^\top(\mathbf x)\boldsymbol\theta + \epsilon = \sum_{k=0}^{K-1}\theta_k\,\phi_k(\mathbf x) + \epsilon. $$
With $\boldsymbol\phi(\mathbf x)=\mathbf x$ this is an ordinary line through the origin; with
**polynomial features** $\boldsymbol\phi(x)=\big[1,x,x^2,\dots,x^{K-1}\big]^\top$ it fits polynomials of
degree $K-1$ — still linear regression, because $\boldsymbol\theta$ appears linearly.

**Design / feature matrix.** Stack the $N$ training feature vectors as rows:
$$ \boldsymbol\Phi = \begin{bmatrix} \boldsymbol\phi^\top(\mathbf x_1)\\ \vdots\\ \boldsymbol\phi^\top(\mathbf x_N)\end{bmatrix} \in \mathbb R^{N\times K}, \qquad \mathbf y=[y_1,\dots,y_N]^\top. $$
For a degree-2 polynomial the rows are $\big[1,\;x_n,\;x_n^2\big]$.

**Fitting.** Because the noise is Gaussian, fitting reduces to **least squares**: minimize
$\lVert \mathbf y - \boldsymbol\Phi\boldsymbol\theta\rVert^2$. This has the closed-form solution (the
*normal equations*) — see [[maximum-likelihood-estimation]] for the derivation:
$$ \boldsymbol\theta_{\mathrm{ML}} = (\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\mathbf y. $$

**Worked example (fit a line through origin, 1-D).** Data $(1,2),(2,5),(3,5)$, model $y=\theta x$,
so $\boldsymbol\Phi = [1,2,3]^\top$, $\mathbf y=[2,5,5]^\top$.
$\boldsymbol\Phi^\top\boldsymbol\Phi = 1+4+9 = 14$, $\boldsymbol\Phi^\top\mathbf y = 2+10+15 = 27$, hence
$\theta_{\mathrm{ML}} = 27/14 \approx 1.93$. Prediction at $x_*=4$: $y_* \approx 7.71$.

**ML usage.** Regression underlies forecasting, sensor calibration, line searches in optimization,
forward/inverse models in robotics, and is the backbone of classification (via [[generalized-linear-model|GLMs]]).
The same machinery scales up: a fully-connected neural-net layer without nonlinearity *is* a linear
regression of its outputs on its inputs.

## 🔴 Theory (Expert)
- **Why Gaussian noise ⇒ least squares.** The per-point log-likelihood is
  $\log p(y_n\mid\mathbf x_n,\boldsymbol\theta) = -\tfrac{1}{2\sigma^2}(y_n-\boldsymbol\phi^\top(\mathbf x_n)\boldsymbol\theta)^2 + \text{const}$,
  so maximizing likelihood is exactly minimizing the sum of squared residuals. The choice of *loss*
  is therefore not arbitrary — it is *induced* by the noise model. A Laplace noise model would give
  an $\ell_1$ (least-absolute-deviation) loss instead.
- **Existence/uniqueness.** $\boldsymbol\Phi^\top\boldsymbol\Phi\in\mathbb R^{K\times K}$ is invertible iff
  $\mathrm{rk}(\boldsymbol\Phi)=K$, i.e. the feature columns are [[linear-independence|linearly independent]]
  (requires $N\ge K$ and no degenerate features). Then $\boldsymbol\Phi^\top\boldsymbol\Phi$ is symmetric
  positive definite, the objective is strictly convex, and $\boldsymbol\theta_{\mathrm{ML}}$ is the unique
  global minimizer. For $K>N$ (more parameters than data) the system is underdetermined and there are
  infinitely many least-squares solutions.
- **Linear-algebra view.** Solving $\boldsymbol\Phi\boldsymbol\theta\approx\mathbf y$ is a (generally
  inconsistent) [[system-of-linear-equations]]; the normal equations
  $\boldsymbol\Phi^\top\boldsymbol\Phi\,\boldsymbol\theta = \boldsymbol\Phi^\top\mathbf y$ are its least-squares
  closure. Geometrically $\boldsymbol\Phi\boldsymbol\theta_{\mathrm{ML}}$ is the
  [[orthogonal-projection]] of $\mathbf y$ onto $\mathrm{col}(\boldsymbol\Phi)\subseteq\mathbb R^N$ (see
  [[maximum-likelihood-estimation]] §9.4).
- **Estimating $\sigma^2$.** The ML noise variance is the mean squared residual,
  $\sigma^2_{\mathrm{ML}}=\tfrac1N\lVert\mathbf y-\boldsymbol\Phi\boldsymbol\theta_{\mathrm{ML}}\rVert^2$
  (Eq. 9.22). This is the biased ML estimator (no $N-K$ correction).
- **Model selection.** Polynomial degree $M$ is a hyperparameter; higher $M$ never increases
  *training* error but eventually blows up *test* error — see [[overfitting]].

## Connections
- Builds on: [[vector]], [[matrix]], [[system-of-linear-equations]], [[basis-and-rank]], [[gaussian-distribution]]
- Leads to: [[maximum-likelihood-estimation]], [[maximum-a-posteriori-estimation]], [[bayesian-linear-regression]], [[generalized-linear-model]]
- Used by: forecasting, robotics, [[regularized-least-squares|ridge regression]], neural network layers
- People: [[thomas-bayes]], [[carl-friedrich-gauss]] (least squares / Gaussian)

## Sources
- `raw/mml-book.pdf` §9.1–9.2.1, Eqs. (9.1)–(9.19), printed pp. 289–297 (PDF pp. 295–303).
