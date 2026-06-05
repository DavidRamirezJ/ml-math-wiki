---
title: Maximum Likelihood Estimation
type: concept
tags: [machine-learning, estimation, probabilistic-modeling, optimization]
difficulty: intermediate
sources: [mml-ch09-linear-regression]
created: 2026-06-04
updated: 2026-06-04
---

# Maximum Likelihood Estimation

> **One-liner:** Pick the parameter values that make the data you actually observed as unsurprising as possible.

## 🟢 Intuition (Dummy)
Imagine you have a dice-rolling machine with an unknown bias, and you watch it roll. Maximum
likelihood asks: "Of all the possible biases this machine could have, which one would make *exactly
the rolls I just saw* the most likely?" You then declare that bias to be your best guess.

For curve fitting it's the same idea. Each candidate curve assigns a probability to the data points
you measured (because the points are noisy, no curve makes them certain). Maximum likelihood (ML)
picks the curve under which your measured points are the most probable. For the usual bell-curve
(Gaussian) noise, this turns out to be exactly the curve that minimizes the total squared distance
to the points — least squares.

## 🟡 Mechanics (Practitioner)
Given a parametrized likelihood $p(\mathcal D\mid\boldsymbol\theta)$ over a dataset $\mathcal D$, the
**ML estimate** is
$$ \boldsymbol\theta_{\mathrm{ML}} \in \arg\max_{\boldsymbol\theta}\; p(\mathcal D\mid\boldsymbol\theta). $$
Because data points are i.i.d., the likelihood factorizes, and we work with the **negative
log-likelihood** (NLL) — sums beat products numerically and differentiate cleanly:
$$ -\log p(Y\mid X,\boldsymbol\theta) = -\sum_{n=1}^N \log p(y_n\mid \mathbf x_n,\boldsymbol\theta). $$

**Applied to [[linear-regression]] with Gaussian noise.** Each factor is
$\log p(y_n\mid\mathbf x_n,\boldsymbol\theta) = -\tfrac{1}{2\sigma^2}(y_n-\mathbf x_n^\top\boldsymbol\theta)^2+\text{const}$,
so the NLL (dropping constants) is the **least-squares loss**:
$$ \mathcal L(\boldsymbol\theta) = \tfrac{1}{2\sigma^2}\lVert \mathbf y - \boldsymbol\Phi\boldsymbol\theta\rVert^2. $$
Take the gradient, set it to zero:
$$ \frac{d\mathcal L}{d\boldsymbol\theta} = \tfrac{1}{\sigma^2}\big(\boldsymbol\theta^\top\boldsymbol\Phi^\top\boldsymbol\Phi - \mathbf y^\top\boldsymbol\Phi\big) = \mathbf 0^\top \;\Longrightarrow\; \boxed{\;\boldsymbol\theta_{\mathrm{ML}} = (\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\mathbf y\;} $$
These are the **normal equations**. No iterative optimization needed — there is a closed form.

**Geometric view (§9.4): an orthogonal projection.** In the no-feature 1-D case $y=x\theta$,
$$ \boldsymbol\Phi\boldsymbol\theta_{\mathrm{ML}} = \frac{\mathbf X\mathbf X^\top}{\mathbf X^\top\mathbf X}\,\mathbf y, $$
and $\tfrac{\mathbf X\mathbf X^\top}{\mathbf X^\top\mathbf X}$ is exactly the projection matrix onto the line
spanned by $\mathbf X$. Generally, $\boldsymbol\Phi\boldsymbol\theta_{\mathrm{ML}}$ is the
[[orthogonal-projection]] of the target vector $\mathbf y$ onto $\mathrm{col}(\boldsymbol\Phi)$, the
$K$-dimensional subspace of $\mathbb R^N$ spanned by the feature columns — the point in that subspace
*closest* to $\mathbf y$.

**Worked example.** Fit $y=\theta x$ to $(1,2),(2,5),(3,5)$: $\boldsymbol\Phi^\top\boldsymbol\Phi=14$,
$\boldsymbol\Phi^\top\mathbf y=27$, so $\theta_{\mathrm{ML}}=27/14\approx1.93$.

**ML usage.** ML is the default training principle across ML: it underlies cross-entropy loss
(Bernoulli/categorical likelihoods), softmax classifiers, and most neural-net objectives. For
linear-Gaussian models it has a closed form; for nonlinear models you do gradient descent on the NLL.

## 🔴 Theory (Expert)
- **Optimality conditions.** $\boldsymbol\Phi^\top\boldsymbol\Phi$ is the Hessian; it is symmetric
  positive definite when $\mathrm{rk}(\boldsymbol\Phi)=K$, so $\nabla=\mathbf 0$ is necessary *and*
  sufficient for the unique global minimum (strictly convex quadratic). The right-multiplication by
  $(\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}$ in the derivation is licensed by this positive-definiteness.
- **The likelihood is not a distribution over $\boldsymbol\theta$.** $p(Y\mid X,\boldsymbol\theta)$ is a
  *function* of $\boldsymbol\theta$ that need not integrate to 1 over $\boldsymbol\theta$ (it is a normalized
  density in $y$, not in $\boldsymbol\theta$). This is the conceptual gap that [[maximum-a-posteriori-estimation|MAP]]
  and [[bayesian-linear-regression]] close by adding a prior.
- **ML noise variance.** Differentiating the full log-likelihood w.r.t. $\sigma^2$ gives
  $\sigma^2_{\mathrm{ML}}=\tfrac1N\sum_n(y_n-\boldsymbol\phi^\top(\mathbf x_n)\boldsymbol\theta)^2$ (Eq. 9.22),
  the mean squared residual — a biased estimator.
- **Orthonormal features.** If feature functions are orthonormal then $\boldsymbol\Phi^\top\boldsymbol\Phi=\mathbf I$,
  the projection becomes $\boldsymbol\Phi\boldsymbol\Phi^\top\mathbf y=\sum_k \boldsymbol\phi_k\boldsymbol\phi_k^\top\mathbf y$ —
  a decoupled sum of projections onto each basis vector (the principle behind Fourier/wavelet
  regression). Non-orthogonal bases can be orthogonalized via Gram–Schmidt.
- **Failure mode.** ML maximizes fit to the *training* data only, so it overfits flexible models;
  see [[overfitting]]. It produces a single *point estimate* with no uncertainty.

## Connections
- Builds on: [[linear-regression]], [[gaussian-distribution]], [[gradient]], [[system-of-linear-equations]], [[orthogonal-projection]]
- Leads to: [[maximum-a-posteriori-estimation]], [[regularized-least-squares]], [[bayesian-linear-regression]]
- Used by: least squares, logistic regression, neural-net training (cross-entropy)
- People: [[carl-friedrich-gauss]] (least squares)

## Sources
- `raw/mml-book.pdf` §9.2.1 (Eqs. 9.7–9.22) and §9.4 (Eqs. 9.65–9.71), printed pp. 293–298, 313–315 (PDF pp. 299–304, 319–321).
