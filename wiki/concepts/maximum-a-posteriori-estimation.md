---
title: Maximum A Posteriori Estimation
type: concept
tags: [machine-learning, estimation, bayesian, probabilistic-modeling]
difficulty: intermediate
sources: [mml-ch09-linear-regression]
created: 2026-06-04
updated: 2026-06-04
---

# Maximum A Posteriori Estimation

> **One-liner:** Like maximum likelihood, but you also bring in a prior belief about which parameter values are reasonable, then pick the parameters that best balance that belief against the data.

## 🟢 Intuition (Dummy)
Maximum likelihood listens *only* to the data. If the data is noisy or scarce, it can be talked into
crazy answers. Maximum a posteriori (MAP) adds a voice of common sense — a **prior** — that says
"before I even look at the data, I expect the parameters to be modest, not enormous." The final
estimate is a compromise: the data pulls one way, the prior pulls toward sensible values, and you
land in between.

For curve fitting, the prior "parameters should be small" stops the curve from oscillating wildly to
chase noise. That's why MAP is a built-in defense against [[overfitting]].

## 🟡 Mechanics (Practitioner)
By Bayes' theorem the **posterior** over parameters is
$$ p(\boldsymbol\theta\mid X,Y) = \frac{p(Y\mid X,\boldsymbol\theta)\,p(\boldsymbol\theta)}{p(Y\mid X)} \;\propto\; p(Y\mid X,\boldsymbol\theta)\,p(\boldsymbol\theta). $$
The **MAP estimate** maximizes it — equivalently, minimizes the negative log-posterior, which is
just NLL **+** negative log-prior:
$$ \boldsymbol\theta_{\mathrm{MAP}} \in \arg\min_{\boldsymbol\theta}\;\big\{ -\log p(Y\mid X,\boldsymbol\theta) - \log p(\boldsymbol\theta)\big\}. $$
The evidence $p(Y\mid X)$ is constant in $\boldsymbol\theta$, so it drops out.

**Applied to [[linear-regression]] with a Gaussian prior** $p(\boldsymbol\theta)=\mathcal N(\mathbf 0, b^2\mathbf I)$:
$$ -\log p(\boldsymbol\theta\mid X,Y) = \tfrac{1}{2\sigma^2}\lVert\mathbf y-\boldsymbol\Phi\boldsymbol\theta\rVert^2 + \tfrac{1}{2b^2}\boldsymbol\theta^\top\boldsymbol\theta + \text{const}. $$
Setting the gradient to zero gives the **MAP estimator**:
$$ \boxed{\;\boldsymbol\theta_{\mathrm{MAP}} = \Big(\boldsymbol\Phi^\top\boldsymbol\Phi + \tfrac{\sigma^2}{b^2}\mathbf I\Big)^{-1}\boldsymbol\Phi^\top\mathbf y\;} $$
Compare to the ML estimator $\boldsymbol\theta_{\mathrm{ML}} = (\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\mathbf y$:
**the only change is the added $\tfrac{\sigma^2}{b^2}\mathbf I$ inside the inverse.** This is identical to
[[regularized-least-squares|ridge regression]] with $\lambda=\sigma^2/b^2$.

**Effect.** The added diagonal term shrinks the parameters toward the prior mean ($\mathbf 0$ here),
keeping high-degree polynomial fits smoother (Fig. 9.7) without dropping flexibility. A stronger
prior (smaller $b^2$) ⇒ larger $\lambda$ ⇒ more shrinkage.

## 🔴 Theory (Expert)
- **Always invertible.** $\boldsymbol\Phi^\top\boldsymbol\Phi$ is only positive *semi*-definite (singular if
  $K>N$), but adding the strictly positive-definite $\tfrac{\sigma^2}{b^2}\mathbf I$ makes the sum
  symmetric positive definite, so the inverse always exists and $\boldsymbol\theta_{\mathrm{MAP}}$ is the
  unique solution of a [[system-of-linear-equations]] — even when the pure ML problem is degenerate.
  This is why regularization also *numerically* stabilizes least squares.
- **MAP = regularized ML.** The negative log of a zero-mean Gaussian prior is
  $\tfrac{1}{2b^2}\lVert\boldsymbol\theta\rVert_2^2+\text{const}$, i.e. exactly an $\ell_2$ penalty. Different
  priors induce different regularizers: a **Laplace prior** gives the $\ell_1$ / LASSO penalty, which
  drives parameters to exactly zero (sparsity, variable selection).
- **MAP is the posterior mode = posterior mean here.** For the Gaussian model the posterior is itself
  Gaussian (see [[bayesian-linear-regression]], Thm 9.1), so its mode equals its mean:
  $\boldsymbol\theta_{\mathrm{MAP}}=\mathbf m_N$. This is special to symmetric unimodal posteriors; in general
  the mode and mean differ.
- **Still a point estimate.** MAP returns a single $\boldsymbol\theta$ and discards the spread of the
  posterior, so it cannot report predictive uncertainty and only *delays* overfitting rather than
  curing it. Marginalizing over $\boldsymbol\theta$ ([[bayesian-linear-regression]]) is the principled fix.

## Connections
- Builds on: [[maximum-likelihood-estimation]], [[bayes-theorem]], [[gaussian-distribution]], [[gradient]]
- Leads to: [[regularized-least-squares]], [[bayesian-linear-regression]]
- Used by: ridge regression, LASSO, regularized neural-net training (weight decay)
- People: [[thomas-bayes]]

## Sources
- `raw/mml-book.pdf` §9.2.3 (Eqs. 9.24–9.31, Fig. 9.7), printed pp. 300–302 (PDF pp. 306–308).
