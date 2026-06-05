---
title: "MML Ch. 9 — Linear Regression"
type: source
tags: [linear-regression, machine-learning, probabilistic-modeling, bayesian]
sources: []
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch. 9 — Linear Regression

- **Raw file:** `raw/mml-book.pdf` §9, printed pp. 289–316 (PDF pp. 295–322)
- **Type:** book chapter
- **Ingested:** 2026-06-04

This is the first Part II ("Central Machine Learning Problems") chapter, applying the
linear-algebra, analytic-geometry, vector-calculus, and probability tools of Chapters 2, 3, 5,
6, 7 to **regression** — fitting a function to noisy data. It walks the same problem up a ladder
of sophistication: point estimate by maximum likelihood, point estimate by MAP (= regularization),
and finally the full Bayesian posterior over parameters.

## Section coverage

| § | Title | Wiki page |
|---|---|---|
| 9.1 | Problem Formulation | [[linear-regression]] (setup, Gaussian likelihood) |
| 9.2 | Parameter Estimation | [[maximum-likelihood-estimation]] |
| 9.2.1 | Maximum Likelihood Estimation (+ features, noise variance) | [[maximum-likelihood-estimation]], [[linear-regression]] (feature matrix) |
| 9.2.2 | Overfitting in Linear Regression | [[overfitting]] |
| 9.2.3 | Maximum A Posteriori Estimation | [[maximum-a-posteriori-estimation]] |
| 9.2.4 | MAP Estimation as Regularization | [[regularized-least-squares]] |
| 9.3 | Bayesian Linear Regression | [[bayesian-linear-regression]] |
| 9.3.1 | Model | [[bayesian-linear-regression]] |
| 9.3.2 | Prior Predictions | [[bayesian-linear-regression]] |
| 9.3.3 | Posterior Distribution (Thm 9.1) | [[bayesian-linear-regression]] |
| 9.3.4 | Posterior Predictions | [[bayesian-linear-regression]] |
| 9.3.5 | Computing the Marginal Likelihood | [[marginal-likelihood]] |
| 9.4 | Maximum Likelihood as Orthogonal Projection | [[maximum-likelihood-estimation]] (geometry), [[linear-regression]] |
| 9.5 | Further Reading (GLMs, GPs, LASSO) | [[generalized-linear-model]] |

## TL;DR (5 bullets)

- **Linear regression** means *linear in the parameters* $\boldsymbol\theta$: the model is $y = \boldsymbol\phi^\top(\mathbf x)\boldsymbol\theta + \epsilon$ with i.i.d. zero-mean Gaussian noise $\epsilon\sim\mathcal N(0,\sigma^2)$. Features $\boldsymbol\phi$ can be arbitrarily nonlinear (e.g. polynomial monomials), so this fits curves too.
- A Gaussian likelihood turns the fitting problem into **least squares**: minimizing the negative log-likelihood gives the closed-form normal-equations solution $\boldsymbol\theta_{\mathrm{ML}} = (\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\mathbf y$.
- ML estimation **overfits** in the small-data / high-flexibility regime (large parameter magnitudes, wild oscillation). A Gaussian **prior** yields the MAP estimate $\boldsymbol\theta_{\mathrm{MAP}} = (\boldsymbol\Phi^\top\boldsymbol\Phi + \tfrac{\sigma^2}{b^2}\mathbf I)^{-1}\boldsymbol\Phi^\top\mathbf y$, which is exactly **ridge / regularized least squares** with $\lambda=\sigma^2/b^2$.
- **Bayesian linear regression** keeps the *whole* Gaussian posterior over $\boldsymbol\theta$ (Thm 9.1) instead of a point estimate, and averages over it for prediction — giving calibrated predictive uncertainty (error bars that widen away from the data). The MAP estimate is the posterior mean.
- Geometrically, ML linear regression is an **orthogonal projection** of the target vector $\mathbf y$ onto the subspace of $\mathbb R^N$ spanned by the feature columns. The chapter closes by generalizing to **GLMs** (logistic regression, the building block of deep nets) and pointing to **Gaussian processes** and **LASSO**.

## Key concepts introduced

- [[linear-regression]] — the parametric model and Gaussian likelihood
- [[maximum-likelihood-estimation]] — least-squares point estimate via the normal equations
- [[overfitting]] — flexible models that fit training data but fail to generalize; RMSE on a test set
- [[maximum-a-posteriori-estimation]] — point estimate that maximizes the posterior (prior + likelihood)
- [[regularized-least-squares]] — ridge regression; MAP = penalizing $\lVert\boldsymbol\theta\rVert^2$
- [[bayesian-linear-regression]] — full posterior over parameters; predictive distribution with uncertainty
- [[marginal-likelihood]] — the evidence $p(Y\mid X)$; basis of Bayesian model selection
- [[generalized-linear-model]] — $y=\sigma(f(\mathbf x))$; logistic regression and the link to neural networks

## Notable results / theorems

- **Normal equations / ML estimator** (Eq. 9.12c, 9.19): $\boldsymbol\theta_{\mathrm{ML}} = (\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\mathbf y$, valid when $\mathrm{rk}(\boldsymbol\Phi)=K$ (so $\boldsymbol\Phi^\top\boldsymbol\Phi$ is invertible / positive definite). The Hessian $\boldsymbol\Phi^\top\boldsymbol\Phi$ is positive definite, so the optimum is the unique global minimum.
- **ML noise variance** (Eq. 9.22): $\sigma^2_{\mathrm{ML}} = \tfrac1N\sum_n (y_n - \boldsymbol\phi^\top(\mathbf x_n)\boldsymbol\theta)^2$ — empirical mean squared residual.
- **MAP estimator** (Eq. 9.31) and **ridge solution** (Eq. 9.34): equal when $\lambda=\sigma^2/b^2$.
- **Theorem 9.1 (Parameter Posterior):** the posterior is Gaussian $\mathcal N(\boldsymbol\theta\mid\mathbf m_N,\mathbf S_N)$ with $\mathbf S_N=(\mathbf S_0^{-1}+\sigma^{-2}\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}$ and $\mathbf m_N=\mathbf S_N(\mathbf S_0^{-1}\mathbf m_0+\sigma^{-2}\boldsymbol\Phi^\top\mathbf y)$ — derived by *completing the square*.
- **Posterior predictive** (Eq. 9.57c): $p(y_*\mid X,Y,\mathbf x_*) = \mathcal N\!\big(\boldsymbol\phi^\top(\mathbf x_*)\mathbf m_N,\; \boldsymbol\phi^\top(\mathbf x_*)\mathbf S_N\boldsymbol\phi(\mathbf x_*)+\sigma^2\big)$.
- **Marginal likelihood** (Eq. 9.64b): $p(Y\mid X)=\mathcal N(\mathbf y\mid \mathbf X\mathbf m_0,\; \mathbf X\mathbf S_0\mathbf X^\top+\sigma^2\mathbf I)$.

## What this changes in the wiki

- Establishes the first concrete ML *method* pages (Part II). Depends on Chapter 6 ([[gaussian-distribution]], [[bayes-theorem]], conjugacy, marginalization) and Chapter 3 ([[orthogonal-projection]]) — all ingested in the same batch, so these cross-links resolve.
- Introduces [[thomas-bayes]] (person) for Bayes' theorem, used by MAP and Bayesian regression.
- No contradictions with existing linear-algebra pages; reuses [[system-of-linear-equations]] (the normal equations are a linear system) and [[basis-and-rank]] (rank condition for invertibility).
