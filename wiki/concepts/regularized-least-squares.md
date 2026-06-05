---
title: Regularized Least Squares
type: concept
tags: [machine-learning, regularization, optimization, linear-regression]
difficulty: intermediate
sources: [mml-ch09-linear-regression]
created: 2026-06-04
updated: 2026-06-04
---

# Regularized Least Squares

> **One-liner:** Plain least squares but with an extra penalty that keeps the parameters small, so the fit stays smooth and doesn't chase noise.

## 🟢 Intuition (Dummy)
Least squares only cares about hugging the data points. Give it a flexible enough model and it will
contort itself into a wild shape to get closer to every point — including the noisy ones. Regularized
least squares adds a second goal: *also keep the knobs (parameters) small*. Now the fit must trade off
"match the data" against "stay tame." The result is a smoother curve that generalizes better.

The strength of the "stay tame" goal is one dial, $\lambda$. Turn it to zero and you're back to plain
least squares. Turn it up and the curve gets flatter and simpler.

## 🟡 Mechanics (Practitioner)
Add an $\ell_2$ penalty (the **regularizer**) to the squared-error loss:
$$ \mathcal L_{\mathrm{RLS}}(\boldsymbol\theta) = \underbrace{\lVert\mathbf y-\boldsymbol\Phi\boldsymbol\theta\rVert^2}_{\text{data-fit / misfit term}} + \underbrace{\lambda\,\lVert\boldsymbol\theta\rVert_2^2}_{\text{regularizer}}, \qquad \lambda\ge 0. $$
The **regularization parameter** $\lambda$ controls strictness. Minimizing gives the closed form
(ridge regression):
$$ \boxed{\;\boldsymbol\theta_{\mathrm{RLS}} = (\boldsymbol\Phi^\top\boldsymbol\Phi + \lambda\mathbf I)^{-1}\boldsymbol\Phi^\top\mathbf y\;} $$

**This is identical to the [[maximum-a-posteriori-estimation|MAP]] estimate** under a Gaussian prior
$p(\boldsymbol\theta)=\mathcal N(\mathbf 0,b^2\mathbf I)$, with the correspondence
$$ \lambda = \frac{\sigma^2}{b^2}, $$
where $\sigma^2$ is the noise variance and $b^2$ the prior variance. The regularizer *is* a negative
log-Gaussian prior: $-\log p(\boldsymbol\theta)=\tfrac{1}{2b^2}\lVert\boldsymbol\theta\rVert_2^2+\text{const}$.

**Worked intuition.** With one parameter, $\theta_{\mathrm{RLS}}=\tfrac{\boldsymbol\Phi^\top\mathbf y}{\boldsymbol\Phi^\top\boldsymbol\Phi+\lambda}$.
The $+\lambda$ in the denominator shrinks the estimate toward 0 — exactly the "keep it small" effect.

**ML usage.** This is **ridge regression** (a.k.a. Tikhonov regularization, weight decay in deep
learning). Picking $\lambda$ is done by cross-validation. Swapping the $\ell_2$ for an $\ell_1$ penalty
($\lambda\lVert\boldsymbol\theta\rVert_1$) gives **LASSO**, which produces *sparse* solutions (many
$\theta_d=0$) for automatic variable selection.

## 🔴 Theory (Expert)
- **Conditioning.** $\boldsymbol\Phi^\top\boldsymbol\Phi$ is symmetric positive semi-definite (singular when
  features are collinear or $K>N$); adding $\lambda\mathbf I$ ($\lambda>0$) shifts every eigenvalue up by
  $\lambda$, making it strictly positive definite and well-conditioned. So regularization guarantees a
  unique solution and tames numerical instability.
- **Loss ↔ prior duality.** Regularizing is equivalent to MAP: the data-fit term is proportional to the
  negative log-likelihood, the penalty to the negative log-prior. This is the chapter's central message
  that *loss functions are induced by probabilistic models*. $\ell_2\leftrightarrow$ Gaussian prior;
  $\ell_1\leftrightarrow$ Laplace prior (sharply peaked at 0, hence sparsity).
- **$p$-norms and sparsity.** Using a general $\lVert\cdot\rVert_p$ penalty, smaller $p$ yields sparser
  solutions. $p=1$ is the convex sparsity-inducing limit (LASSO, Tibshirani 1996); $p<1$ is sparser but
  non-convex.
- **Still a point estimate.** Like MAP, RLS returns one $\boldsymbol\theta$; it reduces but does not
  eliminate [[overfitting]] and gives no predictive uncertainty. The fully probabilistic treatment is
  [[bayesian-linear-regression]].

## Connections
- Builds on: [[maximum-likelihood-estimation]], [[maximum-a-posteriori-estimation]], [[linear-regression]]
- Leads to: [[bayesian-linear-regression]], LASSO / sparse models
- Used by: ridge regression, weight decay in neural nets, ill-posed inverse problems
- People: —

## Sources
- `raw/mml-book.pdf` §9.2.4 (Eqs. 9.32–9.34), printed pp. 302–303 (PDF pp. 308–309).
