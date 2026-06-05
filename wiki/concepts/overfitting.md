---
title: Overfitting
type: concept
tags: [machine-learning, model-selection, generalization]
difficulty: beginner
sources: [mml-ch09-linear-regression]
created: 2026-06-04
updated: 2026-06-04
---

# Overfitting

> **One-liner:** A model overfits when it memorizes the noise in the training data instead of learning the real pattern, so it predicts badly on new data.

> This page is the **regression-specific** treatment (Ch. 9). For the general concept, including underfitting and the risk framing, see [[overfitting-and-underfitting]] (Ch. 8).

## 🟢 Intuition (Dummy)
Suppose you're given 10 dots and asked to draw a curve through them. A wiggly enough curve can pass
*through every single dot exactly* — perfect score on those 10. But the dots were noisy, so that
wiggly curve has bent itself to chase the random jitter. Show it a new dot and it guesses wildly,
because it learned the noise, not the trend.

That's overfitting: the model fits the training data "too well." A simpler curve that misses the dots
a little often predicts new points far better. The goal isn't to ace the practice questions — it's to
do well on the real exam (unseen data). That ability is called **generalization**.

## 🟡 Mechanics (Practitioner)
In polynomial [[linear-regression]] the model flexibility is the **degree** $M$. As $M$ grows the model
can fit more wiggles. The diagnostic is to split data into a **training set** and a held-out **test set**
and compare error on each, using **root mean square error (RMSE)**:
$$ \mathrm{RMSE} = \sqrt{\tfrac1N\lVert\mathbf y-\boldsymbol\Phi\boldsymbol\theta\rVert^2} = \sqrt{\tfrac1N\sum_{n=1}^N\big(y_n-\boldsymbol\phi^\top(\mathbf x_n)\boldsymbol\theta\big)^2}. $$
RMSE is preferred over the raw squared error because it (a) is comparable across dataset sizes and
(b) carries the same units as $y$ (fit house prices in EUR → RMSE in EUR).

**The signature of overfitting** (Fig. 9.6): as $M$ increases,
- **training error monotonically decreases** (never goes up — more flexibility always fits training data at least as well),
- **test error first decreases, then increases** sharply.

The sweet spot is the $M$ minimizing *test* error (in the book's example, $M=4$). At $M=N-1$ the
polynomial passes through every training point exactly but oscillates wildly between them.

**Fixes.** Reduce flexibility (lower $M$ — model selection, §8.6), or keep flexibility but
**regularize**: add a penalty on parameter size, giving [[regularized-least-squares|ridge regression]] /
[[maximum-a-posteriori-estimation|MAP]], or go fully [[bayesian-linear-regression|Bayesian]] and report
uncertainty instead of a single overfit curve.

## 🔴 Theory (Expert)
- **Why magnitudes blow up.** Overfitting in linear regression is signalled by *large parameter
  magnitudes* (Bishop, 2006): to make a polynomial thread every noisy point, coefficients take huge
  positive/negative values that nearly cancel. This is precisely why an $\ell_2$ penalty on
  $\boldsymbol\theta$ (a Gaussian prior) suppresses it — see [[regularized-least-squares]].
- **Degrees of freedom & invertibility.** For $M<N$ the ML estimator is unique. At $M\ge N$ there are
  more parameters than data points: $\boldsymbol\Phi^\top\boldsymbol\Phi$ is singular (its null space is
  nontrivial), the [[system-of-linear-equations|normal equations]] are underdetermined, and there are
  infinitely many zero-training-error solutions. Brute-force model selection only needs to scan
  $0\le M\le N-1$.
- **Generalization gap.** Test error = training error + generalization gap; the gap grows with model
  capacity and shrinks with data. This is the bias–variance trade-off: low-$M$ models are biased
  (underfit), high-$M$ models have high variance (overfit). MAP/Bayesian methods trade a little bias
  for a large variance reduction.
- **Not eliminated by MAP.** A prior *pushes back* the overfitting boundary but does not remove it;
  the principled cure is to marginalize over parameters ([[bayesian-linear-regression]]) so uncertainty
  is reported rather than a single confident-but-wrong fit.

## Connections
- Builds on: [[linear-regression]], [[maximum-likelihood-estimation]]
- Leads to: [[regularized-least-squares]], [[maximum-a-posteriori-estimation]], [[bayesian-linear-regression]]
- Used by: model selection, cross-validation, every supervised learner
- People: —

## Sources
- `raw/mml-book.pdf` §9.2.2 (Eq. 9.23, Figs. 9.5–9.6), printed pp. 298–300 (PDF pp. 304–306).
