---
title: Loss Function
type: concept
tags: [machine-learning, optimization, supervised-learning]
difficulty: beginner
sources: [mml-ch08-when-models-meet-data]
created: 2026-06-04
updated: 2026-06-04
---

# Loss Function

> **One-liner:** A loss function is the rule that turns "how wrong was this one prediction?" into a single non-negative number you can try to make small.

## 🟢 Intuition (Dummy)
You guessed a friend's height as 170 cm; they're actually 180 cm. How bad is that guess? A loss
function answers with one number — say "10" if you score by the raw gap, or "100" if you score by
the gap *squared* (which punishes big misses much harder). Zero loss means a perfect guess. The
choice of how you score the gap quietly shapes what the learner cares about: square it and a few
huge errors dominate; use the absolute gap and every error counts proportionally.

## 🟡 Mechanics (Practitioner)
A **loss function** $\ell(y_n,\hat y_n)\ge 0$ takes the ground-truth label $y_n$ and the prediction
$\hat y_n$ and returns the cost of that single prediction (MML §8.2.2). It is the inner ingredient of
the empirical risk averaged over the data ([[empirical-risk-minimization]], Eq. 8.6):
$$ R_{\text{emp}}=\frac1N\sum_{n=1}^N \ell(y_n,\hat y_n). $$

**The squared loss (Example 8.2):**
$$ \ell(y_n,\hat y_n)=(y_n-\hat y_n)^2 . $$
With a linear predictor $\hat y_n=\boldsymbol\theta^\top\mathbf x_n$ it produces the
**least-squares problem** $\min_{\boldsymbol\theta}\frac1N\lVert\mathbf y-X\boldsymbol\theta\rVert^2$.

**Worked example.** Labels $\mathbf y=(2,4,6)$, predictions $\hat{\mathbf y}=(1,4,8)$. Per-example
squared losses are $(2-1)^2=1$, $(4-4)^2=0$, $(6-8)^2=4$. Empirical risk $=\frac{1+0+4}{3}=\tfrac53\approx1.67$.

**ML usage.** Common losses by task:
| Task | Typical loss | Note |
|---|---|---|
| Regression | squared $(y-\hat y)^2$, absolute $\lvert y-\hat y\rvert$ | squared ⇒ least squares / [[maximum-likelihood-estimation|Gaussian MLE]] |
| Classification | cross-entropy, hinge | hinge ⇒ [[support-vector-machine]] margin |
| Counts | (negative log) Poisson likelihood | a generalized linear model |

Ideally the loss equals the task's **performance measure**; in practice they differ for ease of
optimization (a smooth surrogate replaces a non-differentiable metric like accuracy).

## 🔴 Theory (Expert)
- Every loss has a **probabilistic alter ego**: minimizing average loss is equivalent to maximizing
  a likelihood, with $\ell(y,\hat y)=-\log p(y\mid\hat y)$ up to constants. Squared loss
  $\Leftrightarrow$ Gaussian likelihood (see [[maximum-likelihood-estimation]], Example 8.5); hinge
  has no exact likelihood (it bounds 0–1 loss). This is the **loss $\leftrightarrow$ negative
  log-likelihood** correspondence the chapter builds the whole §8.2–§8.3 parallel on.
- The **0–1 loss** (1 if wrong, 0 if right) is the "honest" classification objective but is
  non-convex and non-differentiable; convex **surrogate losses** (hinge, logistic, exponential) are
  minimized instead, with consistency guarantees under conditions (classification calibration).
- Choosing the loss is choosing the noise model: squared loss assumes symmetric Gaussian noise, so
  it is sensitive to outliers; heavier-tailed assumptions (Laplace ⇒ absolute loss) are more robust.

## Connections
- Builds on: [[vector]]
- Leads to: [[empirical-risk-minimization]], [[regularization]]
- Used by: [[linear-regression]], [[support-vector-machine]]
- Parallels: likelihood in [[maximum-likelihood-estimation]]
- People: [[carl-friedrich-gauss]] (least squares)

## Sources
- `raw/mml-book.pdf` §8.2.2 (Eq. 8.6) & Example 8.2 (Eqs. 8.7–8.9), printed pp. 260–261 (PDF pp. 266–267).
