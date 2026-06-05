---
title: Hinge Loss
type: concept
tags: [classification, loss-function, support-vector-machine, convex-optimization]
difficulty: intermediate
sources: [mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Hinge Loss

> **One-liner:** The hinge loss charges nothing once a point is comfortably on the correct side of the line, and a growing fine the closer it gets to (or crosses) the line.

## 🟢 Intuition (Dummy)
Think of grading a classifier on each point. Counting mistakes (right/wrong) seems natural, but a pure
mistake-counter is impossible to optimize smoothly — nudging the line changes nothing until a point
suddenly flips. The hinge loss is a smarter grade: it's zero if a point is correct *and* safely past the
buffer; otherwise the penalty rises in a straight line the deeper the point is into the danger zone. This
gives the optimizer a gentle slope to follow downhill, and it looks like a door hinge — flat, then sloping
up — hence the name.

## 🟡 Mechanics (Practitioner)
Let $t = y\,f(x) = y(\langle w,x\rangle+b)$ be the **margin score** (positive when correct, larger when
more confident). The **hinge loss** is
$$ \ell(t) = \max\{0,\ 1 - t\} \;=\; \begin{cases} 0 & t\ge 1\\ 1-t & t<1.\end{cases} $$
- $t\ge1$: correct and outside the margin → no loss.
- $0<t<1$: correct but inside the margin → small positive loss.
- $t\le0$: wrong side → loss $\ge1$, growing linearly.

Summing it with an $\ell_2$ [[regularization|regularizer]] gives the [[soft-margin-svm|soft-margin SVM]]:
$$ \min_{w,b}\ \tfrac12\|w\|^2 + C\sum_{n=1}^N \max\{0,\,1-y_n(\langle w,x_n\rangle+b)\}. $$

**Subgradient (for optimization).** The hinge is differentiable everywhere except the kink at $t=1$:
$$ g(t) = \begin{cases} -1 & t<1\\ [-1,0] & t=1\\ 0 & t>1.\end{cases} $$
Use this with subgradient / [[stochastic-gradient-descent|SGD]].

**ML usage.** The defining loss of the SVM; also used in many margin-based classifiers and in some deep
nets. Compare to logistic loss (smooth, gives probabilities) and squared loss (wrong for classification).

## 🔴 Theory (Expert)
The hinge loss is a **convex upper bound on the zero-one loss** $\mathbf 1(yf(x)<0)$ (Fig. 12.8) — minimizing
it is a tractable convex surrogate for the intractable combinatorial zero-one objective. It is the tightest
*piecewise-linear* convex surrogate that is an upper bound and is zero for $t\ge1$. Three equivalent
representations are used: the max-form $\max\{0,1-t\}$ (good for comparing losses), the two-piece form
(good for subgradients), and the **constrained slack form**
$$ \min_t \max\{0,1-t\} \;\equiv\; \min_{\xi,t}\ \xi \ \text{ s.t. } \xi\ge0,\ \xi\ge1-t, $$
which converts the unconstrained SVM into the constrained QP (and recovers the slack-variable SVM). The
hard-margin loss is the limiting $\{0,\infty\}$ loss (Eq. 12.30). Statistically, the hinge loss is
**classification-calibrated** but, unlike the logistic loss, does *not* yield a consistent estimate of the
posterior probability $P(Y=1\mid x)$ — the SVM is non-probabilistic by design.

## Connections
- Builds on: [[loss-function]], [[empirical-risk-minimization]], [[convex-optimization]]
- Leads to: [[soft-margin-svm]], [[support-vector-machine]]
- Used by: SVM, margin classifiers, subgradient methods
- People: [[vladimir-vapnik]]

## Sources
- `raw/mml-book.pdf` §12.2.5 (Eqs. 12.28–12.33), §12.5 (Eq. 12.54), printed pp. 380–382, 390–391 (PDF pp. 386–388, 396–397).
