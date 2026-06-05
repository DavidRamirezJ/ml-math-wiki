---
title: Soft Margin SVM
type: concept
tags: [classification, support-vector-machine, regularization, convex-optimization]
difficulty: advanced
sources: [mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Soft Margin SVM

> **One-liner:** The soft-margin SVM is the practical SVM that still wants a wide buffer zone but lets a few points sneak inside or onto the wrong side, paying a price for each.

## 🟢 Intuition (Dummy)
Real data is messy — two groups usually overlap a bit, so no perfectly clean dividing street exists. The
soft-margin SVM is forgiving: it draws a wide street and *allows* some points to intrude, but charges a
fine for every intruder. A knob called $C$ sets how steep the fine is. Big $C$ = strict (few intruders,
narrow street); small $C$ = lenient (wider street, more intruders tolerated). You tune $C$ to balance a
roomy buffer against making too many mistakes.

## 🟡 Mechanics (Practitioner)
Introduce a **slack variable** $\xi_n\ge0$ per example that measures how far it intrudes past its margin.
The **geometric (constrained) form** is
$$ \min_{w,b,\xi}\ \tfrac12\|w\|^2 + C\sum_{n=1}^N \xi_n \quad\text{s.t.}\quad y_n(\langle w,x_n\rangle+b)\ge 1-\xi_n,\ \ \xi_n\ge0. $$
- $\xi_n=0$: point is correctly outside the margin.
- $0<\xi_n<1$: point is inside the margin but correctly classified.
- $\xi_n>1$: point is on the wrong side (misclassified).

$C>0$ is the **[[regularization|regularization]] parameter** (hence "$C$-SVM"). **Large $C$ ⇒ low
regularization** (slack is expensive, prioritize fitting); small $C$ ⇒ strong regularization (wide margin).
Note $b$ is *not* regularized.

**Equivalent loss-function (unconstrained) form.** Eliminating the slack gives the
[[empirical-risk-minimization|regularized empirical risk]]
$$ \min_{w,b}\ \underbrace{\tfrac12\|w\|^2}_{\text{regularizer}} + C\sum_{n=1}^N \underbrace{\max\{0,\,1-y_n(\langle w,x_n\rangle+b)\}}_{\text{[[hinge-loss|hinge loss]] / error term}}. $$
The two forms are identical because $\min_t\max\{0,1-t\}$ equals $\min_{\xi}\xi$ s.t. $\xi\ge0,\ \xi\ge1-t$.
**Margin maximization $\equiv$ regularization.**

**ML usage.** The default SVM in libraries (LIBSVM, scikit-learn `SVC`/`LinearSVC`). Choose $C$ (and any
kernel parameters) by nested [[cross-validation]].

## 🔴 Theory (Expert)
The constrained form is a convex [[quadratic-programming|QP]] in $D+1+N$ variables (primal). Its
[[dual-support-vector-machine|dual]] differs from the hard-margin dual only by the upper **box constraint**
$\alpha_n\le C$ (geometrically, the convex hulls shrink to **reduced hulls**). KKT classification of points:
$\alpha_n=0$ (outside margin, irrelevant), $0<\alpha_n<C$ (exactly on the margin), $\alpha_n=C$ (inside
margin / misclassified, $\xi_n>0$).

**Numerical solution.** The loss form is convex but the hinge is non-differentiable at $t=1$, so use a
**subgradient** (Eq. 12.54): $g(t)=-1$ for $t<1$, $0$ for $t>1$, and any value in $[-1,0]$ at $t=1$ — then
apply (stochastic) [[stochastic-gradient-descent|subgradient descent]]. Alternatively cast the constrained
primal or dual as a standard-form QP (Eqs. 12.56–12.57) and call a QP solver. The hard-margin SVM is the
$C\to\infty$ limit (hinge replaced by the $\{0,\infty\}$ loss of Eq. 12.30).

## Connections
- Builds on: [[hard-margin-svm]], [[hinge-loss]], [[regularization]], [[empirical-risk-minimization]], [[quadratic-programming]]
- Leads to: [[dual-support-vector-machine]], [[kernel-trick]]
- Used by: [[support-vector-machine]], practical classification
- People: [[vladimir-vapnik]]

## Sources
- `raw/mml-book.pdf` §12.2.4–12.2.5 (Eqs. 12.26, 12.31), §12.5 (subgradient, QP), printed pp. 379–392 (PDF pp. 385–398).
