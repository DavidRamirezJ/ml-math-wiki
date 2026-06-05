---
title: Dual Support Vector Machine
type: concept
tags: [classification, support-vector-machine, convex-optimization, duality, quadratic-programming]
difficulty: advanced
sources: [mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Dual Support Vector Machine

> **One-liner:** The dual SVM solves the same problem from the data's point of view — one weight per training example instead of one per feature — and only ever looks at how similar examples are to each other.

## 🟢 Intuition (Dummy)
The original ("primal") SVM has one knob per feature. The **dual** flips the bookkeeping: one knob
$\alpha_n$ per *training example*. Most knobs end up at zero — those examples are irrelevant. The handful
with nonzero knobs are the [[support-vector|support vectors]] that actually prop up the dividing line.
Crucially, the dual only needs to know how similar each pair of examples is (their inner product), never
the raw feature values — which is the secret door to the [[kernel-trick]].

## 🟡 Mechanics (Practitioner)
Form the Lagrangian of the [[soft-margin-svm|soft-margin]] primal with multipliers $\alpha_n\ge0$
(for the margin constraints) and $\gamma_n\ge0$ (for $\xi_n\ge0$). Setting derivatives to zero gives the
stationarity conditions
$$ w=\sum_{n=1}^N \alpha_n y_n x_n \quad(\text{representer theorem}),\qquad \sum_{n=1}^N \alpha_n y_n=0,\qquad C-\alpha_n-\gamma_n=0. $$
Substituting back eliminates $w,b,\xi$ and yields the **dual SVM** — a convex
[[quadratic-programming|QP]] purely in $\alpha$:
$$ \min_{\alpha}\ \tfrac12\sum_{i=1}^N\sum_{j=1}^N y_i y_j \alpha_i \alpha_j \langle x_i,x_j\rangle - \sum_{i=1}^N \alpha_i $$
$$ \text{subject to}\quad \sum_{i=1}^N y_i\alpha_i=0,\qquad 0\le \alpha_i\le C\ \ \forall i. $$
The $\alpha_i\le C$ upper bounds are **box constraints** (the $\gamma_n\ge0$ together with $C-\alpha_n-\gamma_n=0$
force $\alpha_n\le C$). In matrix form with $Y=\operatorname{diag}(y)$ and kernel/Gram matrix $K_{ij}=\langle x_i,x_j\rangle$:
$\min_\alpha \tfrac12\,\alpha^\top Y K Y \alpha - \mathbf 1^\top\alpha$.

**Recovering the primal.** $w^\* = \sum_n \alpha_n y_n x_n$; the bias from any margin point ($0<\alpha_n<C$):
$b^\* = y_n - \langle w^\*, x_n\rangle$ (in practice, the median over support vectors).

**ML usage.** Preferred when $N<D$ (number of variables = $N$, not $D$), and **required** to apply kernels —
just swap $\langle x_i,x_j\rangle \to k(x_i,x_j)$.

## 🔴 Theory (Expert)
This is an application of **convex (Lagrangian) duality** ([[lagrange-multipliers]], §7.2): since the primal
is convex with affine constraints, strong duality holds and the dual optimum matches the primal. The
**representer theorem** (Kimeldorf & Wahba, 1970) — $w^\*\in\operatorname{span}\{x_n\}$ — holds far beyond
SVMs, for general regularized empirical risk minimization. **KKT complementary slackness** classifies points
by $\alpha$: $\alpha_n=0$ (outside margin), $0<\alpha_n<C$ (exactly on the margin), $\alpha_n=C$ (margin
violator). 

**Convex-hull view (§12.3.2).** Build the [[convex-optimization|convex hull]] of each class,
$\operatorname{conv}(X)=\{\sum_n\alpha_n x_n:\sum_n\alpha_n=1,\alpha_n\ge0\}$. Pick the closest pair of points
$c$ (positive hull) and $d$ (negative hull); minimizing $\|w\|=\|c-d\|$, i.e. $\min\tfrac12\|c-d\|^2$ with
$\sum_{y_n=+1}\alpha_n=1$ and $\sum_{y_n=-1}\alpha_n=1$ (which implies $\sum_n y_n\alpha_n=0$), reproduces the
hard-margin dual exactly (Bennett & Bredensteiner, 2000). The soft margin corresponds to the **reduced hull**
(coefficients capped, shrinking the hull). Box constraints make the dual particularly efficient for
specialized QP solvers.

## Connections
- Builds on: [[lagrange-multipliers]], [[soft-margin-svm]], [[quadratic-programming]], [[inner-product]], [[convex-optimization]]
- Leads to: [[kernel-trick]], [[support-vector]]
- Used by: [[support-vector-machine]], kernel methods
- People: [[joseph-louis-lagrange]], [[vladimir-vapnik]]

## Sources
- `raw/mml-book.pdf` §12.3 (Eqs. 12.34–12.51), §12.5 (Eq. 12.57), printed pp. 383–388, 391–392 (PDF pp. 389–394, 397–398).
