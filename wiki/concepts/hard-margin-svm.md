---
title: Hard Margin SVM
type: concept
tags: [classification, support-vector-machine, convex-optimization, quadratic-programming]
difficulty: advanced
sources: [mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Hard Margin SVM

> **One-liner:** The hard-margin SVM is the strictest version of the max-margin classifier: every single point must stay outside the buffer zone, no exceptions.

## 🟢 Intuition (Dummy)
This is the "no mistakes allowed" SVM. It draws the widest possible empty street between the two groups
*and demands* that not a single point intrudes into the street or crosses to the wrong side. That works
beautifully when the two groups don't overlap — but if even one point is on the wrong side, no such street
exists and the method fails. That fragility is why the relaxed [[soft-margin-svm]] is used in practice.

## 🟡 Mechanics (Practitioner)
Assume the data is **linearly separable**. Using unit-[[margin]] scaling (closest point at
$y(\langle w,x\rangle+b)=1$), the margin is $1/\|w\|$, so maximizing it means minimizing $\|w\|$. The
**hard margin SVM** is the convex program
$$ \min_{w,b}\ \tfrac12\|w\|^2 \quad\text{subject to}\quad y_n(\langle w,x_n\rangle + b)\ge 1,\quad n=1,\dots,N. $$
The factor $\tfrac12$ and the square are cosmetic (monotone transform, tidy gradient) and do not change the
optimizer. "Hard" means the constraints permit **no violations** of the margin condition.

This is a **convex [[quadratic-programming|quadratic program]]**: a quadratic objective with linear
inequality constraints. It has no analytic solution; solve numerically (or via its
[[dual-support-vector-machine|dual]]). The number of optimization variables ($w,b$) is $D+1$ — it scales
with the **feature dimension**.

**Worked example.** Points $x_1=(0,0),y_1=-1$ and $x_2=(2,0),y_2=+1$. The plane must satisfy
$-(b)\ge1$ and $(2w_1+b)\ge1$. The optimum makes both tight: $b=-1$, $2w_1+b=1\Rightarrow w_1=1$, $w_2=0$.
So $w=(1,0)$, $\|w\|=1$, margin $=1$, boundary at $x^{(1)}=1$ — exactly halfway, as expected.

**ML usage.** The conceptual core of the SVM. Rarely used as-is because real data is seldom perfectly
separable; the [[soft-margin-svm]] generalizes it.

## 🔴 Theory (Expert)
Feasibility of the constraints is *exactly* linear separability; if the data is not separable the feasible
set is empty and the problem is infeasible (its hinge-loss form, §12.2.5, assigns $+\infty$ loss to any
violation — Eq. 12.30). The objective is strictly convex in $w$, so the optimal $w^\*$ is unique (though
$b$ can be non-unique in degenerate cases). By [[lagrange-multipliers|KKT]] the solution is supported by the
active constraints — the points with $y_n f(x_n)=1$ lying exactly on the margin, the
[[support-vector|support vectors]]. The representer theorem then gives $w^\*=\sum_n\alpha_n y_n x_n$. The
hard-margin objective is the $C\to\infty$ limit of the [[soft-margin-svm]] (infinite penalty for any slack).

## Connections
- Builds on: [[margin]], [[separating-hyperplane]], [[quadratic-programming]], [[convex-optimization]]
- Leads to: [[soft-margin-svm]], [[dual-support-vector-machine]], [[support-vector]]
- Used by: [[support-vector-machine]]
- People: [[vladimir-vapnik]], [[joseph-louis-lagrange]]

## Sources
- `raw/mml-book.pdf` §12.2.2 (Eq. 12.18–12.19), printed pp. 376–377 (PDF pp. 382–383).
