---
title: Support Vector
type: concept
tags: [classification, support-vector-machine, sparsity, representer-theorem]
difficulty: intermediate
sources: [mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Support Vector

> **One-liner:** Support vectors are the handful of training points sitting right on the edge of the buffer zone — they alone determine the dividing line; the rest could be deleted.

## 🟢 Intuition (Dummy)
After an SVM finds its dividing street, ask: which marbles actually pinned the street in place? Only the
few touching the curbs (or that strayed into the street). Move any of the far-away marbles and the street
doesn't budge. Those critical, line-supporting marbles are the **support vectors** — they "hold up" the
hyperplane, which is exactly where the name *support vector machine* comes from.

## 🟡 Mechanics (Practitioner)
From the [[dual-support-vector-machine|dual SVM]], each example $x_n$ has a dual weight $\alpha_n\ge0$, and
the optimal hyperplane normal is the **representer theorem** combination
$$ w^\* = \sum_{n=1}^N \alpha_n\, y_n\, x_n. $$
Examples with $\alpha_n=0$ contribute **nothing** to $w^\*$. The examples with $\alpha_n>0$ are the
**support vectors**. By the KKT conditions:
- $\alpha_n=0$ — outside the margin, irrelevant.
- $0<\alpha_n<C$ — lies **exactly on** the margin boundary ($y_n f(x_n)=1$).
- $\alpha_n=C$ — inside the margin or misclassified (a slack-using point, $\xi_n>0$).

This makes the SVM **sparse**: prediction $f(x)=\sum_{n\in SV}\alpha_n y_n\langle x_n,x\rangle + b$ sums only
over support vectors. The bias is read off any on-margin support vector: $b^\*=y_n-\langle w^\*,x_n\rangle$.

**ML usage.** Fewer support vectors ⇒ a more compact, faster model; the support-vector count is also a rough
indicator of problem difficulty / overfitting risk.

## 🔴 Theory (Expert)
The **representer theorem** (Kimeldorf & Wahba, 1970; generalized by Schölkopf et al., 2001) states that the
minimizer of a regularized empirical risk lies in the span of the training examples — here in the
[[vector-space|subspace]] spanned by $\{x_n\}$ — so the solution is fully described by the dual coefficients.
Sparsity is a consequence of the [[hinge-loss|hinge loss]]'s flat region: points with $y_n f(x_n)>1$ incur
zero loss and zero gradient, hence $\alpha_n=0$. With a [[kernel-trick|kernel]] the same identity holds with
$x_n$ replaced by feature maps $\phi(x_n)$, and prediction uses $k(x_n,x)$ — so even in an infinite-dimensional
feature space the model is a finite sum over support vectors. This is what keeps kernel SVMs computable.

## Connections
- Builds on: [[dual-support-vector-machine]], [[lagrange-multipliers]], [[hinge-loss]]
- Leads to: [[kernel-trick]], sparse kernel machines
- Used by: [[support-vector-machine]]
- People: [[vladimir-vapnik]]

## Sources
- `raw/mml-book.pdf` §12.3.1 (Eq. 12.38, representer theorem; "support vector" remark), printed pp. 384–385 (PDF pp. 390–391).
