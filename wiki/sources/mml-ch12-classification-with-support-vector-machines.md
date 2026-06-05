---
title: "MML Ch. 12 — Classification with Support Vector Machines"
type: source
tags: [classification, support-vector-machine, kernels, convex-optimization, machine-learning]
sources: []
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch. 12 — Classification with Support Vector Machines

- **Raw file:** `raw/mml-book.pdf` Chapter 12, printed pp. 370–394 (PDF pp. 376–400).
- **Type:** book chapter
- **Ingested:** 2026-06-04

The fourth and final application chapter of Part II. Where Ch. 9 (regression) attacked supervised
learning **probabilistically** (maximum likelihood), this chapter solves **binary classification**
$f:\mathbb R^D\to\{+1,-1\}$ **geometrically**: design a loss/objective from geometric intuition, then
optimize it. The headline method is the **support vector machine (SVM)** — find the separating
[[separating-hyperplane|hyperplane]] with the largest [[margin]]. The chapter develops three equivalent
views (geometric margin, hinge-loss, convex-hull), the Lagrangian **dual**, and the **kernel trick**
for nonlinear classification.

## Section coverage

| § | Title | Wiki page(s) |
|---|---|---|
| 12.0 | Intro: binary classification, $\{+1,-1\}$ labels | [[support-vector-machine]] |
| 12.1 | Separating Hyperplanes | [[separating-hyperplane]] |
| 12.2 | Primal Support Vector Machine | [[support-vector-machine]] |
| 12.2.1 | Concept of the Margin | [[margin]] |
| 12.2.2 | Traditional Derivation of the Margin | [[margin]], [[hard-margin-svm]] |
| 12.2.3 | Why We Can Set the Margin to 1 (Thm 12.1) | [[hard-margin-svm]] |
| 12.2.4 | Soft Margin SVM: Geometric View (slack) | [[soft-margin-svm]] |
| 12.2.5 | Soft Margin SVM: Loss Function View (hinge) | [[soft-margin-svm]], [[hinge-loss]] |
| 12.3 | Dual Support Vector Machine | [[dual-support-vector-machine]] |
| 12.3.1 | Convex Duality via Lagrange Multipliers | [[dual-support-vector-machine]], [[support-vector]] |
| 12.3.2 | Dual SVM: Convex Hull View | [[dual-support-vector-machine]] (convex-hull view) |
| 12.4 | Kernels | [[kernel-trick]] |
| 12.5 | Numerical Solution | [[soft-margin-svm]] (subgradient), [[dual-support-vector-machine]] (QP) |
| 12.6 | Further Reading | — |

## TL;DR (5 bullets)

- A **[[separating-hyperplane]]** $\{x:\langle w,x\rangle+b=0\}$ splits $\mathbb R^D$ into a positive and a
  negative half-space; classify by the sign of $\langle w,x\rangle+b$. $w$ is the **normal vector** to the plane.
- When data is linearly separable there are infinitely many separating hyperplanes; the SVM picks the unique
  one that **maximizes the [[margin]]** (the distance to the closest example), which provably generalizes well.
- Scaling so the closest point satisfies $y(\langle w,x\rangle+b)=1$, the **[[hard-margin-svm]]** is
  $\min_{w,b}\tfrac12\|w\|^2$ s.t. $y_n(\langle w,x_n\rangle+b)\ge 1$ — a convex quadratic program.
- The **[[soft-margin-svm]]** adds slack $\xi_n\ge0$ (penalty $C\sum_n\xi_n$) for points inside/over the margin;
  equivalently it minimizes the **[[hinge-loss]]** plus $\ell_2$-regularization — margin maximization *is* regularization.
- The Lagrangian **[[dual-support-vector-machine|dual]]** depends only on **inner products** $\langle x_i,x_j\rangle$
  between examples; this unlocks the **[[kernel-trick]]** $k(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle$ for
  nonlinear decision boundaries, and shows the solution is a sum over the few **[[support-vector|support vectors]]**.

## Key concepts introduced

- [[support-vector-machine]] — the binary-classification method and its overall geometric framing.
- [[separating-hyperplane]] — affine decision surface $\langle w,x\rangle+b=0$ with normal $w$.
- [[margin]] — distance from the hyperplane to the nearest example; equals $1/\|w\|$ under unit-margin scaling.
- [[hard-margin-svm]] — max-margin primal for linearly separable data: $\min\tfrac12\|w\|^2$.
- [[soft-margin-svm]] — the practical $C$-SVM with slack variables / hinge loss for non-separable data.
- [[hinge-loss]] — $\ell(t)=\max\{0,1-t\}$, the convex surrogate for zero-one loss that yields the SVM.
- [[dual-support-vector-machine]] — the Lagrangian dual QP in $\alpha\in\mathbb R^N$ with box constraints; convex-hull view.
- [[support-vector]] — examples with $\alpha_n>0$ that determine the hyperplane (via the representer theorem).
- [[kernel-trick]] — replace $\langle x_i,x_j\rangle$ by a kernel $k$ to classify in an implicit feature space.

## Notable results / theorems

- **Hyperplane normal (Eq. 12.4):** for $x_a,x_b$ on $\{f=0\}$, $\langle w, x_a-x_b\rangle=0$, so $w\perp$ the hyperplane.
- **Margin = $1/\|w\|$ (Eq. 12.14):** scaling the data so the closest example has $\langle w,x_a\rangle+b=1$ gives distance $r=1/\|w\|$.
- **Theorem 12.1 (equivalence):** maximizing the margin $r$ with $\|w\|=1$ is equivalent to $\min_{w,b}\tfrac12\|w\|^2$ s.t. $y_n(\langle w,x_n\rangle+b)\ge1$ — the two derivations of the hard-margin SVM coincide.
- **Hinge loss is a convex upper bound** of the zero-one loss (Fig. 12.8); the constrained reformulation (12.33) recovers the slack-variable soft-margin SVM exactly.
- **Representer theorem (Eq. 12.38, Kimeldorf & Wahba 1970):** $w^\*=\sum_n \alpha_n y_n x_n$ — the optimal weight is a linear combination of the training examples; only [[support-vector|support vectors]] ($\alpha_n>0$) contribute.
- **Dual SVM (Eq. 12.41):** $\min_\alpha \tfrac12\sum_{i,j} y_iy_j\alpha_i\alpha_j\langle x_i,x_j\rangle - \sum_i\alpha_i$ s.t. $\sum_i y_i\alpha_i=0,\ 0\le\alpha_i\le C$ — a convex QP with **box constraints**; $0<\alpha_i<C$ ⇔ point lies on the margin (KKT).
- **Kernel matrix (Gram matrix)** $K_{ij}=k(x_i,x_j)$ is symmetric **positive semidefinite** (Eq. 12.53); every kernel has a unique reproducing-kernel Hilbert space.

## What this changes in the wiki

- Introduces the **geometric / max-margin** style of supervised learning, complementing the probabilistic
  (maximum-likelihood, Bayesian) thread from Chs. 9–11; both are instances of [[empirical-risk-minimization]].
- Ties together earlier machinery into one worked method: [[inner-product]] & [[orthogonal-projection]] (Ch. 3,
  margin geometry), [[lagrange-multipliers]] & [[convex-optimization]] (Ch. 7, the dual), [[quadratic-programming]]
  (Ch. 7, the solver), [[regularization]] & [[loss-function]] (Ch. 8). Establishes the **kernel trick** as a
  reusable idea reaching beyond SVMs.
- Adds [[vladimir-vapnik]] (margin / statistical learning theory) to the people pages; reuses [[joseph-louis-lagrange]].
- No contradictions with existing pages.
