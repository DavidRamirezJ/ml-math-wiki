---
title: Norm
type: concept
tags: [analytic-geometry, inner-product, foundations]
difficulty: beginner
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Norm

> **One-liner:** A norm is a rule that measures how "long" a vector is — a single non-negative number standing in for its size.

## 🟢 Intuition (Dummy)
Think of a vector as an arrow starting at the origin. Its *length* is how far the tip is from the
start. A norm is just the recipe for that length. The familiar recipe is the straight-line distance
(Pythagoras). But there are other sensible recipes: if you can only walk along city blocks
(no diagonals), the "distance" is how far you walk along the streets — that's the Manhattan norm.
Different recipes, all called norms, share three common-sense rules: scaling an arrow scales its
length, the direct route is never longer than a detour, and only the zero arrow has length zero.

## 🟡 Mechanics (Practitioner)
**Definition 3.1.** A **norm** on a vector space $V$ is a function $\|\cdot\| : V \to \mathbb R$,
$x \mapsto \|x\|$, satisfying for all $\lambda\in\mathbb R$ and $x,y\in V$:

- **Absolutely homogeneous:** $\|\lambda x\| = |\lambda|\,\|x\|$
- **Triangle inequality:** $\|x+y\| \le \|x\| + \|y\|$
- **Positive definite:** $\|x\| \ge 0$, and $\|x\| = 0 \iff x = \mathbf 0$

**Two common norms on $\mathbb R^n$:**
$$ \|x\|_1 := \sum_{i=1}^n |x_i| \quad(\text{Manhattan / }\ell_1), \qquad
   \|x\|_2 := \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x^\top x} \quad(\text{Euclidean / }\ell_2). $$

The Euclidean norm is the default throughout MML. The set of unit-norm vectors ($\|x\|=1$) is a
diamond for $\ell_1$ and a circle for $\ell_2$ (Fig. 3.3).

**Worked example.** For $x = \big[3, 4\big]^\top$: $\|x\|_1 = |3|+|4| = 7$ and
$\|x\|_2 = \sqrt{9+16} = 5$.

**ML usage.** Norms measure error magnitudes and regularize models: $\ell_2$ ("ridge")
penalizes large weights; $\ell_1$ ("lasso") drives weights to exactly zero, giving sparsity.
Loss functions like mean-squared error are squared $\ell_2$ norms of residuals.

## 🔴 Theory (Expert)
- A norm always **induces a metric** $d(x,y) = \|x-y\|$ (see [[metric]]) and the unit ball
  $\{x : \|x\|\le 1\}$ is convex (a direct consequence of the triangle inequality and homogeneity).
- Every **[[inner-product]] induces a norm** $\|x\| := \sqrt{\langle x,x\rangle}$, but not conversely:
  the Manhattan norm is **not** induced by any inner product (it fails the parallelogram law
  $\|x+y\|^2 + \|x-y\|^2 = 2\|x\|^2 + 2\|y\|^2$, which holds iff a norm comes from an inner product).
- The family $\|x\|_p = \big(\sum_i |x_i|^p\big)^{1/p}$ for $p\ge 1$ generalizes these; $p=\infty$
  gives the max norm $\max_i |x_i|$. On finite-dimensional spaces all norms are **equivalent**
  (each is bounded by constant multiples of any other), so they induce the same topology — though
  not the same geometry or the same optimal solutions.

## Connections
- Builds on: [[vector]], [[vector-space]]
- Leads to: [[metric]], [[inner-product]], [[angle-between-vectors]]
- Used by: [[orthogonal-projection]] (projection error), regularization, loss functions
- People: —

## Sources
- `raw/mml-book.pdf` §3.1 (Definition 3.1, Examples 3.1–3.2) and §3.3, printed pp. 71–72, 75 (PDF pp. 76–77, 80).
