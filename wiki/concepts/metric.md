---
title: Metric (Distance)
type: concept
tags: [analytic-geometry, inner-product, foundations]
difficulty: beginner
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Metric (Distance)

> **One-liner:** A metric is a rule for measuring how far apart two points are, obeying the common-sense laws of distance.

## 🟢 Intuition (Dummy)
A metric answers "how far is A from B?" with a single non-negative number. It must obey three
obvious rules: the distance is zero only when the two points are the same; the distance from A to
B equals the distance from B to A; and a detour through C is never shorter than going straight
(the triangle inequality). Distance and similarity move in *opposite* directions — two very
similar vectors are *close* (small distance) but have a *large* inner product.

## 🟡 Mechanics (Practitioner)
**Definition 3.6.** In an [[inner-product]] space $(V,\langle\cdot,\cdot\rangle)$, the **distance**
between $x,y\in V$ is
$$ d(x,y) := \|x-y\| = \sqrt{\langle x-y,\ x-y\rangle}. $$
With the dot product this is the **Euclidean distance**. The mapping $d:V\times V\to\mathbb R$ is
called a **metric** and satisfies:
1. **Positive definite:** $d(x,y)\ge 0$, and $d(x,y)=0 \iff x=y$
2. **Symmetric:** $d(x,y)=d(y,x)$
3. **Triangle inequality:** $d(x,z) \le d(x,y) + d(y,z)$

**Worked example.** For $x=\big[1,2,3\big]^\top$, $y=\big[-1,-1,0\big]^\top$ with the dot product:
$x-y = \big[2,3,3\big]^\top$, so $d(x,y) = \sqrt{4+9+9} = \sqrt{22}$.

**ML usage.** Distances drive $k$-nearest-neighbors, $k$-means clustering, and the projection/
reconstruction error minimized by PCA and least-squares regression.

## 🔴 Theory (Expert)
- A **norm suffices** to define a distance ($d(x,y)=\|x-y\|$); an inner product is not required.
  When the norm comes from an inner product, the distance depends on the *choice* of inner product.
- The Manhattan metric $d_1(x,y)=\|x-y\|_1$ is a valid metric not induced by any inner product.
- Note the contrast with [[inner-product]] $\langle x,y\rangle$: similarity grows as vectors align,
  while the metric $d(x,y)$ shrinks — they "behave in opposite directions" (Remark, p. 76).
- General metric spaces drop the vector-space structure entirely; the three axioms above are all
  that is needed for analysis (convergence, continuity, completeness).

## Connections
- Builds on: [[norm]], [[inner-product]]
- Leads to: [[orthogonal-projection]] (minimizing distance), clustering, nearest-neighbor methods
- Used by: $k$-means, $k$-NN, PCA reconstruction error
- People: —

## Sources
- `raw/mml-book.pdf` §3.3 (Definition 3.6), printed pp. 75–76 (PDF pp. 80–81).
