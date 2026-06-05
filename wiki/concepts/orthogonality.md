---
title: Orthogonality
type: concept
tags: [analytic-geometry, inner-product, foundations]
difficulty: beginner
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Orthogonality

> **One-liner:** Two vectors are orthogonal when their inner product is zero — the precise, general meaning of "at right angles."

## 🟢 Intuition (Dummy)
Two arrows are orthogonal when they meet at a right angle — neither one has any "shadow" along the
other. If you push something purely north, it makes no progress east; north and east are
orthogonal directions. The clean test is: their inner product is exactly zero. Orthogonal
directions are independent in the strongest geometric sense — they share nothing.

## 🟡 Mechanics (Practitioner)
**Definition 3.7.** Vectors $x$ and $y$ are **orthogonal** iff
$$ \langle x,y\rangle = 0, \qquad\text{written } x\perp y. $$
If additionally $\|x\|=\|y\|=1$ (both are unit vectors), they are **orthonormal**.

- The zero vector is orthogonal to every vector.
- Orthogonality is **defined relative to the inner product**: $x=\big[1,1\big]^\top$ and
  $y=\big[-1,1\big]^\top$ are orthogonal under the dot product ($90^\circ$), but *not* under the
  inner product with $A=\big[\begin{smallmatrix}2&0\\0&1\end{smallmatrix}\big]$ (about $109.5^\circ$).

**Worked example.** $x=\big[1,1\big]^\top$, $y=\big[-1,1\big]^\top$: $x^\top y = -1+1 = 0$, so
$x\perp y$ under the dot product. Normalizing each by $\frac1{\sqrt2}$ makes them orthonormal.

**ML usage.** Orthogonal features carry independent information (no redundancy) — the goal of PCA's
principal axes. Orthonormal bases simplify projections (no matrix inverse needed) and make
computations numerically stable.

## 🔴 Theory (Expert)
- Orthogonality generalizes perpendicularity to any symmetric bilinear form, not just the dot
  product (Remark, p. 77).
- An **[[orthogonal-matrix]]** $A$ has orthonormal columns, $A^\top A = AA^\top = I$, hence
  $A^{-1}=A^\top$; such transformations preserve lengths and angles:
  $\|Ax\|^2 = x^\top A^\top A x = x^\top x = \|x\|^2$. These are the [[rotation|rotations]]
  (and reflections).
- A whole subspace can be orthogonal to another: the **[[orthogonal-complement]]** $U^\perp$
  collects all vectors orthogonal to every vector in $U$.
- Pairwise orthogonal nonzero vectors are automatically linearly independent
  ([[linear-independence]]) — a key fact behind orthonormal bases.

## Connections
- Builds on: [[inner-product]], [[angle-between-vectors]]
- Leads to: [[orthonormal-basis]], [[orthogonal-complement]], [[orthogonal-matrix]], [[orthogonal-projection]]
- Used by: PCA, Gram–Schmidt, QR decomposition, SVM margins
- People: —

## Sources
- `raw/mml-book.pdf` §3.4 (Definition 3.7, Example 3.7), printed pp. 76–78 (PDF pp. 81–83).
