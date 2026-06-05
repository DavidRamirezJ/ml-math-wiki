---
title: Orthogonal Complement
type: concept
tags: [analytic-geometry, linear-algebra, inner-product]
difficulty: intermediate
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Orthogonal Complement

> **One-liner:** The orthogonal complement of a subspace is everything that is perpendicular to all of it — together they fill the whole space without overlap.

## 🟢 Intuition (Dummy)
Picture a flat tabletop (a plane) inside a room (3D space). The single direction that points
straight up from the table — perpendicular to every direction lying flat on it — captures
"everything orthogonal to the plane." That upward line is the plane's orthogonal complement. Any
point in the room can be split uniquely into "how far across the table" plus "how far above it."

## 🟡 Mechanics (Practitioner)
Let $U\subseteq V$ be an $M$-dimensional subspace of a $D$-dimensional [[inner-product]] space. Its
**orthogonal complement** $U^\perp$ is the set of all vectors orthogonal to every vector in $U$:
$$ U^\perp = \{\, v\in V : \langle v,u\rangle = 0 \text{ for all } u\in U \,\}. $$

Key facts:
- $U^\perp$ is a subspace of dimension $D - M$.
- $U \cap U^\perp = \{\mathbf 0\}$, and **every** $x\in V$ decomposes **uniquely** as
$$ x = \sum_{m=1}^{M}\lambda_m b_m + \sum_{j=1}^{D-M}\psi_j b_j^\perp, $$
where $(b_1,\dots,b_M)$ is a basis of $U$ and $(b_1^\perp,\dots,b_{D-M}^\perp)$ a basis of $U^\perp$.

**Worked example (plane in 3D).** A plane $U\subseteq\mathbb R^3$ has a 1-dimensional complement
spanned by a unit **normal vector** $w$ ($\|w\|=1$) perpendicular to the plane. Every vector
orthogonal to $w$ lies in $U$ (Fig. 3.7).

**ML usage.** Orthogonal complements underlie geometric **dimensionality reduction**: PCA keeps the
projection onto an informative subspace $U$ and discards the complement $U^\perp$ (the "lost"
directions). The residual of an [[orthogonal-projection]] lives in $U^\perp$.

## 🔴 Theory (Expert)
- $V = U \oplus U^\perp$ (orthogonal direct sum); equivalently $\dim U + \dim U^\perp = \dim V$.
- $(U^\perp)^\perp = U$ in finite dimensions — the complement of the complement is the original.
- Orthogonal complements describe **hyperplanes** in $n$-dimensional vector and [[affine-space|affine spaces]]:
  a hyperplane is the set orthogonal to a single normal vector, central to the separating
  hyperplane of SVMs (§12.1).
- The decomposition is exactly the orthogonal-projection split $x = \pi_U(x) + \pi_{U^\perp}(x)$,
  with the two projection matrices summing to the identity: $P_U + P_{U^\perp} = I$.

## Connections
- Builds on: [[orthogonality]], [[vector-space]] (subspaces), [[basis-and-rank]]
- Leads to: [[orthogonal-projection]], separating hyperplanes (Ch. 12), dimensionality reduction (Ch. 10)
- Used by: PCA, SVM
- People: —

## Sources
- `raw/mml-book.pdf` §3.6 (Eq 3.36, Fig. 3.7), printed pp. 79–80 (PDF pp. 84–85).
