---
title: Orthonormal Basis
type: concept
tags: [analytic-geometry, linear-algebra, inner-product]
difficulty: intermediate
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Orthonormal Basis

> **One-liner:** An orthonormal basis is a set of coordinate directions that are all mutually perpendicular and exactly one unit long — the cleanest possible coordinate system.

## 🟢 Intuition (Dummy)
A basis is a set of reference directions for describing every vector by coordinates. The nicest
reference directions are mutually perpendicular and each exactly one unit long — like the corner of
a room where the three edges meet at right angles. With such directions, finding a vector's
coordinates is effortless: you just measure its shadow along each direction. No tilted, stretched,
or skewed axes to untangle.

## 🟡 Mechanics (Practitioner)
**Definition 3.9.** In an $n$-dimensional [[inner-product]] space $V$, a basis
$\{b_1,\dots,b_n\}$ is an **orthonormal basis (ONB)** if
$$ \langle b_i,b_j\rangle = 0 \ \ (i\ne j), \qquad \langle b_i,b_i\rangle = 1. $$
If only the first condition holds, it is an **orthogonal basis** (perpendicular but not
unit-length).

**Worked example.** In $\mathbb R^2$ with the dot product,
$$ b_1 = \tfrac1{\sqrt2}\begin{bmatrix}1\\ 1\end{bmatrix},\qquad b_2 = \tfrac1{\sqrt2}\begin{bmatrix}1\\ -1\end{bmatrix} $$
form an ONB: $b_1^\top b_2 = 0$ and $\|b_1\|=\|b_2\|=1$. The standard basis $\{e_1,\dots,e_n\}$ is
the canonical ONB of $\mathbb R^n$.

**Why it is convenient.** With an ONB, the coordinates of any $x$ are just inner products,
$\alpha_i = \langle b_i, x\rangle$, and projection onto the span needs **no matrix inverse**:
$\pi_U(x) = B B^\top x$ (see [[orthogonal-projection]]). Any basis can be converted to an ONB by
[[gram-schmidt-orthogonalization|Gram–Schmidt]].

**ML usage.** PCA produces an orthonormal basis of principal directions (Ch. 10); SVM theory and
many numerical solvers (conjugate gradients, GMRES) exploit orthonormal bases for stability and speed.

## 🔴 Theory (Expert)
- The matrix $B=[b_1,\dots,b_n]$ of an ONB is an [[orthogonal-matrix]]: $B^\top B = I$.
- Existence is guaranteed: every finite-dimensional inner product space has an ONB (constructed via
  Gram–Schmidt). One construction concatenates non-orthogonal vectors $\tilde B=[\tilde b_1,\dots]$
  and applies Gaussian elimination to $[\tilde B\tilde B^\top \mid \tilde B]$.
- Orthonormality makes the Gram matrix the identity, so changes of basis between ONBs are
  orthogonal transformations — exactly the length/angle-preserving maps.
- Generalizes to infinite dimensions (Hilbert spaces), where complete orthonormal systems (e.g. the
  Fourier basis $\{1,\cos x,\sin x,\dots\}$) replace finite ONBs.

## Connections
- Builds on: [[basis-and-rank]], [[linear-independence]], [[orthogonality]], [[inner-product]]
- Leads to: [[orthogonal-matrix]], [[orthogonal-projection]], [[gram-schmidt-orthogonalization]]
- Used by: PCA (Ch. 10), SVM (Ch. 12), Krylov solvers
- People: [[jorgen-pedersen-gram]], [[erhard-schmidt]]

## Sources
- `raw/mml-book.pdf` §3.5 (Definition 3.9, Example 3.8), printed pp. 78–79 (PDF pp. 83–84).
