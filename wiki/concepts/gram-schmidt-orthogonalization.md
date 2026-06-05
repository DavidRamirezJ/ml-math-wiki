---
title: Gram–Schmidt Orthogonalization
type: concept
tags: [analytic-geometry, linear-algebra, inner-product]
difficulty: intermediate
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Gram–Schmidt Orthogonalization

> **One-liner:** Gram–Schmidt is a step-by-step recipe that straightens any set of basis vectors into mutually perpendicular ones.

## 🟢 Intuition (Dummy)
You have a few reference directions, but they are tilted and overlapping — awkward to work with.
Gram–Schmidt cleans them up one at a time. Keep the first direction as-is. For the second, remove
the part of it that overlaps with the first, leaving only the perpendicular leftover. For the
third, remove its overlap with the first two. Each step subtracts the "shadow" already covered, so
what remains is a fresh, perpendicular direction. Normalize each to length one, and you have a
perfect right-angled coordinate system.

## 🟡 Mechanics (Practitioner)
Given any basis $(b_1,\dots,b_n)$ of $V$, build an **orthogonal** basis $(u_1,\dots,u_n)$
iteratively (§3.8.3):
$$ u_1 := b_1, \qquad u_k := b_k - \pi_{\operatorname{span}[u_1,\dots,u_{k-1}]}(b_k), \quad k=2,\dots,n. $$
Each $u_k$ is $b_k$ with its [[orthogonal-projection|projection]] onto the previously built
directions subtracted off, so $u_k \perp \operatorname{span}[u_1,\dots,u_{k-1}]$. Normalizing
$u_k \mapsto u_k/\|u_k\|$ gives an [[orthonormal-basis|orthonormal basis (ONB)]]. The span is
preserved: $\operatorname{span}[b_1,\dots,b_n]=\operatorname{span}[u_1,\dots,u_n]$.

**Worked example (§3.8.3).** $b_1=\big[2,0\big]^\top$, $b_2=\big[1,1\big]^\top$ in $\mathbb R^2$:
$$ u_1 = b_1 = \begin{bmatrix}2\\0\end{bmatrix}, \qquad
   u_2 = b_2 - \frac{u_1 u_1^\top}{\|u_1\|^2}b_2 = \begin{bmatrix}1\\1\end{bmatrix} - \begin{bmatrix}1&0\\0&0\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}0\\1\end{bmatrix}. $$
Indeed $u_1^\top u_2 = 0$.

**ML usage.** Gram–Schmidt is the conceptual basis of **QR decomposition** ($A=QR$ with $Q$
orthonormal), used for solving least-squares problems stably. Orthonormal bases it produces speed
up projections (no matrix inverse) and stabilize numerical algorithms.

## 🔴 Theory (Expert)
- The construction proves **existence** of an ONB in any finite-dimensional inner product space.
- The vectors $u_k$ are nonzero precisely because $(b_1,\dots,b_n)$ is linearly independent
  ([[linear-independence]]); if a $b_k$ were dependent, $u_k$ would collapse to $\mathbf 0$.
- **Classical** Gram–Schmidt is numerically unstable (loss of orthogonality under rounding);
  **modified** Gram–Schmidt and Householder/QR are preferred in practice.
- Orthogonal bases produced this way underpin Krylov-subspace solvers (conjugate gradients, GMRES),
  which minimize residuals that are mutually orthogonal.

## Connections
- Builds on: [[orthogonal-projection]], [[orthogonality]], [[basis-and-rank]], [[linear-independence]]
- Leads to: [[orthonormal-basis]], QR decomposition, Krylov methods
- Used by: least squares solvers, numerical linear algebra
- People: [[jorgen-pedersen-gram]], [[erhard-schmidt]]

## Sources
- `raw/mml-book.pdf` §3.8.3 (Eqs 3.67–3.71, Example 3.12), printed pp. 89–90 (PDF pp. 94–95).
