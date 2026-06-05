---
title: Basis and Rank
type: concept
tags: [linear-algebra, foundations]
difficulty: intermediate
sources: [mml-ch02-linear-algebra]
created: 2026-06-04
updated: 2026-06-04
---

# Basis and Rank

> **One-liner:** A basis is the smallest set of vectors that can build the whole space; its size is the *dimension*, and a matrix's *rank* counts how many independent directions it really has.

## 🟢 Intuition (Dummy)
Think of building blocks. A **generating set** is *any* pile of blocks you can assemble every point
in the space from. A **basis** is a *minimal* such pile — no wasted, redundant blocks. In ordinary
3D space the three arrows pointing along x, y, z form a basis: three blocks, and you can reach
anywhere by combining them. The number of blocks in a basis is the **dimension** (3, here). The
**rank** of a matrix asks the same "how many real directions?" question about that matrix's columns.

## 🟡 Mechanics (Practitioner)
- **Span** of $A=\{\mathbf x_1,\dots,\mathbf x_k\}$: the set of *all* linear combinations. If it
  covers $V$, $A$ is a **generating set**, $V=\text{span}[A]$.
- **Basis (Def 2.14):** a generating set that is also **[[linear-independence|linearly independent]]**
  — equivalently a *minimal* generating set, *maximal* independent set, and the set in which every
  $\mathbf x\in V$ has a **unique** coordinate representation.
- **Dimension** $\dim(V)$: the number of basis vectors. All bases of $V$ have the same size. ⚠️ Not
  the number of entries in a vector — e.g. the span of the single vector $[0,1]^\top$ is 1-dimensional.
- **Canonical basis** of $\mathbb R^3$: $[1,0,0]^\top,[0,1,0]^\top,[0,0,1]^\top$ — but bases are *not* unique.

**Finding a basis of $U=\text{span}[\mathbf x_1,\dots,\mathbf x_m]$:** columns → matrix → REF →
the original vectors at the **pivot columns** form a basis (Example 2.17: $\{\mathbf x_1,\mathbf x_2,\mathbf x_4\}$).

**Rank** $\text{rk}(A)$: the number of linearly independent columns ( $=$ independent rows). Find it
by counting pivots after Gaussian elimination.

## 🔴 Theory (Expert)
Properties of rank (§2.6.2):
- $\text{rk}(A) = \text{rk}(A^\top)$ — **column rank = row rank**.
- Columns of $A\in\mathbb R^{m\times n}$ span a subspace of $\mathbb R^m$ of dimension $\text{rk}(A)$
  — later called the **image / range**.
- **Null space / kernel** of $A$ has dimension $n - \text{rk}(A)$ — the **rank–nullity** relation.
- $A\in\mathbb R^{n\times n}$ is **invertible ⇔ $\text{rk}(A)=n$**.
- $A\mathbf x=\mathbf b$ is solvable **iff** $\text{rk}(A)=\text{rk}([A\,|\,\mathbf b])$.
- **Full rank:** $\text{rk}(A)=\min(m,n)$; otherwise **rank deficient**.
- Subspace nesting: $U\subseteq V \Rightarrow \dim(U)\le\dim(V)$, with equality iff $U=V$.

Rank is the single number tying together solvability, invertibility, dimension, and the geometry of
[[linear-mapping|linear maps]] — and it powers low-rank approximation (SVD, PCA — Chs. 4, 10).

## Connections
- Builds on: [[linear-independence]], [[gaussian-elimination]], [[vector-space]]
- Leads to: [[linear-mapping]], [[affine-space]], PCA & SVD (Chs. 4, 10)
- Used by: model capacity, dimensionality reduction, solvability checks
- People: —

## Sources
- `raw/mml-book.pdf` §2.6, Defs 2.13–2.14, printed pp. 44–48 (PDF pp. 49–53). Validated 2026-06-04.
