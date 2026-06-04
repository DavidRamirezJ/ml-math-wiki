---
title: Vector Space
type: concept
tags: [linear-algebra, foundations]
difficulty: intermediate
sources: [mml-ch02-linear-algebra]
created: 2026-06-04
updated: 2026-06-04
---

# Vector Space

> **One-liner:** A vector space is the complete collection of every vector you could ever reach by adding and scaling a starting set — nothing escapes it.

## 🟢 Intuition (Dummy)
Start with a couple of arrows. Add them, scale them, add the results, scale those... Where can
you end up? The set of *all* possible destinations is the vector space. It's "closed": you can
never operate your way *out* of it. If your arrows live in a flat plane, no amount of adding and
scaling lets you leave the plane — the plane is the vector space.

That property — "you can't escape" — is called **closure**, and it's the single most important
idea in this chapter.

## 🟡 Mechanics (Practitioner)
A **real vector space** is a set $V$ with two operations:
- vector addition $+ : V \times V \to V$
- scalar multiplication $\cdot : \mathbb{R} \times V \to V$

satisfying these axioms (closure is built into the codomains above):
- **Associativity & commutativity** of $+$
- **Additive identity** $\mathbf{0}$: $\;\mathbf{x} + \mathbf{0} = \mathbf{x}$
- **Additive inverse** $-\mathbf{x}$: $\;\mathbf{x} + (-\mathbf{x}) = \mathbf{0}$
- **Distributivity**: $\lambda(\mathbf{x}+\mathbf{y}) = \lambda\mathbf{x}+\lambda\mathbf{y}$ and $(\lambda+\psi)\mathbf{x} = \lambda\mathbf{x}+\psi\mathbf{x}$
- **Associativity of scaling**: $\lambda(\psi\mathbf{x}) = (\lambda\psi)\mathbf{x}$
- **Scaling identity**: $1\cdot\mathbf{x} = \mathbf{x}$

**Canonical example:** $\mathbb{R}^n$ with component-wise $+$ and scaling. **Other examples:**
polynomials, real-valued functions, $m\times n$ matrices.

## 🔴 Theory (Expert)
Algebraically, $(V, +)$ is an **abelian group** and scalar multiplication makes $V$ a **module over
the field $\mathbb{R}$** (a module over a field *is* a vector space). The chapter's mind map (Fig. 2.2)
makes this lineage explicit: *Group → Vector space → ...*.

- A **subspace** $U \subseteq V$ is a subset that is itself closed under $+$ and scaling (and contains
  $\mathbf{0}$). Solution sets of *homogeneous* linear systems are subspaces; general solution sets are
  [[affine-space|affine subspaces]] (a subspace shifted off the origin).
- The **span** of a set of vectors is the smallest subspace containing them — the operational meaning
  of closure.
- [[linear-independence]] + spanning ⇒ a [[basis-and-rank|basis]]; the basis size is the **dimension**,
  and it pins down the 1:1 correspondence with $\mathbb{R}^{\dim V}$ for finite-dimensional spaces.

"The concept of a vector space and its properties underlie much of machine learning" (p. 18).

## Connections
- Builds on: [[vector]]
- Leads to: [[linear-independence]], [[basis-and-rank]], [[linear-mapping]], [[affine-space]]
- Used by: [[system-of-linear-equations]]
- People: —

## Sources
- `raw/mml-book.pdf` §2 intro & §2.4 (**Definition 2.9**), printed pp. 17–18, 35–37 (PDF pp. 22–23, 40–42).
- ✅ Axioms validated verbatim against Definition 2.9 on 2026-06-04.
