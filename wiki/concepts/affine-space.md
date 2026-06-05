---
title: Affine Space
type: concept
tags: [linear-algebra, foundations]
difficulty: advanced
sources: [mml-ch02-linear-algebra]
created: 2026-06-04
updated: 2026-06-04
---

# Affine Space

> **One-liner:** An affine space is a vector subspace that's been picked up and shifted off the origin — a flat that no longer passes through zero.

## 🟢 Intuition (Dummy)
A [[vector-space|subspace]] always contains the origin (the zero vector) — lines and planes through
$(0,0,0)$. An **affine** space is what you get when you slide one of those flats away from the
origin: a line or plane that *doesn't* go through zero. It still looks flat and straight, it's just
been *offset*. You describe it with a **support point** (where it sits) plus a **direction space**
(which way it extends). Lines and planes in everyday space that miss the origin are affine spaces.

## 🟡 Mechanics (Practitioner)
**Definition (Def 2.25):** given a [[vector-space]] $V$, a point $\mathbf x_0\in V$, and a subspace
$U\subseteq V$,
$$ L = \mathbf x_0 + U := \{\mathbf x_0 + \mathbf u : \mathbf u\in U\} $$
is an **affine subspace** (a.k.a. *linear manifold*). $U$ is the **direction / direction space**;
$\mathbf x_0$ is the **support point**. ⚠️ If $\mathbf x_0\notin U$, then $\mathbf 0\notin L$, so $L$
is **not** a vector subspace.

**Parametric equation:** with $(\mathbf b_1,\dots,\mathbf b_k)$ an ordered basis of $U$,
$$ \mathbf x = \mathbf x_0 + \lambda_1\mathbf b_1 + \dots + \lambda_k\mathbf b_k, \quad \lambda_i\in\mathbb R $$

**The hierarchy in $\mathbb R^n$:**
| Dimension of $L$ | Name |
|---|---|
| 1 | **line** $\;\mathbf y = \mathbf x_0 + \lambda \mathbf b_1$ |
| 2 | **plane** $\;\mathbf y = \mathbf x_0 + \lambda_1\mathbf b_1 + \lambda_2\mathbf b_2$ |
| $n-1$ | **hyperplane** |

(In $\mathbb R^2$ a line is a hyperplane; in $\mathbb R^3$ a plane is a hyperplane.)

⚠️ **ML caution (from the book):** the literature often blurs *linear* vs *affine*, calling affine
maps "linear." A neural net layer $\mathbf x\mapsto W\mathbf x + \mathbf b$ is **affine**, not linear
(the bias $\mathbf b$ is the support point shift).

## 🔴 Theory (Expert)
- **Affine mapping (Def 2.26):** an affine map is a [[linear-mapping]] composed with a translation,
  $\mathbf x \mapsto \Phi(\mathbf x) + \mathbf a$. Many linear-map properties carry over (e.g.
  compositions of affine maps are affine).
- **Connection to linear systems:** the solution set of an *inhomogeneous* system $A\mathbf x=\mathbf b$
  is **either empty or an affine subspace** of dimension $n-\text{rk}(A)$ — this is exactly the
  "particular $+$ homogeneous" decomposition from [[gaussian-elimination]]. The homogeneous case
  $A\mathbf x=\mathbf 0$ is the special affine space with support point $\mathbf x_0=\mathbf 0$ (a
  genuine subspace).
- **Hyperplanes** are the geometric objects that separate classes in Support Vector Machines
  (Ch. 12) — the affine subspace is where the "linear" decision boundary lives.

## Connections
- Builds on: [[vector-space]], [[linear-mapping]], [[basis-and-rank]]
- Leads to: hyperplanes & SVMs (Ch. 12), separating hyperplanes
- Used by: neural network affine layers, decision boundaries, regression offsets
- People: —

## Sources
- `raw/mml-book.pdf` §2.8, Defs 2.25–2.26, printed pp. 61–62 (PDF pp. 66–67). Validated 2026-06-04.
