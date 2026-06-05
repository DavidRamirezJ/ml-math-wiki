---
title: Linear Independence
type: concept
tags: [linear-algebra, foundations]
difficulty: intermediate
sources: [mml-ch02-linear-algebra]
created: 2026-06-04
updated: 2026-06-04
---

# Linear Independence

> **One-liner:** A set of vectors is linearly independent when none of them is redundant — you can't build any one out of the others.

## 🟢 Intuition (Dummy)
The book's geography example: from Nairobi you can reach Kigali by going "506 km Northwest then
374 km Southwest." Those two directions are **independent** — neither can be made from the other.
But "751 km West" is just a combination of the first two, so adding it is **redundant**: it carries
no new information. Linearly independent vectors are like a minimal set of directions where every
one *adds something*; remove any and you lose the ability to reach somewhere.

## 🟡 Mechanics (Practitioner)
A **linear combination** of $\mathbf{x}_1,\dots,\mathbf{x}_k$ is $\mathbf v = \sum_{i=1}^k \lambda_i \mathbf{x}_i$, $\lambda_i\in\mathbb R$.

**Definition (Def 2.12):** the vectors are **linearly independent** if the *only* way to get
$$ \sum_{i=1}^k \lambda_i \mathbf{x}_i = \mathbf 0 $$
is the **trivial** solution $\lambda_1=\dots=\lambda_k=0$. If a *non-trivial* combination (some
$\lambda_i\neq 0$) also gives $\mathbf 0$, they are **linearly dependent**.

**Quick facts:**
- $k$ vectors are *either* dependent *or* independent — no third option.
- If any vector is $\mathbf 0$, or two are identical, or one is a multiple of another → dependent.
- $m$ combinations of $k$ vectors are dependent whenever $m > k$.

**Practical test (Gaussian elimination):** stack the vectors as columns of a matrix, reduce to REF.
- **Every column is a pivot column ⇔ linearly independent.**
- Any non-pivot column ⇒ dependent; that column is a combination of the pivot columns to its left.

Example 2.14: three vectors in $\mathbb R^4$ reduce to three pivot columns → independent. Example
2.15/2.16: a fourth reduces to a non-pivot column with $\mathbf x_4 = -7\mathbf x_1 -15\mathbf x_2 -18\mathbf x_3$ → dependent.

## 🔴 Theory (Expert)
- **Change of basis invariance:** if $\mathbf{b}_1,\dots,\mathbf{b}_k$ are independent and
  $\mathbf{x}_j = B\boldsymbol\lambda_j$, then $\{\mathbf x_j\}$ are independent **iff** the
  coordinate vectors $\{\boldsymbol\lambda_j\}$ are independent (independence is a property of the
  *coordinates*, not the embedding). This is why the REF column test works.
- Independence is the gatekeeper for [[basis-and-rank|basis]]: a [[basis-and-rank|basis]] is a
  *linearly independent generating set*, and the **rank** of a matrix is exactly its number of
  independent columns (= independent rows).
- "One of the most important concepts in linear algebra" — it formalizes *non-redundancy*, which
  underlies dimension, rank, and the uniqueness of coordinate representations.

## Connections
- Builds on: [[vector]], [[vector-space]], [[gaussian-elimination]]
- Leads to: [[basis-and-rank]], [[linear-mapping]]
- Used by: dimensionality reduction (PCA, Ch. 10), feature redundancy analysis
- People: —

## Sources
- `raw/mml-book.pdf` §2.5, Defs 2.11–2.12, printed pp. 40–44 (PDF pp. 45–49). Validated 2026-06-04.
