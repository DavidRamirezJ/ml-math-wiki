---
title: Vector
type: concept
tags: [linear-algebra, foundations]
difficulty: beginner
sources: [mml-ch02-linear-algebra]
created: 2026-06-04
updated: 2026-06-04
---

# Vector

> **One-liner:** A vector is anything you can *add to another of its kind* and *stretch by a number* and still get the same kind of thing back.

## 🟢 Intuition (Dummy)
Think of an arrow on paper. You can lay two arrows tip-to-tail to get a new arrow (addition),
and you can make an arrow twice as long (scaling). That's a vector.

The surprising part: arrows aren't the only vectors. A **polynomial**, an **audio clip**, and a
**list of numbers** are *also* vectors, because you can add two of them and scale them and still
get the same kind of object. Math doesn't care what the thing *is* — only that it obeys those two
rules. So "vector" is a club, and the membership test is just: *can I add, and can I scale?*

## 🟡 Mechanics (Practitioner)
In this book (and almost all of ML) a vector is an element of $\mathbb{R}^n$ — a tuple of $n$ real numbers:

$$
\mathbf{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \in \mathbb{R}^3
$$

The two operations, done **component-wise**:
- **Addition:** $\mathbf{a} + \mathbf{b} = \mathbf{c} \in \mathbb{R}^n$
- **Scalar multiplication:** $\lambda \mathbf{a} \in \mathbb{R}^n$ for $\lambda \in \mathbb{R}$

**Worked example:**
$$
\begin{bmatrix}1\\2\end{bmatrix} + \begin{bmatrix}3\\0\end{bmatrix} = \begin{bmatrix}4\\2\end{bmatrix},
\qquad
3\begin{bmatrix}1\\2\end{bmatrix} = \begin{bmatrix}3\\6\end{bmatrix}
$$

**In ML:** a data point with $n$ features is a vector in $\mathbb{R}^n$; an image is a vector of pixel
values; word embeddings are vectors. The whole pipeline runs on $\mathbb{R}^n$. ⚠️ On a computer,
check that your array operations actually perform *vector* operations (broadcasting can fool you).

## 🔴 Theory (Expert)
A vector is, formally, an **element of a [[vector-space]]** — there is no intrinsic "vectorness"
to an object, only membership in a set equipped with addition and scalar multiplication satisfying
the eight vector-space axioms. Hence the book's examples:
- **Geometric vectors** (directed segments) — concrete, drawable.
- **Polynomials** — abstract, infinite-dimensional in general.
- **Audio signals** — sampled function values.
- **$\mathbb{R}^n$** — the canonical finite-dimensional case.

Key fact: for **finite-dimensional** vector spaces there is a **1:1 correspondence** between any
kind of vector and $\mathbb{R}^n$ (via a choice of [[basis-and-rank|basis]]). This is why focusing on
$\mathbb{R}^n$ loses no generality in finite dimensions. The unifying principle is **closure**: the
set of all things reachable by adding and scaling a starting set *is* a vector space (§2.4).

## Connections
- Builds on: *(none — foundational)*
- Leads to: [[vector-space]], [[linear-independence]], [[basis-and-rank]]
- Used by: [[system-of-linear-equations]], essentially all of ML
- People: —

## Sources
- `raw/mml-book.pdf` §2 intro, pp. 17–18 (PDF pp. 22–23)
