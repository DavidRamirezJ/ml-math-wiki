---
title: "MML Ch. 3 — Analytic Geometry"
type: source
tags: [linear-algebra, analytic-geometry, inner-product, projection]
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch. 3 — Analytic Geometry

- **Raw file:** `raw/mml-book.pdf` §3.1–3.10, printed pp. 70–97 (PDF pp. 75–102)
- **Type:** book chapter
- **Ingested:** 2026-06-04

This chapter adds *geometry* to the abstract [[vector-space|vector spaces]] and [[linear-mapping|linear mappings]]
of Chapter 2. By equipping a vector space with an **inner product**, we obtain lengths, distances,
angles, orthogonality, projections, and rotations — the toolkit later used by regression (Ch. 9),
PCA (Ch. 10), and support vector machines (Ch. 12).

## Section coverage

| Section | Title | Wiki page |
|---|---|---|
| 3.1 | Norms | [[norm]] |
| 3.2 | Inner Products (dot product, bilinear/symmetric/positive-definite forms, SPD matrices) | [[inner-product]], [[symmetric-positive-definite-matrix]] |
| 3.3 | Lengths and Distances (induced norm, Cauchy–Schwarz, metric) | [[norm]], [[metric]] |
| 3.4 | Angles and Orthogonality (angle via inner product, orthogonal matrices) | [[angle-between-vectors]], [[orthogonality]], [[orthogonal-matrix]] |
| 3.5 | Orthonormal Basis | [[orthonormal-basis]] |
| 3.6 | Orthogonal Complement | [[orthogonal-complement]] |
| 3.7 | Inner Product of Functions | [[inner-product]] (§ Theory) |
| 3.8 | Orthogonal Projections (lines, general subspaces, normal equation, Gram–Schmidt, affine) | [[orthogonal-projection]], [[gram-schmidt-orthogonalization]] |
| 3.9 | Rotations (R², R³, Givens, properties) | [[rotation]] |
| 3.10 | Further Reading | — |

## TL;DR (5 bullets)
- A **[[norm]]** assigns a length to a vector (absolutely homogeneous, triangle inequality, positive definite); the Euclidean ($\ell_2$) norm is the default, the Manhattan ($\ell_1$) norm an alternative.
- An **[[inner-product]]** is a symmetric, positive-definite bilinear form; it *induces* a norm $\|x\|=\sqrt{\langle x,x\rangle}$ and through it lengths, distances ([[metric]]), and **[[angle-between-vectors|angles]]** via $\cos\omega = \frac{\langle x,y\rangle}{\|x\|\|y\|}$.
- Every inner product on a finite-dimensional space corresponds to a **[[symmetric-positive-definite-matrix]]** $A$ via $\langle x,y\rangle=\hat x^\top A\hat y$ (Thm 3.5); $x\perp y \iff \langle x,y\rangle=0$.
- **[[orthogonal-projection|Orthogonal projection]]** of $x$ onto a subspace with basis matrix $B$ is $\pi_U(x)=B(B^\top B)^{-1}B^\top x$ (the **normal equation** $B^\top B\lambda = B^\top x$); this is the geometric root of least squares, PCA, and SVM margins. **[[gram-schmidt-orthogonalization|Gram–Schmidt]]** turns any basis into an orthonormal one.
- **[[rotation|Rotations]]** are length- and angle-preserving [[orthogonal-matrix|orthogonal]] automorphisms; in $\mathbb R^2$, $R(\theta)=\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}$, generalized in $n$D by **Givens rotations**.

## Key concepts introduced
- [[norm]] — length of a vector ($\ell_1$, $\ell_2$)
- [[inner-product]] — symmetric positive-definite bilinear form; dot product is the canonical case
- [[symmetric-positive-definite-matrix]] — matrix representation of an inner product
- [[metric]] — distance function induced by a norm
- [[angle-between-vectors]] — angle defined via the inner product
- [[orthogonality]] — $\langle x,y\rangle=0$; orthonormal vectors
- [[orthogonal-matrix]] — $A^\top A = I$; preserves lengths and angles
- [[orthonormal-basis]] — basis of mutually orthogonal unit vectors
- [[orthogonal-complement]] — subspace of all vectors orthogonal to a given subspace
- [[orthogonal-projection]] — closest point in a subspace; normal equation, projection matrix, pseudo-inverse
- [[gram-schmidt-orthogonalization]] — constructive orthonormalization of a basis
- [[rotation]] — length/angle-preserving orthogonal transformation; Givens rotations

## Notable results / theorems
- **Definition 3.1** — Norm axioms.
- **Definitions 3.2–3.3** — Bilinear, symmetric, positive-definite forms ⇒ inner product.
- **Definition 3.4 + Theorem 3.5** — Inner products ↔ symmetric positive-definite matrices ($\langle x,y\rangle = \hat x^\top A\hat y$).
- **Cauchy–Schwarz inequality (3.17)** — $|\langle x,y\rangle| \le \|x\|\,\|y\|$; the basis for defining angles.
- **Definition 3.6** — Distance and metric.
- **Definitions 3.7–3.8** — Orthogonality, orthonormality, orthogonal matrices ($A^{-1}=A^\top$).
- **Definition 3.9** — Orthonormal basis.
- **Definition 3.10** — Projection ($\pi^2=\pi$), projection matrix ($P_\pi^2=P_\pi$); normal equation (3.56), pseudo-inverse (3.57).
- **Definition 3.11** — Givens rotation.

## What this changes in the wiki
- Refines [[linear-mapping]]: rotations and orthogonal projections are concrete linear maps; the "rotations (Ch. 3.9)" forward-reference there now resolves to [[rotation]].
- Extends [[affine-space]] with projection onto affine subspaces $\pi_L(x)=x_0+\pi_U(x-x_0)$.
- Connects to [[basis-and-rank]] / [[linear-independence]] (orthonormal bases) and [[system-of-linear-equations]] (least-squares / normal equation for unsolvable systems).
- No contradictions with existing pages.
