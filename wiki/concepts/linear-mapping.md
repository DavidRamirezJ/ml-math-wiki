---
title: Linear Mapping
type: concept
tags: [linear-algebra, foundations]
difficulty: advanced
sources: [mml-ch02-linear-algebra]
created: 2026-06-04
updated: 2026-06-04
---

# Linear Mapping

> **One-liner:** A linear mapping is a function between vector spaces that respects adding and scaling — and every such function is just a matrix in disguise.

## 🟢 Intuition (Dummy)
Some functions are "well-behaved": if you stretch the input twice as much, the output stretches
twice as much; if you add two inputs, the outputs add. A linear mapping is exactly that kind of
function on vectors — it doesn't bend or warp space unevenly, it only rotates, scales, shears, or
projects. The magic fact: *any* such map can be written as multiplication by a [[matrix]]. So
"function on vectors" and "matrix" are two views of the same thing.

## 🟡 Mechanics (Practitioner)
**Definition (Def 2.15):** $\Phi: V\to W$ is a **linear mapping** (a.k.a. *vector space
homomorphism* / *linear transformation*) if for all $\mathbf x,\mathbf y\in V$, $\lambda,\psi\in\mathbb R$:
$$ \Phi(\lambda\mathbf x + \psi\mathbf y) = \lambda\,\Phi(\mathbf x) + \psi\,\Phi(\mathbf y) $$
(Equivalently the two conditions $\Phi(\mathbf x+\mathbf y)=\Phi(\mathbf x)+\Phi(\mathbf y)$ and $\Phi(\lambda\mathbf x)=\lambda\Phi(\mathbf x)$.)

**Special maps (Def 2.16 + 2.7):**
| Term | Meaning |
|---|---|
| Injective | $\Phi(\mathbf x)=\Phi(\mathbf y)\Rightarrow \mathbf x=\mathbf y$ |
| Surjective | $\Phi(V)=W$ (everything in $W$ is reached) |
| Bijective | injective **and** surjective (can be undone by $\Phi^{-1}$) |
| Isomorphism | linear **and** bijective |
| Endomorphism | linear, $V\to V$ |
| Automorphism | linear, bijective, $V\to V$ |

**Coordinates (Def 2.18):** pick an *ordered basis* $B=(\mathbf b_1,\dots,\mathbf b_n)$. Any
$\mathbf x = \alpha_1\mathbf b_1+\dots+\alpha_n\mathbf b_n$ has a **coordinate vector**
$\boldsymbol\alpha=[\alpha_1,\dots,\alpha_n]^\top$. A basis *is* a coordinate system; the same vector
$\mathbf x$ has different coordinates in different bases (Fig. 2.9: $[2,3]^\top$ in $(e_1,e_2)$ becomes
$\tfrac12[-1,5]^\top$ in $(b_1,b_2)$).

**Transformation matrix (Def 2.19):** with ordered bases $B$ of $V$, $C$ of $W$, write each
$\Phi(\mathbf b_j) = \sum_i \alpha_{ij}\mathbf c_i$; then $A_\Phi=(\alpha_{ij})$ is the matrix of $\Phi$.
**The coordinates of $\Phi(\mathbf b_j)$ in $C$ are the $j$-th column of $A_\Phi$.**

## 🔴 Theory (Expert)
- **Theorem 2.17:** finite-dimensional $V,W$ are **isomorphic ⇔ $\dim V = \dim W$.** Same dimension
  ⇒ "the same up to relabeling." This justifies treating any $n$-dim space as $\mathbb R^n$, and
  $\mathbb R^{m\times n}\cong\mathbb R^{mn}$ (the matrix-stacking of [[matrix]]).
- **Closure of the class:** compositions of linear maps are linear; the inverse of an isomorphism is
  an isomorphism; sums $\Phi+\Psi$ and scalings $\lambda\Phi$ are linear (linear maps form a vector
  space themselves).
- **Kernel & image** of $\Phi$ (matrix $A$): $\ker(A)$ has dimension $n-\text{rk}(A)$, $\text{image}$
  has dimension $\text{rk}(A)$ — the [[basis-and-rank|rank–nullity]] theorem governs injectivity
  (trivial kernel) and surjectivity (full-rank image).
- In ML, every fully-connected layer *without* its bias is a linear mapping; with bias it becomes an
  [[affine-space|affine mapping]].

## Connections
- Builds on: [[vector-space]], [[matrix]], [[basis-and-rank]]
- Leads to: [[affine-space]], change of basis (Ch. 2.7.2), matrix decompositions (Ch. 4)
- Used by: neural network layers, PCA projections, rotations (Ch. 3.9)
- People: —

## Sources
- `raw/mml-book.pdf` §2.7, Defs 2.15–2.19, Thm 2.17, printed pp. 48–60 (PDF pp. 53–65). Validated 2026-06-04.
