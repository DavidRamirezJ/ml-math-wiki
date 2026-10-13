---
title: Basis and Rank
type: concept
tags: [linear-algebra, foundations, llm]
difficulty: intermediate
sources: [mml-ch02-linear-algebra, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-appA-introduction-to-pytorch]
created: 2026-06-04
updated: 2026-10-13
---

# Basis and Rank

> **One-liner:** A basis is the smallest set of vectors that can build the whole space; its size is the *dimension*, and a matrix's *rank* counts how many independent directions it really has.

## 🟢 Intuition (Dummy)
Think of building blocks. A **generating set** is *any* pile of blocks you can assemble every point
in the space from. A **basis** is a *minimal* such pile — no wasted, redundant blocks. In ordinary
3D space the three arrows pointing along x, y, z form a basis: three blocks, and you can reach
anywhere by combining them. The number of blocks in a basis is the **dimension** (3, here). The
**rank** of a matrix asks the same "how many real directions?" question about that matrix's columns.

**In LLMs.** Rank is the "low" in [[low-rank-adaptation|LoRA (low-rank adaptation)]]. When you
fine-tune a language model, LoRA only lets each weight matrix change along a few directions, say 16
instead of 768. In rank language: the change is a matrix of rank at most 16. Fewer directions means
far fewer numbers to train (book p. 322).

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

### In LLMs: rank as the LoRA budget (Build a Large Language Model, App. E)
LoRA replaces a full weight update $\Delta W$ by a product of two small matrices:

$$\Delta W \approx AB,\qquad A\in\mathbb R^{d_{in}\times r},\ B\in\mathbb R^{r\times d_{out}}\quad\Longrightarrow\quad \operatorname{rk}(AB)\le r$$

Read it as: "$\Delta W$ is approximately $A$ times $B$; $A$ has $d_{in}$ rows and $r$ columns, $B$ has
$r$ rows and $d_{out}$ columns; so the rank of $AB$ is at most $r$." Here $r$ is the "inner
dimension", a [[hyperparameter]] (book pp. 322–323; [[lora-rank-and-alpha]]).

Tiny example: $A=[1,2,0]^\top$ ($3\times1$) and $B=[1,-1,3]$ ($1\times3$) give
$AB=\begin{bmatrix}1&-1&3\\2&-2&6\\0&0&0\end{bmatrix}$. Every row is a multiple of $[1,-1,3]$, so
Gaussian elimination leaves one pivot: $\operatorname{rk}(AB)=1=r$.

The book's phrase "limiting model adjustments to a smaller dimensional subspace of the total
weight parameter space" (book p. 322) is this rank limit. With $r=16$ on GPT-2 small, the update
of each $768\times768$ attention matrix uses at most 16 of 768 possible column directions.

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
- **Rank of a product** (used by LoRA): $\text{rk}(AB)\le\min(\text{rk}(A),\text{rk}(B))$. Proof
  sketch: each column of $AB$ is $A$ times a column of $B$, so it lies in the column space of $A$;
  hence $\text{rk}(AB)\le\text{rk}(A)$. Apply the same to $(AB)^\top=B^\top A^\top$ and use
  column rank = row rank to get $\le\text{rk}(B)$. If the inner dimension is $r$, both ranks are $\le r$.
- **Converse (rank factorization):** every $M\in\mathbb R^{m\times n}$ with $\text{rk}(M)=r$ can be
  written $M=AB$ with $A\in\mathbb R^{m\times r}$, $B\in\mathbb R^{r\times n}$ (take $A$ = a basis of
  the column space, $B$ = the coordinates of each column in that basis). So LoRA's $AB$ reaches exactly
  the matrices of rank $\le r$. Note that this set is **not** a subspace (two rank-1 matrices can add up
  to rank 2), even though the book's intuition calls it one.

Rank is the single number tying together solvability, invertibility, dimension, and the geometry of
[[linear-mapping|linear maps]] — and it powers low-rank approximation (SVD, PCA — Chs. 4, 10).

## Connections
- Builds on: [[linear-independence]], [[gaussian-elimination]], [[vector-space]]
- Leads to: [[linear-mapping]], [[affine-space]], PCA & SVD (Chs. 4, 10), [[low-rank-approximation]], [[low-rank-adaptation]], [[lora-rank-and-alpha]]
- Used by: model capacity, dimensionality reduction, solvability checks, [[low-rank-adaptation]] (LoRA fine-tuning), [[parameter-efficient-fine-tuning]]
- People: —

> **From [[llm-appA-introduction-to-pytorch]]:** The word *rank* has a second meaning in PyTorch. In §A.2 the book says a tensor's "order (or rank)" is its number of dimensions (axes): a scalar has rank 0, a [[vector]] rank 1, a [[matrix]] rank 2 (Figure A.6, book p. 258). Figure A.6 also warns that a three-dimensional vector (3 entries) is still a rank-1 tensor. This *tensor rank* is not the matrix rank on this page. Example: $\begin{bmatrix}1&2\\2&4\end{bmatrix}$ is a rank-2 [[tensor]] (two axes), but its matrix rank is 1, because row 2 = 2 × row 1. In §A.9.3 "rank" means a third thing: the index of a GPU process in [[distributed-data-parallel]] (book pp. 284–286).

## Sources
- `raw/mml-book.pdf` §2.6, Defs 2.13–2.14, printed pp. 44–48 (PDF pp. 49–53). Validated 2026-06-04.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix E intro and §E.1, book pp. 322–324 (PDF pp. 344–346): "low-rank" as a smaller-dimensional subspace of weight space; $\Delta W\approx AB$ with inner dimension $r$ (Figure E.1, p. 323). The rank-of-a-product proof and the $3\times3$ example are this wiki's additions.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2 (tensor order/rank, Figure A.6), book p. 258 (PDF p. 280); §A.9.3 (process rank in DDP), book pp. 284–286 (PDF pp. 306–308).
