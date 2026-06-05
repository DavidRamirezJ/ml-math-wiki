---
title: Cholesky Decomposition
type: concept
tags: [linear-algebra, matrix-decomposition]
difficulty: intermediate
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Cholesky Decomposition

> **One-liner:** It's the "square root" of a nice symmetric matrix — it splits the matrix into a triangular factor times its mirror image.

## 🟢 Intuition (Dummy)
For a positive number like $9$ you can take a square root: $9=3\cdot 3$. The **Cholesky
decomposition** does the same trick for a special family of matrices (symmetric, positive definite —
the matrix analogue of "positive numbers"). It writes the matrix as a triangular matrix $L$ times its
flipped version $L^\top$, so $A=LL^\top$. The triangular factor $L$ acts like the matrix's square
root, and it's quick to compute and reuse. This is one of the workhorses of numerical machine
learning.

## 🟡 Mechanics (Practitioner)
**Theorem (Thm 4.18):** a **symmetric, positive definite** matrix $A$ can be uniquely factorized as
$$A=LL^\top,$$
where $L$ is **lower-triangular with positive diagonal entries** (the **Cholesky factor**).

**Computing $L$ (for a $3\times 3$, from Example 4.10):** match entries of $LL^\top$ to $A$ and solve
top-down:
$$l_{11}=\sqrt{a_{11}},\quad l_{22}=\sqrt{a_{22}-l_{21}^2},\quad l_{33}=\sqrt{a_{33}-(l_{31}^2+l_{32}^2)},$$
$$l_{21}=\tfrac{1}{l_{11}}a_{21},\quad l_{31}=\tfrac{1}{l_{11}}a_{31},\quad l_{32}=\tfrac{1}{l_{22}}\big(a_{32}-l_{31}l_{21}\big).$$
Each new entry is computed from $A$ and previously-found entries — a single forward sweep.

**Cheap determinant:** since $\det(A)=\det(L)\det(L^\top)=\det(L)^2$ and $L$ is triangular,
$$\det(A)=\prod_i l_{ii}^2.$$

**ML usage:**
- **Sampling from a Gaussian:** if $\Sigma=LL^\top$ is the covariance and $\mathbf z\sim\mathcal N(\mathbf 0,I)$,
  then $\boldsymbol\mu+L\mathbf z\sim\mathcal N(\boldsymbol\mu,\Sigma)$ — the basis of the
  **reparameterization trick** in variational autoencoders.
- **Solving linear systems** with SPD matrices (e.g. Gaussian-process regression, normal equations):
  $A\mathbf x=\mathbf b$ becomes two fast triangular solves.
- **Efficient log-determinants** for Gaussian log-likelihoods via $\log\det\Sigma=2\sum_i\log l_{ii}$.

## 🔴 Theory (Expert)
- **Existence & uniqueness:** positive definiteness ($\mathbf x^\top A\mathbf x>0$ for all
  $\mathbf x\neq\mathbf 0$) guarantees every $a_{ii}-\sum l_{ij}^2$ under a square root stays strictly
  positive, so the recursion never breaks and yields a unique $L$ with positive diagonal. It is a
  special, numerically stable case of LU decomposition exploiting symmetry (roughly half the work,
  no pivoting needed).
- The factorization places Cholesky at the bottom of the [[singular-value-decomposition|matrix
  phylogeny]] (Fig. 4.13): it exists precisely for the positive-definite subset of the symmetric
  matrices.
- Positive *semi*definite matrices admit a (non-unique) Cholesky-like factorization with possibly
  zero diagonal entries; strict positive definiteness is what buys uniqueness.

## Connections
- Builds on: [[symmetric-positive-definite-matrix|positive definite matrix]], [[determinant]], [[gaussian-elimination]], [[matrix]]
- Leads to: [[singular-value-decomposition]] (matrix phylogeny), efficient Gaussian computations
- Used by: multivariate Gaussian sampling, reparameterization trick, Gaussian processes, [[principal-component-analysis]] preprocessing
- People: [[andre-louis-cholesky]]

## Sources
- `raw/mml-book.pdf` §4.3, Thm 4.18 & Example 4.10, printed pp. 114–115 (PDF pp. 119–120).
