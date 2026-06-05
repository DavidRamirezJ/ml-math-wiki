---
title: Eigenvalues and Eigenvectors
type: concept
tags: [linear-algebra, matrix-decomposition]
difficulty: intermediate
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Eigenvalues and Eigenvectors

> **One-liner:** An eigenvector is a special direction that a matrix only stretches or shrinks (never rotates off course), and its eigenvalue is the stretch factor.

## 🟢 Intuition (Dummy)
A [[matrix]] usually does something complicated to vectors: it rotates them, shears them, stretches
them — all at once. But almost every matrix has a few magic directions where the effect is *boring*:
the vector just gets longer or shorter, still pointing along the same line. Those magic directions are
the **eigenvectors**, and the amount of stretching is the **eigenvalue**. "Eigen" is German for
"own/characteristic." If the eigenvalue is $2$, that direction doubles; if it's $0.5$, it halves; if
it's negative, it flips around; if it's $0$, that direction gets crushed to nothing.

## 🟡 Mechanics (Practitioner)
**Definition (Def 4.6):** $\lambda\in\mathbb R$ is an **eigenvalue** of $A\in\mathbb R^{n\times n}$
and $\mathbf x\in\mathbb R^n\setminus\{\mathbf 0\}$ a corresponding **eigenvector** if
$$A\mathbf x=\lambda\mathbf x \qquad\text{(the eigenvalue equation)}.$$

**How to find them (procedure):**
1. **Eigenvalues:** solve $\det(A-\lambda I)=0$ — the roots of the [[characteristic-polynomial]].
2. **Eigenvectors:** for each $\lambda$, solve the homogeneous system $(A-\lambda I)\mathbf x=\mathbf 0$.

**Worked example** for $A=\begin{pmatrix}4&2\\1&3\end{pmatrix}$:
$p_A(\lambda)=(\lambda-2)(\lambda-5)$ gives $\lambda_1=5,\ \lambda_2=2$.
- $\lambda=5$: $(A-5I)\mathbf x=\mathbf 0 \Rightarrow E_5=\operatorname{span}\big[(2,1)^\top\big]$.
- $\lambda=2$: $(A-2I)\mathbf x=\mathbf 0 \Rightarrow E_2=\operatorname{span}\big[(1,-1)^\top\big]$.

**Key facts:**
- Eigenvectors are non-unique: any nonzero scalar multiple $c\mathbf x$ is also an eigenvector for the
  same $\lambda$ (they are *collinear*).
- The set of all eigenvectors for a given $\lambda$ (plus $\mathbf 0$) forms the **eigenspace**
  $E_\lambda=\ker(A-\lambda I)$, a subspace of $\mathbb R^n$ (Def 4.10). The set of all eigenvalues is
  the **eigenspectrum** (or spectrum).
- $\det A=\prod_i\lambda_i$ (Thm 4.16) and $\operatorname{tr}A=\sum_i\lambda_i$ (Thm 4.17).

**Geometric reading (Fig. 4.4):** $A_1=\operatorname{diag}(\tfrac12,2)$ stretches one axis by $2$ and
compresses the other by $\tfrac12$ (area preserved, $\det=1$); a rotation has only complex
eigenvalues (no real fixed directions); $A_4=\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$ has an eigenvalue
$0$, collapsing one direction to nothing.

**ML usage:** eigenvectors are the backbone of [[principal-component-analysis]] (principal directions
of variance), spectral clustering, Google's PageRank (the dominant eigenvector of a transition
matrix is the page-importance vector, Example 4.9), and the analysis of dynamical systems and
optimization landscapes (eigenvalues of the Hessian).

## 🔴 Theory (Expert)
- **Equivalences:** $\lambda$ is an eigenvalue $\iff (A-\lambda I)\mathbf x=\mathbf 0$ has a nontrivial
  solution $\iff \operatorname{rk}(A-\lambda I)<n \iff \det(A-\lambda I)=0 \iff \lambda$ is a root of $p_A$.
- **Multiplicities (Defs 4.9, 4.11):** the **algebraic multiplicity** of $\lambda_i$ is its multiplicity
  as a root of $p_A$; the **geometric multiplicity** is $\dim E_{\lambda_i}$. Always
  $1\le \text{geometric} \le \text{algebraic}$.
- **Thm 4.12:** eigenvectors belonging to $n$ *distinct* eigenvalues are linearly independent — they
  form a basis of $\mathbb R^n$. This is the gateway to the [[eigendecomposition]].
- **Defective matrices (Def 4.13):** a matrix with fewer than $n$ linearly independent eigenvectors is
  *defective* and **cannot** be diagonalized over $\mathbb R$ (e.g. $\begin{pmatrix}2&1\\0&2\end{pmatrix}$:
  algebraic multiplicity $2$, geometric multiplicity $1$). Defectiveness requires a repeated eigenvalue.
- $A$ and $A^\top$ share eigenvalues (not necessarily eigenvectors); [[linear-mapping|similar matrices]]
  share eigenvalues, so the spectrum is a basis-invariant property of a linear map.
- **Thm 4.14:** for any $A$, $S:=A^\top A$ is symmetric positive semidefinite (positive definite if
  $\operatorname{rk}A=n$) — the foundation of the [[singular-value-decomposition]]. Symmetric positive
  definite matrices have strictly positive real eigenvalues; symmetric matrices have an orthonormal
  eigenbasis (the [[spectral-theorem]]).

## Connections
- Builds on: [[characteristic-polynomial]], [[determinant]], [[system-of-linear-equations]], [[linear-independence]]
- Leads to: [[eigendecomposition]], [[spectral-theorem]], [[singular-value-decomposition]]
- Used by: [[principal-component-analysis]], PageRank, spectral clustering, [[low-rank-approximation]]
- People: —

## Sources
- `raw/mml-book.pdf` §4.2, Defs 4.6–4.13, Thms 4.8, 4.12, 4.14, 4.16, 4.17, Examples 4.4–4.9, printed pp. 105–114 (PDF pp. 110–119).
