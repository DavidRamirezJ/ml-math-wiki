---
title: Spectral Theorem
type: concept
tags: [linear-algebra, matrix-decomposition]
difficulty: advanced
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Spectral Theorem

> **One-liner:** Every symmetric matrix can be fully described by a set of perpendicular stretch-directions with plain real stretch-factors — no rotation or complex weirdness needed.

## 🟢 Intuition (Dummy)
Symmetric matrices (the ones that look the same when you flip them across the diagonal) are the
"nicest" matrices there are. The **spectral theorem** promises that such a matrix's magic stretch
directions ([[eigenvalues-and-eigenvectors|eigenvectors]]) are not only real but also mutually
*perpendicular*, and every stretch factor (eigenvalue) is a real number. So a symmetric matrix is just
"stretch space by some real amounts along a clean set of right-angle axes" — nothing more
complicated. This is why symmetric matrices (like covariance matrices) are so well-behaved in ML.

## 🟡 Mechanics (Practitioner)
**Theorem (Thm 4.15):** if $A\in\mathbb R^{n\times n}$ is **symmetric** ($A=A^\top$), then there exists
an **orthonormal basis** of $\mathbb R^n$ consisting of eigenvectors of $A$, and every eigenvalue is
**real**.

**Direct consequence:** the [[eigendecomposition]] of a symmetric matrix always exists and can be
written with an *orthogonal* matrix $P$ (columns = orthonormal eigenvectors):
$$A=PDP^\top,\qquad P^\top P=I,\qquad D=\operatorname{diag}(\lambda_1,\dots,\lambda_n).$$
Because $P$ is orthogonal, $P^{-1}=P^\top$ — the inverse is free.

**Worked example (Example 4.8):** $A=\begin{pmatrix}3&2&2\\2&3&2\\2&2&3\end{pmatrix}$ has
$p_A(\lambda)=-(\lambda-1)^2(\lambda-7)$, so $\lambda_1=1$ (algebraic mult. $2$) and $\lambda_2=7$. The
eigenspace $E_1=\operatorname{span}\{(-1,1,0)^\top,(-1,0,1)^\top\}$ — but those two vectors aren't
orthogonal. The theorem guarantees an orthogonal basis *exists*; you construct it with
[[gram-schmidt-orthogonalization|Gram-Schmidt]] within $E_1$, yielding e.g. $(-1,1,0)^\top$ and $\tfrac12(-1,-1,2)^\top$, both
orthogonal to each other and to $E_7=\operatorname{span}\{(1,1,1)^\top\}$.

**ML usage:** covariance matrices, Gram matrices, and kernel matrices are all symmetric (and usually
positive semidefinite), so the spectral theorem is what makes [[principal-component-analysis]],
kernel methods, and spectral clustering work — it guarantees a real, orthogonal eigenbasis to project
onto.

## 🔴 Theory (Expert)
- **Why eigenvectors of distinct eigenvalues are orthogonal (for symmetric $A$):** if
  $A\mathbf x_i=\lambda_i\mathbf x_i$ and $A\mathbf x_j=\lambda_j\mathbf x_j$ with $\lambda_i\neq\lambda_j$,
  then $\lambda_i\mathbf x_i^\top\mathbf x_j=(A\mathbf x_i)^\top\mathbf x_j=\mathbf x_i^\top A\mathbf x_j=\lambda_j\mathbf x_i^\top\mathbf x_j$,
  forcing $\mathbf x_i^\top\mathbf x_j=0$.
- **Repeated eigenvalues:** within a single eigenspace any linear combination of eigenvectors is again
  an eigenvector ($A(\alpha\mathbf x_1+\beta\mathbf x_2)=\lambda(\alpha\mathbf x_1+\beta\mathbf x_2)$),
  so [[gram-schmidt-orthogonalization|Gram-Schmidt]] can orthonormalize each eigenspace independently — the geometric multiplicity of
  a symmetric matrix always equals its algebraic multiplicity (it is never [[eigenvalues-and-eigenvectors|defective]]).
- **Thm 4.21:** every symmetric matrix is therefore diagonalizable, with an *orthogonal* diagonalizer.
- The theorem underlies the construction of the [[singular-value-decomposition]]: $A^\top A$ and
  $AA^\top$ are symmetric positive semidefinite (Thm 4.14), so the spectral theorem supplies their
  orthonormal eigenbases — the right- and left-singular vectors.
- Generalizations: the complex version applies to Hermitian (and more broadly *normal*) matrices; the
  infinite-dimensional version is the spectral theorem for self-adjoint operators on Hilbert space.

## Connections
- Builds on: [[eigenvalues-and-eigenvectors]], [[gram-schmidt-orthogonalization|Gram-Schmidt]], [[orthogonal-matrix]], [[symmetric-positive-definite-matrix|positive definite matrix]]
- Leads to: [[eigendecomposition]], [[singular-value-decomposition]]
- Used by: [[principal-component-analysis]], kernel methods, spectral clustering, covariance analysis
- People: —

## Sources
- `raw/mml-book.pdf` §4.2 (Thm 4.14, Thm 4.15, Example 4.8), §4.4 (Thm 4.21), printed pp. 111–112, 117 (PDF pp. 116–117, 122).
