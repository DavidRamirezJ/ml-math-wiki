---
title: Characteristic Polynomial
type: concept
tags: [linear-algebra, matrix-decomposition]
difficulty: intermediate
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Characteristic Polynomial

> **One-liner:** It's a special polynomial built from a square matrix whose roots are exactly the matrix's eigenvalues.

## 🟢 Intuition (Dummy)
Suppose you want to find the special directions a matrix only stretches (its
[[eigenvalues-and-eigenvectors|eigenvectors]]) and the stretch factors (its eigenvalues). The
**characteristic polynomial** is a single recipe that turns this hunt into an ordinary
"find-the-roots" problem you already know from algebra. You build the polynomial from the matrix, set
it to zero, solve, and the answers are the eigenvalues. It's the bridge between "matrix" and
"high-school polynomial."

## 🟡 Mechanics (Practitioner)
**Definition (Def 4.5):** for $A\in\mathbb R^{n\times n}$ and a scalar $\lambda$,
$$p_A(\lambda):=\det(A-\lambda I)=c_0+c_1\lambda+\dots+c_{n-1}\lambda^{n-1}+(-1)^n\lambda^n.$$
It is a degree-$n$ polynomial in $\lambda$. Two coefficients are easy to read off:
$$c_0=\det(A),\qquad c_{n-1}=(-1)^{n-1}\operatorname{tr}(A).$$

**Why it works:** $\lambda$ is an eigenvalue $\iff (A-\lambda I)\mathbf x=\mathbf 0$ has a nonzero
solution $\iff A-\lambda I$ is *not* invertible $\iff \det(A-\lambda I)=0$.

**Worked example:** for $A=\begin{pmatrix}4&2\\1&3\end{pmatrix}$,
$$p_A(\lambda)=\det\!\begin{pmatrix}4-\lambda&2\\1&3-\lambda\end{pmatrix}=(4-\lambda)(3-\lambda)-2=\lambda^2-7\lambda+10=(\lambda-2)(\lambda-5),$$
so the eigenvalues are $\lambda_1=2,\ \lambda_2=5$. (Check: $c_0=10=\det A$, $-c_1=7=\operatorname{tr}A$.)

**ML usage:** mostly a conceptual/derivational tool. Computing eigenvalues by rooting the
characteristic polynomial is numerically unstable for large $n$, so practical libraries use iterative
QR-type algorithms instead — but the polynomial underpins why eigenvalues exist and how
[[determinant]] and [[trace]] relate to the spectrum.

## 🔴 Theory (Expert)
- **Thm 4.8:** $\lambda\in\mathbb R$ (or $\mathbb C$) is an eigenvalue of $A$ $\iff \lambda$ is a root
  of $p_A$. The number of times a root repeats is its **algebraic multiplicity** (Def 4.9).
- Because $p_A$ has degree $n$, an $n\times n$ matrix has exactly $n$ eigenvalues counted with
  algebraic multiplicity over $\mathbb C$ (fundamental theorem of algebra). Real matrices can have
  complex-conjugate eigenvalues (e.g. rotation matrices, $A_3$ in Fig. 4.4).
- The polynomial is **basis-invariant**: [[linear-mapping|similar matrices]] $A$ and $S^{-1}AS$ share
  the same characteristic polynomial, hence the same eigenvalues, determinant, and trace.
- Factoring $p_A(\lambda)=(-1)^n\prod_i(\lambda-\lambda_i)$ recovers **Thm 4.16**
  $\det A=\prod\lambda_i$ (from $c_0$) and **Thm 4.17** $\operatorname{tr}A=\sum\lambda_i$ (from $c_{n-1}$).
- The Cayley–Hamilton theorem (beyond this chapter) states every matrix satisfies its own
  characteristic polynomial: $p_A(A)=0$.

## Connections
- Builds on: [[determinant]], [[trace]], [[matrix]]
- Leads to: [[eigenvalues-and-eigenvectors]], [[eigendecomposition]]
- Used by: [[eigendecomposition]], spectral analysis, stability analysis
- People: —

## Sources
- `raw/mml-book.pdf` §4.1 (Def 4.5), §4.2 (Thm 4.8, Def 4.9, Example 4.5), printed pp. 104–107 (PDF pp. 109–112).
