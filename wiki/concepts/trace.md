---
title: Trace
type: concept
tags: [linear-algebra, matrix-decomposition]
difficulty: beginner
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Trace

> **One-liner:** The trace is just the sum of the numbers running down a square matrix's main diagonal — a tiny summary that doesn't care which coordinate system you used.

## 🟢 Intuition (Dummy)
A square matrix has a "spine" — the entries from the top-left corner straight down to the
bottom-right. Add those up and you have the **trace**. It's one of the simplest summaries of a
matrix. Surprisingly, even though the individual diagonal numbers change if you describe the same
transformation in a different coordinate system, their *sum* stays the same. So the trace captures
something intrinsic about the transformation, not about your choice of axes.

## 🟡 Mechanics (Practitioner)
**Definition (Def 4.4):** for $A\in\mathbb R^{n\times n}$,
$$\operatorname{tr}(A):=\sum_{i=1}^n a_{ii}.$$

**Properties:**
- $\operatorname{tr}(A+B)=\operatorname{tr}(A)+\operatorname{tr}(B)$ and $\operatorname{tr}(\alpha A)=\alpha\operatorname{tr}(A)$ (linear).
- $\operatorname{tr}(I_n)=n$.
- $\operatorname{tr}(AB)=\operatorname{tr}(BA)$ for conformable $A,B$. (These four properties uniquely characterize the trace.)
- **Cyclic invariance:** $\operatorname{tr}(AKL)=\operatorname{tr}(KLA)$, generalizing to any number of factors.
- For vectors $\mathbf x,\mathbf y\in\mathbb R^n$: $\operatorname{tr}(\mathbf x\mathbf y^\top)=\mathbf y^\top\mathbf x$.

**Worked example:** $\operatorname{tr}\!\begin{pmatrix}4&2\\1&3\end{pmatrix}=4+3=7$ — and indeed the
eigenvalues of this matrix are $2$ and $5$, summing to $7$ (see Theorem 4.17).

**ML usage:** the trace appears all over machine learning — the squared Frobenius norm is
$\|A\|_F^2=\operatorname{tr}(A^\top A)$; expected values of quadratic forms reduce to traces
($\mathbb E[\mathbf x^\top A\mathbf x]=\operatorname{tr}(A\Sigma)+\boldsymbol\mu^\top A\boldsymbol\mu$);
and the "trace trick" with cyclic invariance simplifies many gradient derivations.

## 🔴 Theory (Expert)
- **Basis invariance:** if $B=S^{-1}AS$ is a change of basis ([[linear-mapping|similar matrices]]),
  then by cyclic invariance $\operatorname{tr}(B)=\operatorname{tr}(S^{-1}AS)=\operatorname{tr}(ASS^{-1})=\operatorname{tr}(A)$.
  Hence the trace of a linear map is well-defined independent of basis — making it, alongside the
  [[determinant]] and the [[eigenvalues-and-eigenvectors|eigenvalues]], a key invariant.
- **Spectral link (Thm 4.17):** $\operatorname{tr}(A)=\sum_{i=1}^n\lambda_i$, the sum of the
  eigenvalues (with multiplicity, in $\mathbb C$).
- The trace is the coefficient $(-1)^{n-1}$ times $c_{n-1}$ in the [[characteristic-polynomial]]
  $p_A(\lambda)$ (Eq. 4.24), while $\det(A)=c_0$ (Eq. 4.23).
- **Geometric intuition (Fig. 4.6):** for a $2\times 2$ matrix with orthonormal eigenvectors, the
  perimeter of the unit square scales by $\tfrac12(|\lambda_1|+|\lambda_2|)$ — the absolute
  eigenvalue sum governs how the perimeter changes, just as $|\lambda_1\lambda_2|$ governs area.

## Connections
- Builds on: [[matrix]]
- Leads to: [[characteristic-polynomial]], [[eigenvalues-and-eigenvectors]]
- Used by: Frobenius norm, expected quadratic forms, gradient identities
- People: —

## Sources
- `raw/mml-book.pdf` §4.1, Def 4.4 & Eqs 4.18–4.21, Thm 4.17, printed pp. 103–104, 112 (PDF pp. 108–109, 117).
