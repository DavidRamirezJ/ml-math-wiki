---
title: Orthogonal Projection
type: concept
tags: [analytic-geometry, linear-algebra, inner-product, ml-method]
difficulty: advanced
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Orthogonal Projection

> **One-liner:** An orthogonal projection drops a vector straight down onto a subspace, finding the closest point in that subspace.

## 🟢 Intuition (Dummy)
Stand a flagpole on a sunny day; its shadow on the ground is a projection. The orthogonal
projection of a vector onto a flat surface is the point on that surface directly "below" it — the
*closest* point, found by dropping a perpendicular line. The leftover (the line from the shadow up
to the tip) is exactly perpendicular to the surface. This is how we squash high-dimensional data
into fewer dimensions while losing as little as possible: keep the shadow, discard the height.

## 🟡 Mechanics (Practitioner)
**Definition 3.10.** A linear map $\pi:V\to U$ onto a subspace $U\subseteq V$ is a **projection**
if $\pi^2 = \pi$. Its matrix $P_\pi$ satisfies $P_\pi^2 = P_\pi$.

**Onto a line** spanned by $b$ (dot product). The projection of $x$ is $\pi_U(x)=\lambda b$ where
the orthogonality condition $\langle x-\lambda b, b\rangle = 0$ gives
$$ \lambda = \frac{b^\top x}{\|b\|^2}, \qquad \pi_U(x) = \frac{b^\top x}{\|b\|^2}\,b, \qquad P_\pi = \frac{b b^\top}{\|b\|^2}. $$
$P_\pi$ is symmetric of rank 1. (If $\|b\|=1$, simply $\lambda = b^\top x$.)

**Onto a general subspace** with basis matrix $B=[b_1,\dots,b_m]\in\mathbb R^{n\times m}$.
The displacement $x-B\lambda$ must be orthogonal to every basis vector, $B^\top(x-B\lambda)=\mathbf 0$,
giving the **normal equation**
$$ \boxed{\,B^\top B\,\lambda = B^\top x\,} \ \Longrightarrow\ \lambda = (B^\top B)^{-1}B^\top x, $$
$$ \pi_U(x) = B(B^\top B)^{-1}B^\top x, \qquad P_\pi = B(B^\top B)^{-1}B^\top. $$
The matrix $(B^\top B)^{-1}B^\top$ is the **pseudo-inverse** of $B$ (exists when $B$ has full column
rank, i.e. $B^\top B$ is a [[symmetric-positive-definite-matrix]]). If $B$ is an
[[orthonormal-basis|ONB]] then $B^\top B = I$ and everything simplifies to $\lambda = B^\top x$,
$\pi_U(x) = BB^\top x$.

**Worked example (§3.8.2).** $U=\operatorname{span}\big[\,[1,1,1]^\top,\ [0,1,2]^\top\,\big]$,
$x=[6,0,0]^\top$. With $B=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix}$:
$B^\top B = \begin{bmatrix}3&3\\3&5\end{bmatrix}$, $B^\top x = \begin{bmatrix}6\\0\end{bmatrix}$,
solving gives $\lambda=\begin{bmatrix}5\\-3\end{bmatrix}$, so
$\pi_U(x) = B\lambda = [5,2,-1]^\top$. The **projection error** is
$\|x-\pi_U(x)\| = \|[1,-2,1]^\top\| = \sqrt6$.

**Onto an affine subspace** $L = x_0 + U$ (§3.8.4): shift, project, shift back,
$$ \pi_L(x) = x_0 + \pi_U(x - x_0). $$

**ML usage — the method this enables.** When $Ax=b$ has *no* solution ($b\notin\operatorname{col}(A)$),
the **least-squares solution** projects $b$ onto $\operatorname{col}(A)$, minimizing
$\|Ax-b\|$ — exactly the normal equation $A^\top A x = A^\top b$. This is the geometry of
**linear regression** (Ch. 9) and of PCA's reconstruction error (Ch. 10); projections also build
SVM separating hyperplanes (Ch. 12).

## 🔴 Theory (Expert)
- "Closest" is minimal distance ([[metric]]); the optimum is characterized by orthogonality of the
  residual to $U$ — the geometric content of the normal equation.
- A projection matrix is **idempotent** ($P^2=P$); an *orthogonal* projection is additionally
  **symmetric** ($P^\top=P$). Its eigenvalues are $0$ and $1$: $\pi_U(x)$ is an eigenvector with
  eigenvalue 1 (Remark, p. 85), and $I-P$ projects onto the [[orthogonal-complement]] $U^\perp$.
- In practice a "jitter"/ridge term $\epsilon I$ is added to $B^\top B$ for numerical stability and
  guaranteed positive definiteness; this $\ell_2$ regularization is derived rigorously via Bayesian
  inference (Ch. 9).
- [[gram-schmidt-orthogonalization|Gram–Schmidt]] is built from repeated projections.

## Connections
- Builds on: [[inner-product]], [[orthogonality]], [[symmetric-positive-definite-matrix]], [[basis-and-rank]], [[system-of-linear-equations]]
- Leads to: [[gram-schmidt-orthogonalization]], least squares / linear regression (Ch. 9), PCA (Ch. 10), SVM (Ch. 12)
- Used by: dimensionality reduction, data compression, separating hyperplanes
- People: —

## Sources
- `raw/mml-book.pdf` §3.8 (Def 3.10, Eqs 3.39–3.72, Examples 3.10–3.11), printed pp. 81–90 (PDF pp. 86–95).
