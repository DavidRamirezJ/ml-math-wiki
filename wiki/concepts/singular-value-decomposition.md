---
title: Singular Value Decomposition (SVD)
type: concept
tags: [linear-algebra, matrix-decomposition]
difficulty: advanced
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Singular Value Decomposition (SVD)

> **One-liner:** Any matrix at all — square or not — is really just "rotate, then stretch each axis by a positive amount, then rotate again."

## 🟢 Intuition (Dummy)
Take *any* [[matrix]], even a rectangular one. The **SVD** says whatever complicated thing it does can
be broken into three clean steps: (1) **spin** the input into a good set of perpendicular axes,
(2) **stretch** each of those axes by a fixed positive amount (and possibly add or drop dimensions),
(3) **spin** the result into a good set of output axes. The stretch amounts are the **singular
values** — they rank the directions from "most important" (biggest stretch) to "least." Because this
works for every matrix and always exists, it's been called the fundamental theorem of linear algebra.

## 🟡 Mechanics (Practitioner)
**Theorem 4.22 (SVD):** every $A\in\mathbb R^{m\times n}$ of rank $r$ factorizes as
$$A=U\Sigma V^\top,$$
- $U\in\mathbb R^{m\times m}$ **orthogonal**, columns $\mathbf u_i$ = **left-singular vectors**;
- $V\in\mathbb R^{n\times n}$ **orthogonal**, columns $\mathbf v_j$ = **right-singular vectors**;
- $\Sigma\in\mathbb R^{m\times n}$ "diagonal" (same shape as $A$, zero-padded) with
  $\Sigma_{ii}=\sigma_i\ge 0$, the **singular values**, ordered $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$.

**The singular value equation:** $A\mathbf v_i=\sigma_i\mathbf u_i$ — like the eigenvalue equation,
but the input and output vectors differ.

**How to compute it (Construction, §4.5.2):**
1. **Right-singular vectors / singular values:** diagonalize the symmetric PSD matrix
   $A^\top A=PDP^\top$. Then $V=P$ and $\sigma_i=\sqrt{\lambda_i}$ (the eigenvalues of $A^\top A$).
2. **Left-singular vectors:** normalize the images, $\mathbf u_i=\dfrac{1}{\sigma_i}A\mathbf v_i$.
   (Equivalently, the $\mathbf u_i$ are eigenvectors of $AA^\top$.)
3. Assemble $AV=U\Sigma \Rightarrow A=U\Sigma V^\top$.
   *(Note: this $A^\top A$ route is numerically poor; real libraries use dedicated SVD algorithms.)*

**Worked example (Example 4.13):** for $A=\begin{pmatrix}1&0&1\\-2&1&0\end{pmatrix}$,
$A^\top A$ has eigenvalues $6,1,0$, so $\sigma_1=\sqrt6,\ \sigma_2=1$ (rank $2$), giving
$\Sigma=\begin{pmatrix}\sqrt6&0&0\\0&1&0\end{pmatrix}$ and $U=\tfrac1{\sqrt5}\begin{pmatrix}1&2\\-2&1\end{pmatrix}$.

**ML usage:** the engine behind [[principal-component-analysis]], [[low-rank-approximation]] / data
compression, latent-semantic analysis / topic modeling, recommender systems (Example 4.14 factors a
movie-ratings matrix into "stereotypical movies" $\mathbf u_i$ and "stereotypical viewers"
$\mathbf v_j$), the Moore–Penrose pseudo-inverse for least squares, and decorrelating design matrices.

## 🔴 Theory (Expert)
- **Existence:** $A^\top A$ is symmetric positive semidefinite (Thm 4.14), so by the
  [[spectral-theorem]] it has an orthonormal eigenbasis with eigenvalues $\lambda_i\ge 0$ — this
  furnishes $V$ and $\sigma_i=\sqrt{\lambda_i}$, guaranteeing the SVD for *any* matrix.
- **Orthogonality of images:** for orthonormal eigenvectors $\mathbf v_i,\mathbf v_j$,
  $(A\mathbf v_i)^\top(A\mathbf v_j)=\mathbf v_i^\top A^\top A\,\mathbf v_j=\lambda_j\,\mathbf v_i^\top\mathbf v_j=0$,
  so the normalized images $\mathbf u_i$ are orthonormal.
- **Relation to eigendecomposition (§4.5.3):**
  - left-singular vectors = eigenvectors of $AA^\top$; right-singular vectors = eigenvectors of $A^\top A$;
  - nonzero $\sigma_i$ = $\sqrt{\text{nonzero eigenvalues of }A^\top A\text{ (and }AA^\top)}$;
  - $U,V$ are *always* orthonormal (true rotations); the [[eigendecomposition]]'s $P$ need not be;
  - for symmetric positive definite $A$, setting $U=V=P$ and $\Sigma=D$ makes the SVD *equal to* the
    eigendecomposition.
- **Null space & rank:** when $m<n$, the right-singular vectors with $\sigma_i=0$ (i.e. $A\mathbf v_i=\mathbf 0$)
  form an orthonormal basis of $\ker(A)$. The SVD exposes rank, range, and null space simultaneously.
- **Conventions:** the form above is the **full SVD**; the **reduced/economy SVD** keeps only $r$
  (or $\min(m,n)$) columns so $\Sigma$ is square diagonal; the **truncated SVD** keeps the top $k$ —
  see [[low-rank-approximation]].

## Connections
- Builds on: [[eigenvalues-and-eigenvectors]], [[spectral-theorem]], [[orthogonal-matrix]], [[basis-and-rank]], [[eigendecomposition]]
- Leads to: [[low-rank-approximation]], pseudo-inverse, [[principal-component-analysis]]
- Used by: [[principal-component-analysis]], recommender systems, topic modeling, least squares, image compression
- People: —

## Sources
- `raw/mml-book.pdf` §4.5, Thm 4.22, Examples 4.12–4.14, §§4.5.1–4.5.3, printed pp. 119–128 (PDF pp. 124–133).
