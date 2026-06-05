---
title: Inner Product
type: concept
tags: [analytic-geometry, inner-product, foundations]
difficulty: intermediate
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Inner Product

> **One-liner:** An inner product is a way to multiply two vectors into a single number that tells you how much they point in the same direction.

## 🟢 Intuition (Dummy)
Two arrows can point the same way, opposite ways, or at right angles. An inner product squeezes
that relationship into one number: big and positive when they agree, zero when they're
perpendicular, negative when they oppose. Once you can do this, you get *everything* geometric for
free — the length of an arrow is its inner product with itself, the angle between two arrows comes
from their inner product, and "perpendicular" just means "inner product equals zero." It is the
single tool that turns a bag of vectors into a space with real geometry.

## 🟡 Mechanics (Practitioner)
Start from the familiar **dot product** on $\mathbb R^n$:
$$ x^\top y = \sum_{i=1}^n x_i y_i. $$

An inner product generalizes this. A mapping $\Omega : V\times V \to \mathbb R$ is **bilinear**
(linear in each argument), and is an **inner product** $\langle x,y\rangle$ if it is also
(Defs 3.2–3.3):

- **Symmetric:** $\langle x,y\rangle = \langle y,x\rangle$
- **Positive definite:** $\langle x,x\rangle > 0$ for $x\ne \mathbf 0$, and $\langle \mathbf 0,\mathbf 0\rangle = 0$

$(V,\langle\cdot,\cdot\rangle)$ is an **inner product space**; with the dot product it is a
**Euclidean vector space**. Every inner product **induces a norm** $\|x\|=\sqrt{\langle x,x\rangle}$
([[norm]]) and hence distances ([[metric]]) and [[angle-between-vectors|angles]].

**Matrix form (Thm 3.5).** Relative to an ordered basis $B$, with coordinates $\hat x,\hat y$,
$$ \langle x,y\rangle = \hat x^\top A\,\hat y, \qquad A_{ij} := \langle b_i,b_j\rangle, $$
where $A$ is a [[symmetric-positive-definite-matrix]]. Conversely every such $A$ defines an inner product.

**Worked example (non-dot inner product, §3.2).** On $\mathbb R^2$,
$\langle x,y\rangle = x_1y_1 - (x_1y_2 + x_2y_1) + 2x_2y_2$ is a valid inner product
(symmetric matrix $A=\big[\begin{smallmatrix}1 & -1\\ -1 & 2\end{smallmatrix}\big]$, positive definite).
It gives different lengths and angles than the dot product.

**ML usage.** Inner products measure **similarity** (cosine similarity, attention scores). The
"kernel trick" replaces $\langle x,y\rangle$ with $k(x,y)$, an inner product in a higher-dimensional
feature space — the foundation of SVMs and Gaussian processes (Ch. 12).

## 🔴 Theory (Expert)
- **Cauchy–Schwarz inequality:** $|\langle x,y\rangle| \le \|x\|\,\|y\|$ — the guarantee that
  $\frac{\langle x,y\rangle}{\|x\|\|y\|}\in[-1,1]$, which is what makes the angle definition
  well-posed. See [[augustin-louis-cauchy]] and [[hermann-schwarz]].
- **Inner product of functions (§3.7):** for $u,v:\mathbb R\to\mathbb R$,
  $\langle u,v\rangle := \int_a^b u(x)v(x)\,dx$. With $a=-\pi,b=\pi$, $\sin$ and $\cos$ are orthogonal,
  and $\{1,\cos x,\cos 2x,\dots\}$ is an orthogonal set — the basis of **Fourier series**. Making this
  rigorous (measures, convergence) leads to **Hilbert spaces**.
- Positive definiteness of the Gram matrix $A_{ij}=\langle b_i,b_j\rangle$ is exactly the
  **Mercer condition** that defines valid kernels; positive *semi*definiteness suffices for kernels
  (§12.4).

## Connections
- Builds on: [[vector-space]], [[linear-mapping]] (bilinearity), [[norm]]
- Leads to: [[symmetric-positive-definite-matrix]], [[metric]], [[angle-between-vectors]], [[orthogonality]], [[orthogonal-projection]]
- Used by: kernel methods / SVM (Ch. 12), Gaussian processes, cosine similarity
- People: [[augustin-louis-cauchy]], [[hermann-schwarz]]

## Sources
- `raw/mml-book.pdf` §3.2 (Defs 3.2–3.3), §3.3 (Cauchy–Schwarz 3.17), §3.7, printed pp. 72–75, 80–81 (PDF pp. 77–80, 85–86).
