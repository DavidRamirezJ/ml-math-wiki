---
title: Orthogonal Matrix
type: concept
tags: [analytic-geometry, linear-algebra, inner-product]
difficulty: intermediate
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Orthogonal Matrix

> **One-liner:** An orthogonal matrix is a square matrix whose columns are mutually perpendicular unit vectors — it rotates or reflects space without stretching it.

## 🟢 Intuition (Dummy)
Some matrices stretch, squash, or shear space. An orthogonal matrix does none of that: it only
rotates or flips. Lengths stay the same, angles stay the same — it is a "rigid motion." Because it
never distorts, undoing it is trivially easy: you just transpose it (flip it across its diagonal).

## 🟡 Mechanics (Practitioner)
**Definition 3.8.** A square matrix $A\in\mathbb R^{n\times n}$ is **orthogonal** iff its columns
are orthonormal, equivalently
$$ A A^\top = I = A^\top A \quad\Longrightarrow\quad A^{-1} = A^\top. $$
The inverse is just the transpose — no computation needed.

**Length and angle preservation (dot product).**
$$ \|Ax\|^2 = (Ax)^\top(Ax) = x^\top A^\top A x = x^\top I x = \|x\|^2, $$
$$ \cos\angle(Ax,Ay) = \frac{(Ax)^\top(Ay)}{\|Ax\|\,\|Ay\|} = \frac{x^\top A^\top A y}{\|x\|\,\|y\|} = \frac{x^\top y}{\|x\|\,\|y\|} = \cos\angle(x,y). $$

**Worked example.** $A=\frac1{\sqrt2}\begin{bmatrix}1 & -1\\ 1 & 1\end{bmatrix}$ has orthonormal
columns ($\|c_1\|=\|c_2\|=1$, $c_1^\top c_2=0$); $A^\top A = I$, so $A^{-1}=A^\top$. It is a
$45^\circ$ rotation.

**ML usage.** Orthogonal matrices keep transformations numerically stable (they preserve norms, so
errors do not blow up). They appear as the $U,V$ factors in the **SVD** (Ch. 4), in **QR**
decomposition for least squares, and as orthogonal weight initializations in deep networks.

## 🔴 Theory (Expert)
- The terminology is conventional but loose: the columns are *orthonormal*, so "orthonormal matrix"
  would be more precise (Remark, p. 78).
- $\det(A) = \pm 1$. Those with $\det = +1$ are **[[rotation|rotations]]** (the special orthogonal
  group $SO(n)$); those with $\det = -1$ include a **reflection**.
- Orthogonal matrices form a group under multiplication, the **orthogonal group** $O(n)$ — closed
  under products and inverses, which is why composing rigid motions stays rigid.
- Preservation of the inner product means they are exactly the **isometries** of Euclidean space
  fixing the origin.

## Connections
- Builds on: [[matrix]], [[orthogonality]], [[orthonormal-basis]]
- Leads to: [[rotation]], SVD / QR decomposition (Ch. 4)
- Used by: rotations, reflections, stable numerical linear algebra
- People: —

## Sources
- `raw/mml-book.pdf` §3.4 (Definition 3.8, Eqs 3.29–3.32), printed pp. 77–78 (PDF pp. 82–83).
