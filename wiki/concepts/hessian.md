---
title: Hessian
type: concept
tags: [calculus, differentiation, multivariate, optimization]
difficulty: intermediate
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# Hessian

> **One-liner:** The Hessian is the table of a function's second slopes — it measures how the surface curves, i.e. how bowl-shaped or saddle-shaped it is.

## 🟢 Intuition (Dummy)
The [[gradient]] tells you which way is uphill. But is the ground curving up like
the inside of a bowl, down like a dome, or twisting like a saddle? That's
*curvature*, and the **Hessian** captures it. It is a square table of "how the
slope itself changes" in every pair of directions. Knowing the curvature lets an
optimizer take smarter steps — not just "go downhill" but "go downhill and the
valley is steep this way, shallow that way, so step accordingly."

## 🟡 Mechanics (Practitioner)
The **Hessian** collects all second-order [[partial-derivative|partial
derivatives]] of $f:\mathbb R^n\to\mathbb R$. For two variables (5.147):
$$
H=\nabla^2_{x,y}f=\begin{bmatrix}
\dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x\,\partial y}\\[10pt]
\dfrac{\partial^2 f}{\partial x\,\partial y} & \dfrac{\partial^2 f}{\partial y^2}
\end{bmatrix}.
$$
In general $H\in\mathbb R^{n\times n}$ with $H_{ij}=\partial^2 f/\partial x_i\partial x_j$.
For twice-continuously-differentiable $f$ the **Hessian is symmetric** because
mixed partials commute (5.146):
$$\frac{\partial^2 f}{\partial x\,\partial y}=\frac{\partial^2 f}{\partial y\,\partial x}.$$

**Worked example.** $f(x,y)=x^2+2xy+y^3$:
$$\frac{\partial^2 f}{\partial x^2}=2,\quad
\frac{\partial^2 f}{\partial x\partial y}=2,\quad
\frac{\partial^2 f}{\partial y^2}=6y
\ \Rightarrow\
H(1,2)=\begin{bmatrix}2&2\\2&12\end{bmatrix}.$$

**ML usage:** the Hessian is the second-order term of the multivariate
[[taylor-series]], $\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta$. It drives
**Newton's method** for optimization and the **Laplace approximation** (a
Gaussian fit to a posterior using the Hessian at the mode, §5.9). Its
eigenvalues classify critical points and set the conditioning of gradient
descent.

## 🔴 Theory (Expert)
- **Symmetry = Schwarz/Clairaut theorem.** Continuity of second partials forces
  $H=H^\top$, so $H$ has real eigenvalues and an orthonormal eigenbasis.
- **Curvature classification** at a critical point ($\nabla f=0$): $H\succ0$
  (positive definite) ⇒ local minimum; $H\prec0$ ⇒ local maximum; indefinite ⇒
  saddle point. This is the multivariate second-derivative test.
- **Vector fields.** If $\mathbf f:\mathbb R^n\to\mathbb R^m$, the Hessian becomes
  an $(m\times n\times n)$ tensor (one symmetric $n\times n$ slice per output).
- The Hessian is the Jacobian of the gradient: $H=\nabla(\nabla f)^\top$, linking
  it to the [[jacobian]] machinery of §5.3.

## Connections
- Builds on: [[partial-derivative]], [[gradient]]
- Leads to: [[taylor-series]] (2nd-order term), Newton's method (Ch. 7), Laplace approximation (Ch. 6)
- Used by: second-order optimizers, curvature analysis of loss surfaces
- People: [[ludwig-otto-hesse]]

## Sources
- `raw/mml-book.pdf` §5.7 (5.146–5.147), printed pp. 164–165 (PDF pp. 170–171).
