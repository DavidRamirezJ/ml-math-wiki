---
title: Gradient
type: concept
tags: [calculus, differentiation, multivariate, optimization]
difficulty: intermediate
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# Gradient

> **One-liner:** The gradient is the list of all a function's partial slopes, bundled into one vector that points in the direction the function rises fastest.

## 🟢 Intuition (Dummy)
Stand somewhere on a hilly landscape. There is one direction that is *steepest
uphill*. The **gradient** is an arrow pointing exactly that way, and its length
tells you how steep the climb is. If you want to get *down* the hill fast (which
is what training a model does — going downhill on an error landscape), you walk
opposite the gradient. That single idea, "take a step against the gradient,"
powers almost all of machine learning's learning.

## 🟡 Mechanics (Practitioner)
For $f:\mathbb R^n\to\mathbb R$, the **gradient** collects every
[[partial-derivative]] into a **row vector** (5.40):
$$
\nabla_{\mathbf x} f = \frac{df}{d\mathbf x}
= \Big[\ \tfrac{\partial f}{\partial x_1}\ \ \tfrac{\partial f}{\partial x_2}\ \cdots\ \tfrac{\partial f}{\partial x_n}\ \Big]\in\mathbb R^{1\times n}.
$$
(The book uses the **numerator layout**, so the gradient of a scalar function is
a *row* vector — this makes the [[chain-rule]] literal matrix multiplication.)

**Worked example.** $f(x_1,x_2)=x_1^2 x_2 + x_1 x_2^3$:
$$\nabla f = \big[\,2x_1 x_2 + x_2^3\ \ \ x_1^2 + 3x_1 x_2^2\,\big]\in\mathbb R^{1\times2}.$$

**Useful gradient identities (5.99–5.108)** that recur in ML (constant $\mathbf a$, matrix $B$):
$$
\frac{\partial \mathbf a^\top\mathbf x}{\partial \mathbf x}=\mathbf a^\top,\qquad
\frac{\partial \mathbf x^\top B\mathbf x}{\partial \mathbf x}=\mathbf x^\top(B+B^\top),\qquad
\frac{\partial}{\partial X}\operatorname{tr}(f(X))=\operatorname{tr}\!\Big(\tfrac{\partial f(X)}{\partial X}\Big).
$$

**ML usage — least-squares gradient (Example 5.11).** For a linear model
$\mathbf y=\Phi\boldsymbol\theta$ with loss $L=\lVert\mathbf e\rVert^2$,
$\mathbf e=\mathbf y-\Phi\boldsymbol\theta$, the chain rule gives
$$\frac{\partial L}{\partial\boldsymbol\theta}
=\frac{\partial L}{\partial\mathbf e}\frac{\partial\mathbf e}{\partial\boldsymbol\theta}
= (2\mathbf e^\top)(-\Phi) = -2(\mathbf y-\Phi\boldsymbol\theta)^\top\Phi\in\mathbb R^{1\times D}.$$
Setting this to zero yields the normal equations of linear regression (Ch. 9).

## 🔴 Theory (Expert)
- The gradient is the matrix of the best **linear approximation** of $f$ at a
  point: $f(\mathbf x)\approx f(\mathbf x_0)+(\nabla_{\mathbf x}f)(\mathbf x_0)(\mathbf x-\mathbf x_0)$
  (5.148) — the first-order [[taylor-series]] term.
- **Steepest ascent.** Among all unit directions, $\nabla f/\lVert\nabla f\rVert$
  maximizes the directional derivative $\nabla f\cdot \mathbf u$; this is why
  $-\nabla f$ is the steepest-descent direction used by gradient descent (Ch. 7).
- The gradient of $f:\mathbb R^n\to\mathbb R$ is the special $m=1$ case of the
  [[jacobian]] (a $1\times n$ matrix). Going to vector-valued $f$ promotes it to
  a full matrix; going to matrix arguments (§5.4) promotes it to a **gradient
  tensor** with entries $\partial A_{ij}/\partial B_{kl}$.
- The chain rule and these identities (Petersen & Pedersen, 2012) let you
  differentiate complicated objectives symbolically; at scale this is automated
  by [[automatic-differentiation]].

## Connections
- Builds on: [[partial-derivative]], [[derivative]]
- Leads to: [[jacobian]], [[hessian]], [[taylor-series]]
- Used by: gradient descent (Ch. 7), [[backpropagation]], linear regression (Ch. 9)
- People: —

## Sources
- `raw/mml-book.pdf` §5.2 (5.40), §5.5 (identities 5.99–5.108), Example 5.11, printed pp. 146–148, 153–158 (PDF pp. 152–154, 159–164).
