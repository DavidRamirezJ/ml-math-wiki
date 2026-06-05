---
title: Chain Rule
type: concept
tags: [calculus, differentiation, multivariate]
difficulty: intermediate
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# Chain Rule

> **One-liner:** The chain rule says: to differentiate a function inside a function, multiply the outer slope by the inner slope.

## 🟢 Intuition (Dummy)
Suppose gears are linked: turning gear A turns gear B twice as fast, and gear B
turns gear C three times as fast. Then turning A turns C $2\times3=6$ times as
fast. The **chain rule** is exactly this "multiply the rates" idea for functions
stacked inside one another. If $y$ depends on $u$, and $u$ depends on $x$, then
how fast $y$ changes with $x$ is just (how fast $y$ changes with $u$) times (how
fast $u$ changes with $x$). It is the single most important rule in this chapter
— it is what makes training deep networks possible.

## 🟡 Mechanics (Practitioner)
**Univariate (5.32):** for $g\circ f$,
$$\big(g(f(x))\big)' = g'(f(x))\,f'(x).$$

**Multivariate (5.48):** for $\mathbf x\in\mathbb R^n$,
$$\frac{\partial}{\partial\mathbf x}(g\circ f)(\mathbf x)
=\frac{\partial g}{\partial f}\,\frac{\partial f}{\partial\mathbf x}.$$
This is a **product of [[jacobian|Jacobians]]**. The "denominator" of the first
factor matches the "numerator" of the second, so the inner dimension cancels
just like in matrix multiplication — which works cleanly *because* gradients are
written as row vectors (numerator layout).

**Worked example — $f$ of $t$ through $x_1,x_2$ (5.49).** If
$f(x_1,x_2)$ and $x_1(t),x_2(t)$, then
$$\frac{df}{dt}=\frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial t}
+\frac{\partial f}{\partial x_2}\frac{\partial x_2}{\partial t}.$$
For $f=x_1^2+2x_2$, $x_1=\sin t$, $x_2=\cos t$:
$$\frac{df}{dt}=2\sin t\cos t - 2\sin t = 2\sin t(\cos t - 1).$$

**Two inner variables $(s,t)$ — matrix form (5.53):**
$$\frac{df}{d(s,t)}=\frac{\partial f}{\partial\mathbf x}\frac{\partial\mathbf x}{\partial(s,t)}
=\big[\tfrac{\partial f}{\partial x_1}\ \tfrac{\partial f}{\partial x_2}\big]
\begin{bmatrix}\frac{\partial x_1}{\partial s}&\frac{\partial x_1}{\partial t}\\[2pt]\frac{\partial x_2}{\partial s}&\frac{\partial x_2}{\partial t}\end{bmatrix}.$$

**ML usage:** the chain rule is the engine of [[backpropagation]] — it turns the
gradient of a deep loss into a product of per-layer Jacobians, computed
efficiently by [[automatic-differentiation]].

## 🔴 Theory (Expert)
- The multivariate chain rule is a statement about composing **linear
  approximations**: the best linear map approximating $g\circ f$ at a point is
  the composition (product) of the linear maps approximating $g$ and $f$ — hence
  the product of Jacobians.
- **Why row vectors matter.** With the numerator layout the chain rule needs no
  transposes; for tensor-valued derivatives (gradients of matrices, §5.4) the
  transpose is no longer trivial, so consistent layout prevents bugs.
- **Deep composition (5.111).** For $y=(f_K\circ\cdots\circ f_1)(\mathbf x)$,
  $\frac{\partial L}{\partial\theta_i}=\frac{\partial L}{\partial f_K}\frac{\partial f_K}{\partial f_{K-1}}\cdots\frac{\partial f_{i+1}}{\partial\theta_i}$;
  associativity of matrix multiplication lets you choose the *order* of
  multiplication — left-to-right (forward mode) or right-to-left (reverse mode /
  [[backpropagation]]), with very different costs.
- A note of caution the book repeats: $\partial f/\partial x$ is *not* a fraction;
  the "cancellation" is a mnemonic, not algebra.

## Connections
- Builds on: [[derivative]], [[partial-derivative]], [[jacobian]]
- Leads to: [[backpropagation]], [[automatic-differentiation]]
- Used by: deep networks, [[gradient]] of composite losses (Example 5.11)
- People: —

## Sources
- `raw/mml-book.pdf` §5.1.2 (5.32), §5.2.1–5.2.2 (5.46–5.53), printed pp. 145–148 (PDF pp. 151–154).
