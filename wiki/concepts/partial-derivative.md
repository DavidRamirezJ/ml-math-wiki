---
title: Partial Derivative
type: concept
tags: [calculus, differentiation, multivariate]
difficulty: beginner
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# Partial Derivative

> **One-liner:** A partial derivative measures how a multi-input function changes when you nudge just one input and freeze all the others.

## 🟢 Intuition (Dummy)
A function of many variables is like a landscape: your height depends on how far
east *and* how far north you stand. To ask "how steep is it?" you first have to
pick a direction. A **partial derivative** picks the simplest directions: walk
*only* east (north frozen) and measure the slope — that's the partial derivative
with respect to east. Walk *only* north and you get the other one. Each partial
derivative is just an ordinary [[derivative]] in disguise, with every other
variable temporarily treated as a constant.

## 🟡 Mechanics (Practitioner)
For $f:\mathbb R^n\to\mathbb R$ with inputs $x_1,\dots,x_n$, the partial
derivative w.r.t. $x_i$ (Def 5.5) holds the rest fixed:
$$\frac{\partial f}{\partial x_i}=\lim_{h\to0}\frac{f(x_1,\dots,x_i+h,\dots,x_n)-f(\mathbf x)}{h}.$$
Operationally: differentiate as usual treating $x_i$ as the variable and every
other $x_j$ as a constant. All the ordinary [[derivative|differentiation rules]]
still apply.

**Worked example.** $f(x_1,x_2)=x_1^2 x_2 + x_1 x_2^3$:
$$\frac{\partial f}{\partial x_1}=2x_1 x_2 + x_2^3,\qquad
\frac{\partial f}{\partial x_2}=x_1^2 + 3x_1 x_2^2.$$

Collecting *all* the partials gives the [[gradient]]. For example with the chain
rule, $f(x,y)=(x+2y^3)^2$ gives $\partial f/\partial x = 2(x+2y^3)$ and
$\partial f/\partial y = 12(x+2y^3)y^2$.

**ML usage:** loss functions depend on many parameters at once; the partial
derivative w.r.t. each parameter says how to tweak that one parameter to reduce
the loss. Bundled together they are the gradient used by optimizers.

## 🔴 Theory (Expert)
- Existence of all partial derivatives at a point does **not** by itself imply
  differentiability (i.e. existence of a good linear approximation) — that
  needs the partials to also be continuous near the point. The book works under
  smoothness assumptions where this subtlety does not bite.
- **Higher-order partials** stack: $\frac{\partial^2 f}{\partial y\,\partial x}$
  means differentiate by $x$ first, then $y$. For twice-continuously-
  differentiable $f$, order does not matter (Schwarz's theorem, 5.146), which is
  exactly why the [[hessian]] is symmetric.
- Partials are the entries of every higher object in this chapter: the
  [[gradient]] (a row of them), the [[jacobian]] (a matrix of them), the
  [[hessian]] (second-order ones), and the gradient *tensors* of §5.4
  ($\partial A_{ij}/\partial B_{kl}$).

## Connections
- Builds on: [[derivative]]
- Leads to: [[gradient]], [[jacobian]], [[hessian]]
- Used by: [[chain-rule]], [[backpropagation]], gradient descent (Ch. 7)
- People: —

## Sources
- `raw/mml-book.pdf` §5.2 (Def 5.5), §5.7, printed pp. 146–147, 164–165 (PDF pp. 152–153, 170–171).
