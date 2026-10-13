---
title: Chain Rule
type: concept
tags: [calculus, differentiation, multivariate]
difficulty: intermediate
sources: [mml-ch05-vector-calculus, llm-appA-introduction-to-pytorch, llm-ch04-implementing-a-gpt-model]
created: 2026-06-04
updated: 2026-10-13
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

**In LLMs.** Raschka's one-sentence version: backpropagation is "an
implementation of the chain rule from calculus for neural networks" (book
p. 263). A GPT model is a long line of gears — embedding, 12 transformer
blocks, output layer, loss. To learn how much a weight deep inside block 1
affects the loss, you multiply the "gear ratios" of every stage between that
weight and the loss. And the book's good news: you only need this high-level
picture; PyTorch does the multiplying for you.

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

### In LLMs (Raschka, Appendix A and Ch. 4)

**The chain rule on a computation graph (Fig. A.8, §A.4, p. 263).** For the
one-neuron [[logistic-regression]] graph $u=w_1x_1$, $z=u+b$, $a=\sigma(z)$,
$\text{loss}=L(a,y)$, the loss is a composition $L(a(z(u(w_1))))$, so
$$\frac{\partial L}{\partial w_1}=\frac{\partial u}{\partial w_1}\times\frac{\partial z}{\partial u}\times\frac{da}{dz}\times\frac{\partial L}{\partial a}.$$
Symbol by symbol:
- $\tfrac{\partial L}{\partial a}$ — how the loss moves when the output $a$ moves. For [[binary-cross-entropy]] with $y=1$ it is $-1/a$.
- $\tfrac{da}{dz}$ — slope of the [[sigmoid-function]]: $\sigma(z)(1-\sigma(z))$.
- $\tfrac{\partial z}{\partial u}=1$ — adding $b$ doesn't change the slope.
- $\tfrac{\partial u}{\partial w_1}=x_1$ — the weight is multiplied by the input.

**Plugging in the book's numbers** ($x_1=1.1$, $w_1=2.2$, $b=0$, $y=1$; Listing A.3, p. 264):

| factor | value |
|---|---|
| $a=\sigma(2.42)$ | $\approx0.9183$ |
| $\partial L/\partial a=-1/a$ | $\approx-1.0889$ |
| $da/dz=a(1-a)$ | $\approx0.0750$ |
| $\partial z/\partial u$ | $1$ |
| $\partial u/\partial w_1=x_1$ | $1.1$ |
| **product $=\partial L/\partial w_1$** | $\approx1.1\times1\times0.0750\times(-1.0889)\approx\mathbf{-0.0898}$ |

Swap the last link for $\partial z/\partial b=1$ and you get
$\partial L/\partial b\approx\mathbf{-0.0817}$. Both match PyTorch's printout
exactly. (The book prints only the two final numbers; the intermediate
factors are our hand check.) A neat shortcut hides inside:
$\tfrac{\partial L}{\partial a}\tfrac{da}{dz}=-\tfrac1a\cdot a(1-a)=a-y$ when $y=1$.

**Right to left = reverse mode.** The caption of Fig. A.8 says the common way
is to apply the chain rule "from right to left" — start at the loss, walk
back to the parameters. This is **reverse-mode
[[automatic-differentiation]]**, i.e. [[backpropagation]].

**Deep networks: many links, small factors (§4.4, pp. 109–113).** In a stack
of layers, the gradient for an early layer is a *long* product of per-layer
factors. If many factors are small, the product shrinks toward zero — the
**vanishing gradient** problem ([[vanishing-and-exploding-gradients]]). The
book shows it on a 5-layer toy network: the first layer's mean absolute
gradient is only $0.0002$, versus $0.005$ in the last layer. Adding
[[shortcut-connections]] raises the first layer to $0.22$.

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

**In LLMs (deeper view).**
- **Why residuals rescue the chain.** For a residual layer
  $\mathbf x_{\ell+1}=\mathbf x_\ell+f_\ell(\mathbf x_\ell)$, the chain-rule factor is
  $$\frac{\partial\mathbf x_{\ell+1}}{\partial\mathbf x_\ell}=I+\frac{\partial f_\ell}{\partial\mathbf x_\ell}.$$
  Multiplying $L$ such factors expands into a sum that always contains the
  pure identity path $I\cdot I\cdots I$, so the gradient cannot be wiped out
  by small $\partial f_\ell/\partial\mathbf x_\ell$ alone. Without the $I$, the
  product $\prod_\ell\partial f_\ell/\partial\mathbf x_\ell$ can decay (or blow up)
  geometrically with depth. (Standard argument; the book states only that
  shortcuts keep "the gradient flow consistent," p. 113.)
- **Don't confuse with the chain rule of *probability*.** Chapter 4's
  generation loop relies on a different "chain rule":
  $P(x_1,\dots,x_T)=\prod_{t}P(x_t\mid x_{<t})$ — the autoregressive
  factorization behind [[next-token-prediction]] and
  [[autoregressive-text-generation]]. Same name, unrelated idea: one
  multiplies derivatives, the other multiplies conditional probabilities.
  (Our framing: the book generates one token at a time in §4.7 but never
  writes this factorization formula.)

## Connections
- Builds on: [[derivative]], [[partial-derivative]], [[jacobian]]
- Leads to: [[backpropagation]], [[automatic-differentiation]]
- Used by: deep networks, [[gradient]] of composite losses (Example 5.11), [[computational-graph]], [[training-loop]]
- Related (LLM book): [[logistic-regression]], [[sigmoid-function]], [[binary-cross-entropy]], [[vanishing-and-exploding-gradients]], [[shortcut-connections]], [[autoregressive-text-generation]]
- People: [[sebastian-raschka]]

## Sources
- `raw/mml-book.pdf` §5.1.2 (5.32), §5.2.1–5.2.2 (5.46–5.53), printed pp. 145–148 (PDF pp. 151–154).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.3 (Fig. A.7), book pp. 261–262; §A.4 (Fig. A.8, sidebar "Partial derivatives and gradients", Listing A.3), book pp. 263–264 (PDF pp. 285–286).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4 (vanishing gradients, shortcut connections, Fig. 4.12), book pp. 109–113; §4.7 (summary; autoregressive generation), book pp. 122–127.
