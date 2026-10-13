---
title: Partial Derivative
type: concept
tags: [calculus, differentiation, multivariate]
difficulty: beginner
sources: [mml-ch05-vector-calculus, llm-appA-introduction-to-pytorch]
created: 2026-06-04
updated: 2026-10-13
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

**In LLMs (and every neural network).** A model like GPT-2 has about 124 million knobs
(weights). The loss depends on all of them at once. The partial derivative with respect to one
knob answers: "if I turn *only this one* a tiny bit, does the loss go up or down, and how fast?"
The LLM book's sidebar puts it simply: a partial derivative "measures the rate at which a
function changes with respect to one of its variables" (book p. 263). PyTorch computes all of
these for you; you rarely do it by hand.

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

### In LLMs: the logistic-regression example (LLM book §A.4, pp. 263–264)
The LLM book's Appendix A shows partial derivatives on a tiny [[computational-graph]] (Figure A.8):
$$u = w_1 x_1,\qquad z = u + b,\qquad a = \sigma(z),\qquad L = L(a, y)$$
with $x_1 = 1.1$, $w_1 = 2.2$, $b = 0$, $y = 1$. The loss $L$ depends on *two* parameters, $w_1$ and
$b$, so it has two partial derivatives. Each edge of the graph carries a local one, and the
[[chain-rule]] multiplies them from right to left:
$$\frac{\partial L}{\partial w_1} = \frac{\partial u}{\partial w_1}\times\frac{\partial z}{\partial u}\times\frac{da}{dz}\times\frac{\partial L}{\partial a},
\qquad
\frac{\partial L}{\partial b} = \frac{\partial z}{\partial b}\times\frac{da}{dz}\times\frac{\partial L}{\partial a}$$
Reading the symbols:
- $\frac{\partial L}{\partial w_1}$ = "how fast the loss changes when only the weight $w_1$ moves" ($b$ frozen).
- $\frac{\partial u}{\partial w_1} = x_1 = 1.1$ — treat $x_1$ as a constant, so $u = w_1 x_1$ has slope $x_1$.
- $\frac{\partial z}{\partial u} = 1$ and $\frac{\partial z}{\partial b} = 1$ — in $z = u + b$, each input moves $z$ one-for-one.
- $\frac{da}{dz} = \sigma(z)(1-\sigma(z)) \approx 0.9183 \times 0.0817 \approx 0.0750$ — an ordinary [[derivative]] (written with $d$, since $a$ depends on $z$ alone).
- $\frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a} \approx -1.0890$ — from [[binary-cross-entropy]].

Result: $\frac{\partial L}{\partial b} \approx -0.0817$ and $\frac{\partial L}{\partial w_1} \approx 1.1 \times (-0.0817) \approx -0.0898$,
exactly what PyTorch prints (intermediate factors computed by hand):
```python
from torch.autograd import grad
w1 = torch.tensor([2.2], requires_grad=True)
b  = torch.tensor([0.0], requires_grad=True)
# ... forward pass: z = x1 * w1 + b; a = torch.sigmoid(z); loss = F.binary_cross_entropy(a, y)
grad(loss, w1, retain_graph=True)   # (tensor([-0.0898]),)
grad(loss, b,  retain_graph=True)   # (tensor([-0.0817]),)
loss.backward(); w1.grad, b.grad    # same values, stored in .grad
```
Collecting both numbers gives the [[gradient]] $\nabla L = [-0.0898,\ -0.0817]$, which
[[gradient-descent]] uses to update the parameters ("gradients provide the information needed to
update each parameter to minimize the loss function", book p. 263).

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
- **In LLMs: partials at scale.** For a network with parameter vector $\theta \in \mathbb{R}^P$
  ($P \approx 1.24\times10^8$ for GPT-2 small) and scalar loss $\mathcal L$, `loss.backward()` returns all
  $P$ partials $\partial\mathcal L/\partial\theta_j$ in one reverse sweep, at a cost of the same order as one
  [[forward-pass]]. Computing them one at a time by finite differences would need $P$ extra forward
  passes — the reason reverse-mode [[automatic-differentiation]] (backpropagation) is used. The LLM
  book frames it as "an implementation of the chain rule from calculus for neural networks" (§A.4, p. 263).
- **Notation note (not a contradiction).** The LLM book's Figure A.8 mixes $\partial$ and $d$: it writes
  $\frac{da}{dz}$ because $a=\sigma(z)$ is single-variable, and $\partial$ for edges into multi-input nodes.
  This matches the MML convention above.


## Connections
- Builds on: [[derivative]]
- Leads to: [[gradient]], [[jacobian]], [[hessian]]
- Used by: [[chain-rule]], [[backpropagation]], gradient descent (Ch. 7), [[gradient-descent]], [[automatic-differentiation]], [[computational-graph]], [[llm-training-loop]]
- Related: [[binary-cross-entropy]], [[sigmoid-function]], [[logistic-regression]], [[forward-pass]]
- People: —

## Sources
- `raw/mml-book.pdf` §5.2 (Def 5.5), §5.7, printed pp. 146–147, 164–165 (PDF pp. 152–153, 170–171).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.4, book pp. 263–264 (sidebar "Partial derivatives and gradients"; Figure A.8 chain rule on the logistic-regression graph; Listing A.3 `torch.autograd.grad`, outputs −0.0898 / −0.0817).
