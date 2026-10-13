---
title: Computational Graph
type: concept
tags: [deep-learning, calculus, automatic-differentiation, pytorch]
difficulty: intermediate
sources: [llm-appA-introduction-to-pytorch, mml-ch05-vector-calculus]
created: 2026-10-13
updated: 2026-10-13
---

# Computational Graph

> **One-liner:** A computational graph is a flowchart of every little calculation a model does, drawn as boxes and arrows, so the computer can later walk it backward to find out how to fix each weight.

(Also called a **computation graph** — the book uses both names, book pp. 261–262.)

## 🟢 Intuition (Dummy)
Think of a recipe drawn as a diagram: "take flour and eggs → mix → add sugar → bake →
taste". Each arrow says "this result feeds into that step". If the cake tastes bad, you
can walk the diagram **backward** to ask: how much was the baking to blame? The sugar?
The flour?

A neural network is the same kind of recipe, just with numbers. Inputs and weights get
multiplied, added, squashed, and finally compared with the right answer to get a
**loss** (an error score). PyTorch quietly draws this diagram while your code runs. Then,
when you call `loss.backward()`, it walks the diagram from the loss back to every weight
and works out "if I nudge this weight, how much does the error change?" That's the
[[gradient]], and it's all you need to train the model.

## 🟡 Mechanics (Practitioner)

### The book's graph: a logistic-regression forward pass (Figure A.7, book p. 262)
A [[logistic-regression]] classifier — "a single-layer neural network" — written as a
chain of tiny steps:

$$u = w_1 \times x_1,\qquad z = u + b,\qquad a = \sigma(z),\qquad \text{loss} = L(a, y)$$

Symbol by symbol:
- $x_1$ — the input feature (the data) = **1.1**
- $w_1$ — a **trainable** weight = **2.2**
- $b$ — a **trainable** [[bias-vector|bias]] = **0.0**
- $u$ — an **intermediate result** (a node inside the graph) $= 2.42$
- $z$ — the net input $= 2.42$
- $\sigma$ — the [[sigmoid-function]], which squashes $z$ into $(0,1)$; $a = \sigma(2.42) \approx 0.9183$
- $y$ — the true (target) label = **1.0**
- $L$ — the loss function ([[binary-cross-entropy]]); $L \approx 0.0852$

(The intermediate values are computed here; the book prints only the inputs and the final gradients.)

```python
import torch.nn.functional as F
y  = torch.tensor([1.0])    # true label
x1 = torch.tensor([1.1])    # input feature
w1 = torch.tensor([2.2])    # weight parameter
b  = torch.tensor([0.0])    # bias unit
z = x1 * w1 + b             # net input
a = torch.sigmoid(z)        # activation and output
loss = F.binary_cross_entropy(a, y)
```

Each line of code becomes a **node**; each "this value is used by that operation" becomes
a directed **edge**. Leaves are the values you created yourself ($x_1, w_1, b, y$); the
root is the loss.

### Walking the graph backward: the chain rule (Figure A.8, book p. 263)
To train, we need $\partial L/\partial w_1$ and $\partial L/\partial b$. Each edge carries a
**local** derivative; the [[chain-rule]] says: multiply them along the path from the loss
back to the parameter.

$$\frac{\partial L}{\partial w_1} = \frac{\partial u}{\partial w_1} \times \frac{\partial z}{\partial u} \times \frac{da}{dz} \times \frac{\partial L}{\partial a}$$

$$\frac{\partial L}{\partial b} = \frac{\partial z}{\partial b} \times \frac{da}{dz} \times \frac{\partial L}{\partial a}$$

Reading the notation: $\frac{\partial L}{\partial w_1}$ = "how much $L$ changes per tiny change
in $w_1$, holding everything else fixed" (a [[partial-derivative]]); $\frac{da}{dz}$ = the
ordinary [[derivative]] of the sigmoid; $\times$ = plain multiplication of the local slopes.

Plugging in numbers (worked here, matches the book's printout):

| Edge | Local derivative | Value |
|---|---|---|
| $\partial L/\partial a$ | $-y/a + (1-y)/(1-a)$ | $-1/0.9183 \approx -1.089$ |
| $da/dz$ | $\sigma(z)(1-\sigma(z))$ | $0.9183\times0.0817\approx 0.0750$ |
| $\partial z/\partial u$, $\partial z/\partial b$ | $1$ | $1$ |
| $\partial u/\partial w_1$ | $x_1$ | $1.1$ |

So $\partial L/\partial b \approx -1.089 \times 0.0750 \approx -0.0817$ and
$\partial L/\partial w_1 \approx 1.1 \times (-0.0817) \approx -0.0898$.

The book's Figure A.8 caption: applying the chain rule **right to left** — from the loss back toward
the inputs — is called **reverse-mode automatic differentiation** or **[[backpropagation]]**
(book p. 263).

### Letting PyTorch build and use the graph (Listing A.3, book p. 264)
PyTorch builds the graph internally, by default, **if one of its terminal (leaf) nodes has
`requires_grad=True`** (book p. 263). Its autograd engine does this "by tracking every
operation performed on tensors" (book p. 264):

```python
from torch.autograd import grad
w1 = torch.tensor([2.2], requires_grad=True)
b  = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)   # (tensor([-0.0898]),)
grad_L_b  = grad(loss, b,  retain_graph=True)   # (tensor([-0.0817]),)

loss.backward()          # the usual way: fills .grad on every leaf
print(w1.grad, b.grad)   # -0.0898, -0.0817
```

- `retain_graph=True`: by default PyTorch **destroys the graph after computing gradients**
  to free memory; this keeps it around because we use it again (book p. 264).
- `.backward()` computes gradients for **all leaf nodes** and stores them in `.grad` (book p. 264).

### Graphs in a real network (§A.5, §A.7)
- In a `torch.nn.Module`, the `forward` method describes how data passes through the layers
  and "comes together as a computation graph"; you never write `backward` yourself
  (book p. 265).
- The output tensor remembers its last operation: `grad_fn=<AddmmBackward0>` — "Addmm" is a
  [[matrix-multiplication]] followed by an addition, the final Linear layer (book p. 269).
- **Inference without a graph:** wrap prediction in `with torch.no_grad():`. Building the
  graph is wasted memory and compute when you won't backpropagate (book p. 269); the output
  then has no `grad_fn` — the book prints `tensor([[-0.1262, 0.1080, -0.1792]])`, the same
  numbers without `grad_fn=<AddmmBackward0>`.
- **Why models return logits:** PyTorch's common loss functions combine the softmax (or the
  **sigmoid** for binary classification) with the negative log-likelihood loss in one class,
  "for numerical efficiency and stability" (book p. 269).
- **In the training loop** (book pp. 275–276): `loss.backward()` computes the gradients "in
  the computation graph PyTorch constructed in the background"; `optimizer.zero_grad()` is
  needed because PyTorch **adds** new gradients to existing `.grad` values instead of
  replacing them (see [[gradient-accumulation]]).

## 🔴 Theory (Expert)
- **Formal object.** A computational graph is a **directed acyclic graph** (DAG)
  $G=(V,E)$ whose nodes $v_i$ hold values $v_i = f_i(v_{\text{pa}(i)})$, where $\text{pa}(i)$
  are the parents of $i$ and $f_i$ is an elementary differentiable operation ($\times$, $+$,
  $\sigma$, matmul, …). Any topological ordering of $V$ is a valid evaluation order for the
  forward pass.
- **Reverse-mode accumulation.** Define the *adjoint* $\bar v_i = \partial L/\partial v_i$.
  Starting from $\bar L = 1$ and visiting nodes in reverse topological order,
  $$\bar v_j = \sum_{i:\, j\in\text{pa}(i)} \bar v_i\,\frac{\partial f_i}{\partial v_j}.$$
  The sum over children is the multivariate chain rule: if a value is used in several places
  (a "fan-out" node, such as a weight shared across time steps or [[weight-tying|tied
  embeddings]]), its gradient is the **sum** of the contributions along each path. PyTorch's
  `.grad` accumulation is literally this sum, which is why `zero_grad()` must reset it between
  optimizer steps.
- **Cost.** One backward pass costs a small constant multiple of the forward pass and gives
  the gradient with respect to *all* parameters at once — the cheap-gradient principle that
  makes training 124M-parameter GPT-2 feasible. The price is memory: every intermediate
  activation needed by a local derivative (e.g. $a$ for $\sigma'(z)=a(1-a)$) must be cached
  until the backward pass. That's why `torch.no_grad()` saves memory at inference.
- **Dynamic vs static graphs.** PyTorch's autograd builds a **dynamic** graph — "on the fly"
  as Python runs (book pp. 261–262) — so ordinary loops and `if`s work, and the graph can
  differ per batch (e.g. variable sequence lengths from [[dynamic-padding]]). Static-graph
  frameworks compile the graph once and then run it.
- **Simplifications the graph reveals.** For sigmoid + BCE, the product of the two
  right-most edges collapses: $\frac{\partial L}{\partial a}\frac{da}{dz} = a - y$. Likewise
  softmax + cross-entropy gives $\mathbf{p}-\mathbf{y}$. Fusing such pairs into one node is more
  stable numerically, which is why PyTorch's `F.cross_entropy` takes [[logits]] directly
  (book pp. 269, 276). See [[numerical-stability]].
- **Relation to the MML view.** The [[backpropagation]] page derives the same procedure as a
  product of layer Jacobians $\partial\mathbf f_{i+1}/\partial\mathbf f_i$; the computational graph
  is the finer-grained picture in which each elementary operation, not each layer, is a node.
  See [[automatic-differentiation]] and [[jacobian]].

## Connections
- Builds on: [[derivative]], [[partial-derivative]], [[chain-rule]], [[gradient]]
- Leads to: [[automatic-differentiation]], [[backpropagation]], [[training-loop]]
- Used by: [[pytorch]], [[pytorch-nn-module]], [[forward-pass]], [[logistic-regression]], [[multilayer-perceptron]], [[gradient-accumulation]], [[llm-training-loop]]
- Related: [[sigmoid-function]], [[binary-cross-entropy]], [[jacobian]], [[train-vs-eval-mode]], [[numerical-stability]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.3 (computation graphs, Listing A.2, Figure A.7), book pp. 261–262
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.4 (autograd, chain rule on the graph, Figure A.8, Listing A.3, `retain_graph`, `.backward()`), book pp. 263–264
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (`forward` builds the graph, `grad_fn=<AddmmBackward0>`, `torch.no_grad()`), book pp. 265–269
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7 (`loss.backward()` in the training loop, gradient accumulation note), book pp. 275–276
- `raw/mml-book.pdf` §5.6 (backpropagation and automatic differentiation) — background for the 🔴 tier
