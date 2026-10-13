---
title: Forward Pass
type: concept
tags: [deep-learning, neural-networks, pytorch, computation-graph]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Forward Pass

> **One-liner:** The forward pass is the "make a prediction" step: you push the input through the network, layer by layer, until an answer (and a loss) comes out the other end.

## 🟢 Intuition (Dummy)
Think of a car factory line. Raw metal goes in at one end, each station does one job (cut,
weld, paint), and a finished car rolls out at the other end. A neural network is a line of
math stations. The **forward pass** is one trip down the line: the input goes in, each layer
transforms it, and a prediction comes out.

Training has two halves:
1. **Forward pass**: "What do you predict, and how wrong is it?" (compute the output and the [[loss-function|loss]]).
2. **Backward pass**: "Who is to blame, and by how much?" ([[backpropagation]] computes the gradients).

While the forward pass runs, PyTorch quietly writes down every operation it did, like a
receipt. That receipt is the [[computational-graph]], and the backward pass reads it in
reverse. When you only want predictions (no training), you can tell PyTorch to skip the receipt
and save memory.

## 🟡 Mechanics (Practitioner)

### Smallest example: logistic regression (Figure A.7, book p. 262)
The book's first forward pass is a single "neuron":
$$u = w_1 \times x_1,\qquad z = u + b,\qquad a = \sigma(z),\qquad \text{loss} = L(a, y)$$
Symbol by symbol:
- $x_1$ = the input feature (the data), here $1.1$.
- $w_1$ = a trainable weight, here $2.2$. $b$ = a trainable bias, here $0.0$.
- $u$ = an intermediate result, the product $w_1 x_1$.
- $z$ = the net input, "weighted input plus bias" $= 2.42$.
- $\sigma$ = the [[sigmoid-function]] activation; $a = \sigma(z) \approx 0.9183$ is the output, a score in $(0,1)$.
- $y$ = the true label ($1.0$); $L$ = the loss, here [[binary-cross-entropy]], $\approx 0.0852$.

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
(Listing A.2, "A logistic regression forward pass", book p. 262. $z$, $a$, loss values computed by hand.)

### A real network: the MLP of Listing A.4 (book pp. 266–268)
For the [[multilayer-perceptron]] with layers 50 → 30 → 20 → 3, the forward pass is
$$\mathbf{h}_1 = \mathrm{ReLU}(W_1\mathbf{x} + \mathbf{b}_1),\quad
\mathbf{h}_2 = \mathrm{ReLU}(W_2\mathbf{h}_1 + \mathbf{b}_2),\quad
\mathbf{z} = W_3\mathbf{h}_2 + \mathbf{b}_3$$
- $\mathbf{x} \in \mathbb{R}^{50}$ = the input vector ("$\in \mathbb{R}^{50}$" reads "is a list of 50 real numbers").
- $W_1 \in \mathbb{R}^{30\times 50}$ = weight [[matrix]] of layer 1 (30 rows, 50 columns); $\mathbf{b}_1 \in \mathbb{R}^{30}$ = its [[bias-vector]].
- $W_1\mathbf{x} + \mathbf{b}_1$ = an affine map ([[linear-mapping]] plus a shift).
- $\mathrm{ReLU}(v) = \max(0, v)$, applied to every entry — the [[relu]] [[activation-function]].
- $\mathbf{h}_1, \mathbf{h}_2$ = the hidden-layer outputs; $\mathbf{z} \in \mathbb{R}^{3}$ = the [[logits]] (raw scores, one per class).
- Reading it left to right *is* the forward pass: each line feeds the next.

In PyTorch you write the recipe once in `forward` and then simply *call* the model:
```python
torch.manual_seed(123)
X = torch.rand((1, 50))     # one toy example, 50 features
out = model(X)              # calling the model runs forward()
print(out)
# tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)
```
- Shape in: `(1, 50)`; shape out: `(1, 3)`, three scores (book p. 268).
- `grad_fn=<AddmmBackward0>` is the forward pass's "receipt": it records the last operation in the
  graph, **Addmm** = matrix multiply (`mm`) then add (`Add`), i.e. the last layer's $XW^\top + \mathbf{b}$.
  Backprop uses it later (book p. 269).
- PyTorch's batch convention puts examples in rows, so a `Linear` layer computes $Y = XW^\top + \mathbf{b}$
  with $X \in \mathbb{R}^{n\times d_{\text{in}}}$ and $W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ (that is why
  `model.layers[0].weight.shape` is `torch.Size([30, 50])`).

### Forward pass for inference only
```python
with torch.no_grad():
    out = model(X)
print(out)                  # tensor([[-0.1262,  0.1080, -0.1792]])  -- no grad_fn
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)                  # tensor([[0.3113, 0.3934, 0.2952]])
```
- `torch.no_grad()` = "don't build the graph". Same numbers, but no `grad_fn` and big memory/compute savings (book p. 269).
- [[softmax]] turns logits into class probabilities; $0.3113 + 0.3934 + 0.2952 \approx 1$. They are roughly equal because the model is untrained.

### In the training loop
In Listing A.9 (§A.7) each step is: forward (`logits = model(features)`), loss
(`F.cross_entropy`), `optimizer.zero_grad()`, backward (`loss.backward()`), update
(`optimizer.step()`). The same pattern drives GPT training later in the book (see [[llm-training-loop]]).

## 🔴 Theory (Expert)
- **Function composition.** A network is $f = f_K \circ \cdots \circ f_1$; the forward pass evaluates it
  from the inside out, and the loss adds one more function on top: $\mathcal{L}(\theta) = L(f_\theta(\mathbf{x}), y)$.
- **Forward pass = building a DAG.** In PyTorch's *dynamic* (define-by-run) [[computational-graph]], the
  graph is created on the fly as Python executes the forward code, once any leaf tensor has
  `requires_grad=True` (book pp. 261–263). Control flow (loops, `if`) can therefore differ per input.
- **Caching for the backward pass.** Reverse-mode [[automatic-differentiation]] needs the intermediate
  values (e.g. $a$ for $\sigma'(z) = a(1-a)$, or $\mathbf{h}_1$ for $\partial\mathcal{L}/\partial W_2$). The forward pass
  stores them, which is why training memory grows with depth and batch size, and why `no_grad()` saves
  so much. Activation checkpointing trades this memory for recomputing parts of the forward pass.
- **Cost.** One forward pass of a dense layer costs about $2\,n\,d_{\text{in}}\,d_{\text{out}}$ FLOPs; the backward pass
  costs roughly twice that, so one training step is about $3\times$ a forward pass — the "same order"
  result for [[backpropagation]].
- **Do not confuse with forward-mode AD.** "Forward pass" is evaluating the function; *forward-mode*
  automatic differentiation is a different way of propagating derivatives (Jacobian–vector products).
  Backprop is reverse mode and runs *after* the forward pass.
- **Mode matters.** Layers like [[dropout]] and [[batch-normalization]] behave differently in a training
  forward pass vs. an evaluation forward pass, hence `model.train()` / `model.eval()` (see [[train-vs-eval-mode]]).

## Connections
- Builds on: [[matrix-multiplication]], [[linear-mapping]], [[activation-function]], [[tensor]], [[pytorch-nn-module]]
- Leads to: [[backpropagation]], [[computational-graph]], [[automatic-differentiation]], [[training-loop]]
- Used by: [[multilayer-perceptron]], [[logistic-regression]], [[gpt-architecture]], [[autoregressive-text-generation]]
- Related: [[logits]], [[softmax]], [[binary-cross-entropy]], [[cross-entropy-loss]], [[train-vs-eval-mode]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.3, book pp. 261–262 (computation graphs; Listing A.2 "A logistic regression forward pass"; Figure A.7).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book pp. 265–269 (`forward` method; forward pass output, `grad_fn=<AddmmBackward0>`, `torch.no_grad()`, softmax).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7, book pp. 274–276 (forward pass inside the training loop, Listing A.9).
- FLOP counts, activation caching, forward-mode AD remark: standard background, not in the book.
