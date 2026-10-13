---
title: Activation Function
type: concept
tags: [deep-learning, neural-networks, nonlinearity, llm]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Activation Function

> **One-liner:** An activation function is a small "bend" applied to every number coming out of a layer, so a stack of layers can learn curvy patterns instead of only straight lines.

## 🟢 Intuition (Dummy)
Picture a neural network as a long line of workers. Each worker takes a list of numbers, mixes them
with a recipe (multiply by weights, add a bias), and passes the result on. The catch: if every worker
only *mixes linearly*, then the whole line of workers is no smarter than one single worker — mixing a
mix of a mix is still just one mix.

An **activation function** is a little rule each worker applies to their output before passing it on —
something like "if the number is negative, turn it into zero." That tiny bend is what breaks the
"it's all just one mix" problem. With bends in between, stacking many layers lets the network trace out
curves, corners, and really complicated shapes. No bends, no deep learning.

Different bends have different personalities:
- **ReLU** is a hard switch: negative → 0, positive → unchanged.
- **Sigmoid** squashes anything into a number between 0 and 1 (handy for "how likely?").
- **GELU** is a soft, smooth version of ReLU — the one GPT-2 uses.

## 🟡 Mechanics (Practitioner)
A typical layer computes
$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$
- $\mathbf{x}$ = the input vector coming into the layer,
- $W$ = the weight [[matrix]] (learned),
- $\mathbf{b}$ = the [[bias-vector]] (learned),
- $W\mathbf{x}+\mathbf{b}$ = the "net input" — a plain [[linear-mapping|affine map]] (this is what `nn.Linear` computes),
- $\sigma(\cdot)$ = the activation function, applied **elementwise** (to each entry separately),
- $\mathbf{h}$ = the layer's output, often just called its "activations."

**Why it must be nonlinear.** Without $\sigma$, two layers collapse into one:
$$W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2 = \underbrace{(W_2W_1)}_{W'}\mathbf{x} + \underbrace{(W_2\mathbf{b}_1+\mathbf{b}_2)}_{\mathbf{b}'}$$
Read it as: "a linear layer followed by a linear layer equals *one* linear layer with weight $W'$ and
bias $\mathbf b'$." Depth buys nothing unless something nonlinear sits in between.

**The activation functions you meet in the book:**

| Name | Formula | Read it as | Where in the book |
|---|---|---|---|
| ReLU | $\mathrm{ReLU}(z)=\max(0,z)$ | "keep $z$ if positive, else 0" | §4.2 p. 100; A.5 MLP |
| Sigmoid | $\sigma(z)=\dfrac{1}{1+e^{-z}}$ | "squash $z$ into $(0,1)$" | A.3 p. 262 (logistic regression) |
| GELU | $\mathrm{GELU}(x)=x\cdot\Phi(x)$ | "scale $x$ by the chance a standard normal is below $x$" | §4.3 p. 105 (GPT-2) |
| SwiGLU | gated, sigmoid-based | "let one branch gate another" | named on §4.3 p. 105 |

(The ReLU and sigmoid formulas are standard definitions; the book describes ReLU in words and names
$\sigma$ in Fig. A.7 without writing it out. See [[relu]], [[sigmoid-function]], [[gelu]], [[swiglu]].)

**Worked example — ReLU in action (book §4.2, p. 100).** The book builds a 5-in, 6-out layer:
```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)          # 2 examples, 5 features each
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
# tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
#        grad_fn=<ReluBackward0>)
```
Every zero you see is a place where the linear part produced a negative number and ReLU clipped it.
The `grad_fn=<ReluBackward0>` tag means PyTorch recorded the ReLU so [[backpropagation]] can later
push gradients back through it.

**Worked example — sigmoid as the "activation and output" (book A.3, p. 262).** Logistic regression as a
tiny computation graph:
$$u = w_1 x_1,\qquad z = u + b,\qquad a = \sigma(z),\qquad \text{loss}=L(a,y)$$
- $x_1$ = input feature, $w_1$ = trainable weight, $b$ = trainable bias,
- $z$ = net input, $\sigma$ = sigmoid activation, $a$ = output score in $(0,1)$,
- $y$ = true label (0 or 1), $L$ = [[binary-cross-entropy]] loss.

The book's Listing A.2 uses $y=1.0$, $x_1=1.1$, $w_1=2.2$, $b=0.0$, so $z=2.42$ and
$a=\sigma(2.42)\approx0.918$ (value computed here, not printed in the book).
```python
z = x1 * w1 + b
a = torch.sigmoid(z)              # activation and output
loss = F.binary_cross_entropy(a, y)
```

**In LLMs.** Inside every GPT [[transformer-block]] the [[feed-forward-network]] is
`Linear → GELU → Linear`. The activation is the *only* nonlinear piece in that sub-network. The book's
GPT uses [[gelu]] instead of ReLU (§4.3), and modern LLMs (e.g. Llama-style models) often use [[swiglu]].
Activation functions have **no learnable parameters** in these cases — `GELU()` and `nn.ReLU()` add zero
to the parameter count.

**Why smoothness matters (book §4.3, p. 106).** ReLU has a sharp corner at 0 and outputs exactly zero
(with zero gradient) for every negative input. GELU is smooth and gives a small non-zero output for
negative inputs, so those neurons "can still contribute to learning." The book says smoothness "can lead
to better optimization properties during training."

## 🔴 Theory (Expert)
- **Expressivity.** A composition of affine maps is affine, so a network without nonlinearities has the
  expressive power of a single [[linear-layer]]. The **universal approximation theorem** (Cybenko 1989;
  Hornik 1991; Leshno et al. 1993 — outside the book) states that a one-hidden-layer network with any
  non-polynomial continuous activation can approximate any continuous function on a compact set
  arbitrarily well, given enough hidden units. Depth then buys *efficiency* (far fewer units for the same
  function class).
- **Gradient flow.** In backprop, each layer contributes a factor $\operatorname{diag}(\sigma'(\mathbf z))\,W$
  to the Jacobian chain ([[chain-rule]]). Activations whose derivative is often tiny (sigmoid saturates:
  $\sigma'(z)=\sigma(z)(1-\sigma(z))\le \tfrac14$) feed [[vanishing-and-exploding-gradients]]. ReLU's
  derivative is exactly 1 for $z>0$ (good flow) but exactly 0 for $z<0$ ("dead" units). GELU's derivative
  $\Phi(x)+x\varphi(x)$ is non-zero almost everywhere (zero only at $x\approx-0.752$).
- **Differentiability.** ReLU is not differentiable at 0; frameworks use a subgradient (0 or 1). GELU,
  sigmoid, $\tanh$, and Swish are $C^\infty$.
- **Shortcut connections interact with activations.** The book's §4.4 experiment (book pp. 109–113) stacks
  five `Linear + GELU` layers on the input `[1.0, 0.0, -1.0]`: without shortcuts the first-layer mean
  absolute gradient is ≈ 0.0002; with [[shortcut-connections]] it is ≈ 0.22. The activation is the same —
  the residual path is what keeps gradients alive.
- **Gating view.** GELU ($x\cdot\Phi(x)$), Swish ($x\cdot\sigma(x)$), and GLU variants all have the form
  "input × a soft gate in $[0,1]$." This unifies ReLU (gate $=\mathbb 1[x>0]$) with its smooth successors.

## Connections
- Builds on: [[linear-mapping]], [[matrix]], [[bias-vector]], [[linear-layer]], [[neural-network]]
- Leads to: [[relu]], [[sigmoid-function]], [[gelu]], [[swiglu]], [[feed-forward-network]], [[multilayer-perceptron]]
- Used by: [[transformer-block]], [[logistic-regression]], [[backpropagation]], [[computational-graph]]
- Related: [[vanishing-and-exploding-gradients]], [[shortcut-connections]], [[layer-normalization]], [[derivative]]
- People: [[dan-hendrycks]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book p. 100 (Linear + ReLU layer, printed output).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book pp. 105–108 (ReLU vs. GELU vs. SwiGLU; Fig. 4.8).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4, book pp. 109–113 (Linear + GELU deep net, gradients).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.3, book p. 262 (sigmoid in the logistic-regression computation graph, Fig. A.7).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book pp. 265–269 (ReLU between hidden layers of an MLP).
