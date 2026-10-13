---
title: ReLU (Rectified Linear Unit)
type: concept
tags: [deep-learning, activation-function, neural-network, pytorch]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# ReLU (Rectified Linear Unit)

> **One-liner:** ReLU keeps a number if it's positive and replaces it with zero if it's negative.

## 🟢 Intuition (Dummy)
ReLU is a one-way valve. Positive values flow through untouched. Negative values get blocked and come out as 0.

Why would a neural network want that? Stacking plain linear layers is like stacking magnifying glasses: no matter how many you stack, you only get one bigger magnifying glass. You never get anything *new*. Putting a valve between the layers adds a "bend", and with enough bends the network can draw any shape it needs. ReLU is the simplest possible bend, and that's why it became the default choice in deep learning.

LLMs like GPT-2 swap it for a smoother cousin, [[gelu|GELU]]. GELU looks almost the same but has no sharp corner and lets a trickle of negative values through.

## 🟡 Mechanics (Practitioner)
### The formula
$$\mathrm{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$

Symbol by symbol:
- $x$ = one input number (ReLU is applied to every entry of a tensor separately, "elementwise"),
- $\max(0, x)$ = "the larger of 0 and $x$",
- the big brace = "pick the line whose condition is true".

Its slope (derivative), which [[backpropagation]] uses:
$$\frac{d}{dx}\mathrm{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}\qquad(\text{undefined at } x=0;\ \text{PyTorch uses } 0)$$
- $\frac{d}{dx}$ = "how fast the output changes when $x$ changes a tiny bit".

The book describes it in words: ReLU is **piecewise linear**, it "outputs the input directly if it is positive; otherwise, it outputs zero" (book p. 106), and it "thresholds negative inputs to 0" (book p. 100).

### Example 1: Linear + ReLU (§4.2, book p. 100)
```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
```
Output:
```
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
       grad_fn=<ReluBackward0>)
```
Every negative pre-activation became exactly 0. That's 3 zeros in row 1 and 2 in row 2. The book says the output has "only positive values"; strictly, it has only *non-negative* values. `grad_fn=<ReluBackward0>` shows that PyTorch recorded the operation for [[automatic-differentiation|autograd]].

### Example 2: an MLP with two hidden layers (§A.5, book pp. 266–267)
```python
self.layers = torch.nn.Sequential(
    torch.nn.Linear(num_inputs, 30), torch.nn.ReLU(),
    torch.nn.Linear(30, 20),         torch.nn.ReLU(),
    torch.nn.Linear(20, num_outputs),
)
```
As math:
$$\mathbf{h}_1 = \mathrm{ReLU}(W_1\mathbf{x} + \mathbf{b}_1),\quad \mathbf{h}_2 = \mathrm{ReLU}(W_2\mathbf{h}_1 + \mathbf{b}_2),\quad \mathbf{z} = W_3\mathbf{h}_2 + \mathbf{b}_3$$
- $\mathbf{x}$ = input features (50 of them), $W_k$ = weight matrices, $\mathbf{b}_k$ = [[bias-vector|bias vectors]],
- $\mathbf{h}_1, \mathbf{h}_2$ = hidden activations (30 and 20 numbers),
- $\mathbf{z}$ = the [[logits]] (3 numbers). Note there is **no** ReLU at the end, because logits must be able to go negative.

With `NeuralNetwork(50, 3)` the model has **2,213** trainable parameters, and the ReLU layers contribute **0** of them (book p. 267).

### Example 3: ReLU vs GELU (§4.3, book pp. 105–106)
```python
gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
```
Figure 4.8 plots both over $[-3, 3]$. ReLU is flat at 0 for $x<0$ with a sharp corner at 0. GELU is smooth and dips slightly below 0 (minimum about −0.17 near $x\approx-0.75$). The book says GELU has "a non-zero gradient for almost all negative values (except at approximately x = –0.75)", i.e. exactly at that minimum (book p. 106). The book's points:
- ReLU was popular "due to its simplicity and effectiveness across various neural network architectures" (book p. 105). In LLMs, other activations replace it; the book names GELU and SwiGLU, "more complex and smooth activation functions" that "offer improved performance" (book p. 105). Section 4.2 ends by calling ReLU "the traditional ReLU function we used previously" (book p. 104).
- Its sharp corner at zero "can sometimes make optimization harder, especially in networks that are very deep or have complex architectures" (book p. 106).
- ReLU outputs zero for any negative input, while GELU gives a small non-zero output, so neurons with negative input "can still contribute to the learning process" (book p. 106).

## 🔴 Theory (Expert)
- **Why a nonlinearity is needed.** A composition of affine maps is affine: $W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2 = (W_2W_1)\mathbf{x}+(W_2\mathbf{b}_1+\mathbf{b}_2)$. Without an [[activation-function]], depth adds nothing.
- **Universal approximation.** One hidden layer of ReLUs (wide enough) can approximate any continuous function on a compact set as closely as you like (a standard result). A ReLU network computes a **continuous piecewise-linear** function. The number of linear pieces can grow exponentially with depth, which is one formal reason depth helps.
- **Positive homogeneity.** $\mathrm{ReLU}(cx)=c\,\mathrm{ReLU}(x)$ for $c\ge0$. So you can rescale one layer by $c$ and the next by $1/c$ without changing the function. That's a symmetry in parameter space and one reason [[weight-decay]] and normalization interact with ReLU nets.
- **Gradients.** For active units the slope is exactly 1, so gradients pass through without shrinking. That's the reason ReLU replaced sigmoid/tanh, whose slopes are $\le 1/4$ and $\le 1$ and shrink toward 0 in the tails ([[vanishing-and-exploding-gradients]]). The flip side is **dead neurons**: a unit whose pre-activation is negative for every input gets zero gradient forever and stops learning. That's the problem the book alludes to when it praises GELU's non-zero gradient for negative inputs.
- **Non-differentiability at 0.** ReLU is not differentiable at 0 but it is convex, so any value in $[0,1]$ is a valid *subgradient* there. Frameworks pick 0. In practice inputs are almost never exactly 0, so it rarely matters. Even so, the kink makes the loss surface non-smooth, which is the book's argument for smooth alternatives.
- **Successors.** Leaky ReLU ($\max(\alpha x, x)$), [[gelu|GELU]] ($x\,\Phi(x)$, used by GPT-2), and [[swiglu|SwiGLU]] (gated, used by Llama-style models). GELU can be read as a "soft ReLU": ReLU multiplies $x$ by the hard gate $\mathbb{1}[x>0]$, while GELU multiplies by $\Phi(x)=P(Z\le x)$ (the [[cumulative-distribution-function]] of a standard [[gaussian-distribution]]).

## Connections
- Builds on: [[linear-mapping]], [[linear-layer]], [[derivative]]
- Leads to: [[activation-function]], [[gelu]], [[swiglu]], [[feed-forward-network]]
- Used by: [[multilayer-perceptron]], [[neural-network]], [[deep-learning]], the §4.2 [[layer-normalization]] demo
- Related: [[backpropagation]], [[vanishing-and-exploding-gradients]], [[sigmoid-function]], [[pytorch-nn-module]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 100, 104 (`nn.Sequential(nn.Linear(5, 6), nn.ReLU())` and its output p. 100; "traditional ReLU" p. 104).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book pp. 105–106 (ReLU vs GELU/SwiGLU, Fig. 4.8, why smoothness helps).
  > ⚠️ Book typo: the text on p. 106 says "figure 4.18, right". It means figure 4.8.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book pp. 266–267 (Listing A.4 MLP with ReLU, 2,213 parameters).
