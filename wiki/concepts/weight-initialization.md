---
title: Weight Initialization
type: concept
tags: [deep-learning, optimization, probability, pytorch, llm-training]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Weight Initialization

> **One-liner:** Before training starts, every weight needs a first value — and choosing small *random* values (of the right size) is what lets the network learn at all.

## 🟢 Intuition (Dummy)
Imagine a team of 30 people who all start with the exact same opinion and hear the exact
same feedback. They'll keep agreeing forever — you've hired 30 people but you only have
one brain. A neural network has the same problem: if all the weights in a layer start
equal, every unit computes the same thing and gets the same update, so they never become
different. Starting each weight at a **small random number** gives every unit its own
"personality", so they can specialise.

*Build a Large Language Model (From Scratch)* says: "initializing model weights with
small random numbers is desired to break symmetry during training. Otherwise, the nodes
would be performing the same operations and updates during backpropagation, which would
not allow the network to learn complex mappings from inputs to outputs" (§A.5, book p. 268).

Two more practical points:
- **Size matters.** Too big, and signals (and gradients) blow up as they pass through
  many layers; too small, and they fade away. Good schemes scale the random numbers to the
  layer's size.
- **Random but repeatable.** `torch.manual_seed(123)` makes the "random" start identical
  every run, so you can reproduce the book's numbers (book p. 268).

## 🟡 Mechanics (Practitioner)
**1. Symmetry breaking with small random numbers** (§A.5, book p. 268):
```python
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)
# tensor([[-0.0577,  0.0047, -0.0702,  ...,  0.0222,  0.1260,  0.0865],
#         [ 0.0502,  0.0307,  0.0333,  ...,  0.0951,  0.1134, -0.0297],
#         ...
#         [ 0.0790,  0.1343, -0.0293,  ...,  0.0344, -0.0971, -0.0509]],
#        requires_grad=True)
```
All printed values lie between about −0.14 and +0.14 — "small" indeed. Without the seed,
your numbers differ each time you build the network.

**2. PyTorch's default `nn.Linear` init vs. naive `torch.rand`** (§3.4.2, book pp. 71–73).
`SelfAttention_v1` uses `nn.Parameter(torch.rand(d_in, d_out))`; `SelfAttention_v2` uses
`nn.Linear`, which "has an optimized weight initialization scheme, contributing to more
stable and effective model training". The two give different outputs only because of
their different starting weights:
| Version | Init | Output row 1 (book) |
|---|---|---|
| v1, seed 123 | `torch.rand` = uniform on $[0,1)$ | `[0.2996, 0.8053]` |
| v2, seed 789 | `nn.Linear` default | `[-0.0739, 0.0713]` |

**What the default actually is** (outside knowledge: the book names it only via
`kaiming_uniform_(…, a=math.sqrt(5))`, "the same initialization used for Linear layers in
PyTorch", §E.4, book p. 329): each weight is drawn uniformly from
$$W_{ij}\sim\mathcal U\!\left(-\frac{1}{\sqrt{d_{in}}},\ \frac{1}{\sqrt{d_{in}}}\right)$$
Reading it:
- $W_{ij}$ = one entry of the weight [[matrix]].
- $\sim$ = "is drawn at random from".
- $\mathcal U(-c, c)$ = the uniform distribution: every value between $-c$ and $c$ is equally likely.
- $d_{in}$ = "fan-in", the number of inputs feeding each output.

Check against the book: for `Linear(50, 30)`, $1/\sqrt{50}\approx 0.1414$, and every
printed weight above is inside $\pm0.1414$ (the largest shown is 0.1359). For the
3→2 attention projections, $1/\sqrt3\approx0.577$.

**3. Embedding layers start random too** (§2.7, book p. 42):
```python
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(6, 3)
# first rows: [ 0.3374, -0.1778, -0.1690], [ 0.9178, 1.5810, 1.3010], ...
```
These "small, random values" are the starting point that pretraining optimizes. (PyTorch
draws embeddings from a standard normal $\mathcal N(0,1)$ — an outside fact, which
explains a value like −2.84 in the last row.)

**4. Zero init on purpose: LoRA** (§E.4, book pp. 329–330, 334):
```python
self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # same as nn.Linear
self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
```
(from Listing E.5, `LoRALayer.__init__`). Since $B = 0$, the LoRA update $\tfrac{\alpha}{r}AB$ is the
zero matrix at the start, so the model behaves **exactly** like the pretrained one — the
book checks that accuracy is unchanged right after inserting LoRA (book p. 334).

**5. The best init of all: pretrained weights.** [[fine-tuning]] doesn't start from
random numbers; it starts from OpenAI's GPT-2 weights (§7.5, book p. 226). Random init is
for [[pretraining]] from scratch.

**What a random start looks like in numbers.** The untrained GPT-2 assigns roughly
uniform probabilities: about $1/50{,}257\approx0.00002$ per token (§5.1, book p. 135),
and its initial training loss is 10.988 (book p. 145) — slightly *above*
$\ln 50{,}257\approx10.825$, the loss of a perfectly uniform guess, so the random start is
a little worse than uniform (that comparison is a wiki calculation).

## 🔴 Theory (Expert)
- **Symmetry argument (formal).** Suppose two hidden units $j,k$ in a layer have
  identical incoming weights and identical outgoing weights. Then for every input their
  activations are equal, and by the chain rule their gradients are equal, so after any
  gradient step they are still identical. By induction they stay identical forever — the
  layer's effective width is 1. Any continuous random init breaks this with probability 1.
  (A single layer can be zero-initialized if *other* parameters are random: in LoRA,
  $\partial L/\partial B\neq0$ on the first step because $A$ is random, while
  $\partial L/\partial A = 0$ until $B$ moves.)
- **Variance propagation.** For $y_i=\sum_{j=1}^{d_{in}}W_{ij}x_j$ with independent,
  zero-mean weights and inputs,
  $$\mathrm{Var}(y_i) = d_{in}\,\mathrm{Var}(W_{ij})\,\mathbb E[x_j^2]$$
  so keeping signals at a stable scale across depth requires
  $\mathrm{Var}(W_{ij})\propto 1/d_{in}$ (see [[covariance-and-variance]]). Choices:
  LeCun $1/d_{in}$; Xavier/Glorot $2/(d_{in}+d_{out})$; He/Kaiming $2/d_{in}$ for ReLU
  (He et al., 2015). A uniform $\mathcal U(-c,c)$ has variance $c^2/3$, so PyTorch's
  default ($c=1/\sqrt{d_{in}}$) gives $\mathrm{Var}=1/(3d_{in})$ — a deliberately
  conservative LeCun-like scale. (All outside the book.)
- **Why `torch.rand` is poor.** $\mathcal U[0,1)$ has mean $\tfrac12$, not 0, and does not
  shrink with $d_{in}$. Pre-activations then grow like $d_{in}/2$; in attention this pushes
  the [[softmax]] toward saturation, where gradients vanish (the problem the
  $\sqrt{d_k}$ scaling addresses, §3.4, book p. 69). See
  [[vanishing-and-exploding-gradients]].
- **Depth.** Even with good per-layer variance, very deep stacks can still lose gradient;
  the book's fix is architectural — [[shortcut-connections]] and [[layer-normalization]]
  (§4.2, §4.4) — rather than a fancier init.
- **Reproducibility.** `torch.manual_seed` fixes the pseudo-random generator state, so the
  init is a deterministic function of the seed; §4.4 reseeds with 123 before building the
  shortcut and no-shortcut networks so both comparisons start from the same draw
  sequence (book pp. 111–112). See [[random-seed-reproducibility]].

## Connections
- Builds on: [[probability-distribution]], [[gaussian-distribution]], [[covariance-and-variance]], [[matrix]], [[random-variable]]
- Leads to: [[backpropagation]], [[stochastic-gradient-descent]], [[pretraining]]
- Used by: [[linear-layer]], [[embedding-layer]], [[self-attention]], [[low-rank-adaptation]], [[pytorch-nn-module]]
- Related: [[vanishing-and-exploding-gradients]], [[shortcut-connections]], [[layer-normalization]], [[random-seed-reproducibility]], [[pretrained-weights]], [[fine-tuning]], [[model-parameters]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7 (random embedding init, seed 123), book pp. 41–44.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4 (`torch.rand` vs `nn.Linear` "optimized weight initialization scheme"; v1/v2 outputs), book pp. 64–74 (quotes pp. 71–73).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4 (seeded init for the shortcut experiment), book pp. 109–113.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1 (near-uniform untrained probabilities, initial loss), book pp. 128–146.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5 (start fine-tuning from pretrained weights), book pp. 226–229.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (small random numbers, symmetry breaking, `manual_seed`), book pp. 265–269 (quote p. 268).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4 (`kaiming_uniform_(A, a=sqrt(5))`, $B$ = zeros, no-op at init), book pp. 328–336.
