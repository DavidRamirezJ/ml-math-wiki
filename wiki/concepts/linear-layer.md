---
title: Linear (Fully Connected) Layer
type: concept
tags: [deep-learning, linear-algebra, neural-network, pytorch, llm-building]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Linear (Fully Connected) Layer

> **One-liner:** A linear layer multiplies its input by a learnable weight matrix and adds a learnable bias vector — "mix the inputs with learned recipes, then shift".

## 🟢 Intuition (Dummy)
Picture a row of output "judges". Each judge looks at **every** input number, gives
each one a personal importance score (a weight), adds them all up, and then adds a
personal offset (the bias). That's one output. A linear layer is just many such judges
side by side — and because every judge is connected to every input, it's also called a
**fully connected** layer. Training is the process of finding good importance scores
for every judge.

It's the most common brick in an LLM. In GPT-2, linear layers make the queries, keys and
values in attention, do the heavy lifting in the feed-forward part of every transformer
block, and produce the final scores over all 50,257 tokens.

*Build a Large Language Model (From Scratch)* puts it in one sentence: "the Linear layer
multiplies the inputs with a weight matrix and adds a bias vector. This is sometimes
referred to as a feedforward or fully connected layer" (§A.5, book p. 267).

## 🟡 Mechanics (Practitioner)
**The formula (one example as a column vector):**
$$\mathbf y = W\mathbf x + \mathbf b$$
Reading it symbol by symbol:
- $\mathbf x\in\mathbb R^{d_{in}}$ = the input vector ($d_{in}$ numbers, e.g. one token's embedding).
- $W\in\mathbb R^{d_{out}\times d_{in}}$ = the **weight [[matrix]]**: $d_{out}$ rows (one per output "judge"), $d_{in}$ columns (one per input).
- $W\mathbf x$ = [[matrix-multiplication|matrix–vector product]]: output $i$ is the [[inner-product|dot product]] of row $i$ of $W$ with $\mathbf x$, i.e. $\sum_j W_{ij}x_j$.
- $\mathbf b\in\mathbb R^{d_{out}}$ = the [[bias-vector]], one shift per output.
- $\mathbf y\in\mathbb R^{d_{out}}$ = the output vector.

**The formula PyTorch actually runs (a batch of rows):**
$$Y = XW^\top + \mathbf b$$
- $X\in\mathbb R^{n\times d_{in}}$ = a batch of $n$ inputs, **one per row**.
- $W^\top$ = the [[matrix-transpose|transpose]] of $W$ (shape $d_{in}\times d_{out}$).
- $+\,\mathbf b$ = the bias is added to every row ([[broadcasting]]).
- $Y\in\mathbb R^{n\times d_{out}}$ = one output row per input row.

That's why PyTorch **stores the weight as $d_{out}\times d_{in}$**: for `Linear(50, 30)`,
`model.layers[0].weight.shape` is `torch.Size([30, 50])` (§A.5, book p. 267). The
`grad_fn=<AddmmBackward0>` on its output literally says "**add** + **m**atrix-**m**ultiply"
(book p. 269).

**Counting parameters:** $d_{in}\cdot d_{out}$ weights $+\ d_{out}$ biases.
- `Linear(50, 30)`: $1500+30 = 1530$. The whole 50→30→20→3 MLP has **2,213** (§A.5, book p. 267).
- GPT-2 small feed-forward module, `Linear(768, 3072)` + `Linear(3072, 768)`:
  $(768\cdot3072+3072)+(3072\cdot768+768) = 4{,}722{,}432$ parameters, about twice the
  attention module's 2,360,064 (Exercise 4.1, Appendix C, book p. 302).

**In PyTorch:**
```python
layer = torch.nn.Linear(in_features=50, out_features=30)   # bias=True by default
layer.weight   # Parameter, shape [30, 50], requires_grad=True
layer.bias     # Parameter, shape [30]
y = layer(x)   # x: (..., 50)  ->  y: (..., 30)
```
- Weights and biases are trainable by default (`requires_grad=True`) (book p. 267).
- `nn.Linear` works on the **last** dimension, so a `(batch, tokens, 768)` tensor goes
  in and a `(batch, tokens, d_out)` tensor comes out — every token gets the same layer.

**Where linear layers live in the GPT of the book:**
| Use | Layer | Bias? | Section |
|---|---|---|---|
| query / key / value projections | `nn.Linear(d_in, d_out, bias=qkv_bias)` | off by default (`qkv_bias=False`) | §3.4.2, book p. 72 |
| multi-head output projection | `out_proj = nn.Linear(d_out, d_out)` | yes | §3.6 ([[output-projection]]) |
| feed-forward expand / contract | `Linear(768, 3072)`, `Linear(3072, 768)` | yes | §4.3, book p. 107 |
| output head → vocabulary logits | `nn.Linear(emb_dim, vocab_size, bias=False)` | no | §4.6, book p. 119 |
| spam classifier head | `Linear(768, 2)` | yes | §6.5, book p. 186 ([[classification-head]]) |

**Linear layer vs. raw `nn.Parameter` (§3.4.2).** With biases disabled, `nn.Linear`
does the same matrix multiplication as `x @ W` with `W = nn.Parameter(torch.rand(d_in, d_out))`.
The book prefers `nn.Linear` because it "has an optimized weight initialization scheme,
contributing to more stable and effective model training" (book pp. 71–72); see
[[weight-initialization]]. To copy weights between the two versions you must transpose
(Exercise 3.1, book p. 73; solution Appendix C, book p. 301):
```python
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
```

**Nonlinearity is what makes stacking useful.** Between linear layers the book always
puts an [[activation-function]] — [[relu|ReLU]] in the MLP (§A.5), [[gelu|GELU]] in GPT's
feed-forward (§4.3).

## 🔴 Theory (Expert)
- **Affine, not linear.** Strictly, $\mathbf x\mapsto W\mathbf x+\mathbf b$ is an
  **affine map**: a [[linear-mapping]] followed by a translation (see [[affine-space]]).
  It becomes linear only when $\mathbf b = \mathbf 0$ (e.g. `bias=False`). The bias can be
  folded into $W$ by appending a constant 1 to $\mathbf x$ (homogeneous coordinates) —
  that's what the "bias unit" node in Figure A.9 depicts (book p. 265).
- **Stacking linear layers without activations collapses.** $W_2(W_1\mathbf x+\mathbf b_1)+\mathbf b_2
  = (W_2W_1)\mathbf x + (W_2\mathbf b_1+\mathbf b_2)$ is again a single affine map, with
  [[basis-and-rank|rank]] at most $\min(\mathrm{rank}\,W_1,\mathrm{rank}\,W_2)$. Activations
  break this collapse; that's why every MLP alternates linear and nonlinear layers.
- **Gradients of a linear layer** (for [[backpropagation]]; standard results, row
  convention $Y=XW^\top+\mathbf b$, upstream gradient $G=\partial L/\partial Y$):
  $$\frac{\partial L}{\partial W} = G^\top X,\qquad \frac{\partial L}{\partial \mathbf b} = \sum_{i=1}^n G_{i,:},\qquad \frac{\partial L}{\partial X} = G\,W$$
  $G^\top X$ has shape $d_{out}\times d_{in}$, the same as $W$; the bias gradient sums
  over the batch rows; $GW$ passes the gradient on to the previous layer.
- **Embedding layer = one-hot × linear layer.** An [[embedding-layer]] is "a more
  efficient way" to do [[one-hot-encoding]] followed by a fully connected layer without
  bias: $\mathbf e_i^\top W = W_{i,:}$ (§2.7–2.8, book p. 43). The lookup skips the
  multiplication by zeros.
- **Weight tying.** GPT-2's output head and token embedding both have shape
  $50{,}257\times768$ (§4.6, book p. 121); reusing one matrix for both is [[weight-tying]].
- **Low-rank updates.** [[low-rank-adaptation|LoRA]] wraps a frozen linear layer:
  $\mathbf y = \underbrace{\mathbf xW^\top + \mathbf b}_{\texttt{self.linear(x)}} + \tfrac{\alpha}{r}\,\mathbf xAB$, with $A\in\mathbb R^{d_{in}\times r}$,
  $B\in\mathbb R^{r\times d_{out}}$ (the book's `LoRALayer` computes `(self.alpha / self.rank) * (x @ self.A @ self.B)`,
  input on the left; $A$ gets Kaiming-uniform init, $B$ starts at zeros, book p. 329), adding only
  $r(d_{in}+d_{out})$ parameters instead of $d_{in}d_{out}$ (`LinearWithLoRA`, §E.4, book
  pp. 329–330). In GPT-2 small, every attention projection, feed-forward and output
  `Linear` gets upgraded this way (Fig. E.4, book p. 331). See [[low-rank-approximation]].

### From llm-ch06-fine-tuning-for-classification
In Ch. 6 the book replaces `model.out_head`, a `Linear(768, 50257, bias=False)` layer, with a new `torch.nn.Linear(in_features=768, out_features=2)` (§6.5, book pp. 184–186). Read it as $\mathbf z = \mathbf h W^\top + \mathbf b$: $\mathbf h \in \mathbb R^{768}$ is the last token's hidden vector, $W \in \mathbb R^{2\times 768}$ is the weight matrix (stored as `[out, in]`), $\mathbf b \in \mathbb R^{2}$ is the bias, and $\mathbf z \in \mathbb R^{2}$ holds the 2 [[logits]]. Parameter count: $768\cdot 2 + 2 = 1{,}538$, against $768 \cdot 50{,}257 = 38{,}597{,}376$ for the old head. A new layer has `requires_grad=True` by default, so it is trainable even after the rest of the model is frozen ([[layer-freezing]]). Full treatment: [[classification-head]].

## Connections
- Builds on: [[matrix]], [[vector]], [[matrix-multiplication]], [[inner-product]], [[linear-mapping]], [[affine-space]], [[bias-vector]], [[matrix-transpose]]
- Leads to: [[multilayer-perceptron]], [[feed-forward-network]], [[query-key-value]], [[output-projection]], [[classification-head]], [[low-rank-adaptation]]
- Used by: [[self-attention]], [[multi-head-attention]], [[transformer-block]], [[gpt-architecture]], [[logistic-regression]]
- Related: [[pytorch-nn-module]], [[weight-initialization]], [[embedding-layer]], [[weight-tying]], [[activation-function]], [[parameter-count-and-memory]], [[model-parameters]], [[backpropagation]]
- People: —

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** The sidebar "Weight parameters" in the Ch. 5 intro says that in PyTorch the trainable weights live in linear layers. After `new_layer = torch.nn.Linear(...)` you can read them through `new_layer.weight`, and `model.parameters()` returns every trainable weight and bias of a model (book p. 130). The training loop in §5.2 passes exactly `model.parameters()` to AdamW (book p. 149). In §5.5, loading OpenAI's GPT-2 weights means copying (transposed) arrays into the `.weight` and `.bias` of each `nn.Linear` (book pp. 164–166). See [[model-parameters]] and [[pretrained-weights]].

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7–2.8 (embedding = one-hot + fully connected layer), book pp. 41–49 (note on p. 43).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4.2 (`nn.Linear` for Q/K/V, optimized init, transposed storage, Exercise 3.1), book pp. 70–74.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3 (feed-forward Linear layers), book p. 107; §4.6 (`out_head`), book p. 119.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (definition, shape `[30, 50]`, 2,213 parameters, `AddmmBackward0`, Fig. A.9), book pp. 265–269.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C (Exercise 3.1 solution, Exercise 4.1: 4,722,432 vs 2,360,064), book pp. 300–312.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4 (`LinearWithLoRA`, Fig. E.4), book pp. 328–336.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 184–186 (`Linear(768, 2)` classification head, trainable by default) — see [[llm-ch06-fine-tuning-for-classification]].
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 5 intro, book p. 130 (sidebar "Weight parameters"); §5.2, book p. 149 (`model.parameters()` to AdamW); §5.5, book pp. 164–166 (`load_weights_into_gpt`).
