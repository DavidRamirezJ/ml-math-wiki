---
title: Backpropagation
type: concept
tags: [calculus, deep-learning, optimization, differentiation, pytorch, llm-training]
difficulty: advanced
sources: [mml-ch05-vector-calculus, llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-06-04
updated: 2026-10-13
---

# Backpropagation

> **One-liner:** Backpropagation computes a neural network's gradients cheaply by applying the chain rule backward through the network, reusing work at every layer.

## 🟢 Intuition (Dummy)
A deep network is a tower of functions: the input passes through layer after
layer until it produces a prediction, and an error score says how wrong it was.
To improve, every knob (weight) needs to know "if I nudge you, how much does the
error change?" Computing that one knob at a time, from scratch, would be hugely
wasteful. **Backpropagation** is the clever shortcut: start the blame at the
output error and pass it *backward* through the layers, so each layer reuses the
blame already computed by the layer above it. The total work to get *all* the
gradients ends up about the same as running the network once forward.

### In LLMs
*Build a Large Language Model (From Scratch)* calls backpropagation "the standard
technique for training deep neural networks" and puts it simply: to make the model
give higher probability to the right next token, we nudge the weights — and
backpropagation tells us which way to nudge each one (§5.1, book p. 137). The good news
for you as a builder: in PyTorch you never write it by hand. You compute a loss, call
**`loss.backward()`**, and PyTorch fills in the gradient of every one of GPT-2's ~124
million weights for you (§A.4, book p. 264; §4.4, book p. 112).

It also explains a few design choices you'll see all over the book:
- Tokens are turned into **continuous** embedding vectors *because* LLMs are trained
  with backpropagation, which needs smooth numbers it can nudge (§2.7, book p. 42).
- Attention scores are divided by $\sqrt{d_k}$ so the [[softmax]] doesn't get so
  "peaky" that backprop's gradients become tiny (§3.4, book p. 69).
- [[shortcut-connections]] give gradients a short highway back to the early layers so
  they don't fade away (§4.4, book p. 109).

## 🟡 Mechanics (Practitioner)
A network is a many-level [[chain-rule|composition]] (5.111):
$$\mathbf y=(f_K\circ f_{K-1}\circ\cdots\circ f_1)(\mathbf x),$$
where each layer (5.113) is $\mathbf f_i=\sigma_i(A_{i-1}\mathbf f_{i-1}+\mathbf b_{i-1})$
with weights $A_{i-1}$, biases $\mathbf b_{i-1}$, and activation $\sigma$
(sigmoid $\tfrac{1}{1+e^{-x}}$, $\tanh$, or ReLU). With squared loss
$L(\boldsymbol\theta)=\lVert\mathbf y-\mathbf f_K(\boldsymbol\theta,\mathbf x)\rVert^2$
and parameters $\boldsymbol\theta=\{A_0,\mathbf b_0,\dots\}$, the chain rule gives
the gradients **layer by layer**:
$$
\frac{\partial L}{\partial\theta_{K-1}}=\frac{\partial L}{\partial \mathbf f_K}\frac{\partial \mathbf f_K}{\partial\theta_{K-1}},\qquad
\frac{\partial L}{\partial\theta_{i}}=\frac{\partial L}{\partial \mathbf f_K}\frac{\partial \mathbf f_K}{\partial \mathbf f_{K-1}}\cdots\frac{\partial \mathbf f_{i+1}}{\partial\theta_i}.
$$
The key efficiency: once $\partial L/\partial\theta_{i+1}$ (equivalently
$\partial L/\partial\mathbf f_{i+1}$) is known, computing $\partial L/\partial\theta_i$
**reuses** it — only the local factors $\partial\mathbf f_{i+1}/\partial\mathbf f_i$
(output-w.r.t.-input) and $\partial\mathbf f_{i+1}/\partial\theta_i$
(output-w.r.t.-parameters) are new. Gradients are passed *backward* through the
network (Fig. 5.9).

**The algorithm:**
1. **Forward pass:** compute and cache each layer's output $\mathbf f_i$ and the loss $L$.
2. **Backward pass:** initialize $\partial L/\partial\mathbf f_K$, then for $i=K-1,\dots,0$ multiply by the cached local Jacobians to get $\partial L/\partial\mathbf f_i$ and read off $\partial L/\partial\theta_i$.
3. **Update:** take a gradient-descent step on each $\theta_i$ (Ch. 7).

### In LLMs — a tiny worked example you can check by hand (§A.3–A.4)
The LLM book's Appendix A shows backprop on the smallest possible "network": one
[[logistic-regression]] neuron (Figure A.8, book p. 263). The forward
[[computational-graph]] is:
$$u = w_1 x_1,\qquad z = u + b,\qquad a = \sigma(z),\qquad L = L(a, y)$$
- $x_1$ = the input feature (here $1.1$); $y$ = the true label (here $1.0$).
- $w_1$ = the trainable weight ($2.2$); $b$ = the trainable bias ($0.0$).
- $u$ = the intermediate product; $z$ = the intermediate result ("net input").
- $\sigma$ = the [[sigmoid-function]], $\sigma(z)=1/(1+e^{-z})$; $a$ = its output, a probability.
- $L$ = the [[binary-cross-entropy]] loss comparing $a$ with $y$.

Backprop multiplies the local derivatives **from right (loss) to left (parameters)**:
$$\frac{\partial L}{\partial w_1} = \frac{\partial u}{\partial w_1}\times\frac{\partial z}{\partial u}\times\frac{da}{dz}\times\frac{\partial L}{\partial a},
\qquad
\frac{\partial L}{\partial b} = \frac{\partial z}{\partial b}\times\frac{da}{dz}\times\frac{\partial L}{\partial a}$$
Reading it symbol by symbol:
- $\frac{\partial L}{\partial w_1}$ = "how much the loss changes if I nudge $w_1$ a tiny bit, holding everything else fixed" (a [[partial-derivative]]).
- Each factor is one edge of the graph: e.g. $\frac{\partial u}{\partial w_1} = x_1$ (because $u = w_1x_1$), $\frac{\partial z}{\partial u}=\frac{\partial z}{\partial b}=1$ (because $z=u+b$), $\frac{da}{dz} = \sigma(z)(1-\sigma(z))$.
- $\times$ = ordinary multiplication — this *is* the [[chain-rule]].
- Notice the shared tail $\frac{da}{dz}\times\frac{\partial L}{\partial a}$: it's computed **once** and reused for both $w_1$ and $b$. That reuse is the whole trick.

The numbers (wiki check, matching the book's printout):
| Quantity | Value |
|---|---|
| $z = 1.1\times2.2+0$ | $2.42$ |
| $a=\sigma(2.42)$ | $\approx 0.9183$ |
| $L=-\ln a$ | $\approx 0.0852$ |
| $\partial L/\partial z = a-y$ (sigmoid + BCE simplify) | $\approx -0.0817$ |
| $\partial L/\partial b = 1\times(-0.0817)$ | **$-0.0817$** (book: `tensor([-0.0817])`) |
| $\partial L/\partial w_1 = 1.1\times(-0.0817)$ | **$-0.0898$** (book: `tensor([-0.0898])`) |

Both gradients are negative, so gradient descent will *increase* $w_1$ and $b$, pushing
$a$ closer to the label $1$. The PyTorch code (Listing A.3, book p. 264):
```python
y  = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b  = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)
loss.backward()          # backprop through the recorded graph
print(w1.grad, b.grad)   # -0.0898, -0.0817
```
- `requires_grad=True` marks **leaf** tensors whose gradients we want; PyTorch only
  builds the graph if a leaf has this flag (§A.4, book p. 263).
- `.backward()` fills the `.grad` attribute of *every* leaf in one sweep. The lower-level
  `torch.autograd.grad(loss, w1, retain_graph=True)` gives one gradient at a time;
  `retain_graph=True` is needed because PyTorch normally **frees the graph** right after
  a backward pass to save memory (book p. 264).
- Tensors carry a `grad_fn` tag naming the last operation that made them — e.g.
  `grad_fn=<AddmmBackward0>` after a [[linear-layer]] (matrix multiply + add, §A.5,
  book p. 269), `<EmbeddingBackward0>` after an [[embedding-layer]] lookup (§2.7,
  book p. 43). That is the backward recipe for that node.

### In LLMs — backprop inside the training loop
Every training loop in the book (§5.2, §6.7, §A.7, §D.4) has the same three lines:
```python
optimizer.zero_grad()   # wipe last batch's gradients (PyTorch ADDS into .grad)
loss.backward()         # backpropagation: compute dLoss/dtheta for every parameter
optimizer.step()        # use those gradients to update the weights
```
(§A.7, book pp. 275–276; §5.2, book p. 147). Backprop needs a **[[loss-function]]**
to start from — the scalar that measures "how far off" the predictions are (§5.1,
book p. 137). For GPT pretraining that's the [[cross-entropy-loss]] over next tokens;
the untrained GPT-2 starts at a loss of about 10.79 on the two example texts (book p. 137).

Things that touch the gradients **between** `backward()` and `step()`:
- **[[gradient-clipping]]**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
  rescales the whole gradient so its L2 [[norm]] is at most 1 (§D.3–D.4, book pp. 317–320).
  The book's 2×2 example scales a gradient matrix by $\text{max\_norm}/\lVert G\rVert_2 = 1/5$
  (§D.3, book p. 318).

  > ⚠️ Book erratum: the book prints $G=\begin{bmatrix}1&2\\3&4\end{bmatrix}$ but computes
  > $\lVert G\rVert_2=\sqrt{1^2+2^2+2^2+4^2}=5$; for the printed matrix the norm is
  > $\sqrt{30}\approx5.48$. The arithmetic matches $G=\begin{bmatrix}1&2\\2&4\end{bmatrix}$.

  In the book's GPT example the largest gradient entry drops from 0.0411 to 0.0185 after
  clipping (book p. 319).

Ways to **switch backprop off**:
- `with torch.no_grad():` for inference/evaluation — no graph is built, saving memory and
  compute (§A.5, book p. 269; §5.1, book p. 134; §6.8).
- `param.requires_grad = False` to **freeze** weights: they get no gradient and don't
  change. [[low-rank-adaptation|LoRA]] freezes all 124,441,346 GPT-2 weights and lets
  backprop reach only the 2,666,528 LoRA weights (§E.4, book pp. 331–332). See [[layer-freezing]].

### In LLMs — watching gradients vanish (§4.4)
The book measures backprop's output layer by layer on a 5-layer net (each layer
`Linear` + [[gelu]], `layer_sizes=[3,3,3,3,3,1]`, input `[1., 0., -1.]`, MSE loss vs
target 0) and prints the mean absolute gradient of each weight matrix:
$$\bar g = \frac{1}{mk}\sum_{i=1}^{m}\sum_{j=1}^{k}\left|\frac{\partial \mathcal L}{\partial W_{ij}}\right|$$
- $W$ = the layer's $m\times k$ weight matrix; $W_{ij}$ = one entry.
- $|\cdot|$ = absolute value; $\frac{1}{mk}\sum\sum$ = average over all $mk$ entries.

| Layer | without shortcut | with shortcut |
|---|---|---|
| `layers.0` (first) | 0.00020 | 0.2217 |
| `layers.1` | 0.00012 | 0.2069 |
| `layers.2` | 0.00072 | 0.3290 |
| `layers.3` | 0.00140 | 0.2666 |
| `layers.4` (last) | 0.00505 | 1.3259 |

Without shortcuts the gradient shrinks as it travels back toward the input — the
**[[vanishing-and-exploding-gradients|vanishing gradient problem]]** (book pp. 112–113).

## 🔴 Theory (Expert)
- Backpropagation is precisely **reverse-mode [[automatic-differentiation]]**
  applied to the network's computation graph. The "reverse vs. forward" choice is
  *which order* you multiply the Jacobian chain
  $\frac{dy}{dx}=\frac{dy}{db}\frac{db}{da}\frac{da}{dx}$; associativity of matrix
  multiplication makes both valid (5.120–5.121).
- **Why reverse mode for nets.** When the input dimension $\gg$ output dimension
  (typical: many parameters, scalar loss), grouping right-to-left keeps the
  intermediate objects small (row-vector $\times$ matrix), so reverse mode is far
  cheaper than forward mode.
- Cost result: computing the gradient is of the *same order* as computing the
  function — counterintuitive given how long the symbolic derivative expression
  (e.g. 5.110) is.
- Historical lineage (Kelley 1960; Bryson 1961; Dreyfus 1962; Rumelhart, Hinton &
  Williams 1986). Requires differentiable elementary operations; control flow
  (loops, `if`) needs care.

### In LLMs — deeper notes
- **Same definition in both books.** The LLM book's Figure A.8 caption names it
  "reverse-mode automatic differentiation or backpropagation" and describes computing
  the chain rule "from right to left", starting at the loss and working back to the input
  (book p. 263) — consistent with the MML account above. PyTorch's **autograd** builds the
  graph **dynamically**, by tracking every tensor operation as it runs (§A.3–A.4, book
  pp. 261–264), which is why ordinary Python control flow is allowed.
- **Why shortcuts fix vanishing gradients** (wiki derivation, not written in the book):
  with a residual layer $\mathbf x_{\ell+1} = \mathbf x_\ell + f_\ell(\mathbf x_\ell)$,
  $$\frac{\partial\mathbf x_{\ell+1}}{\partial\mathbf x_\ell} = I + \frac{\partial f_\ell}{\partial\mathbf x_\ell},$$
  so the backward product always contains an identity path instead of only a product
  $\prod_\ell \partial f_\ell/\partial\mathbf x_\ell$ of possibly small [[jacobian|Jacobians]].
- **Why $\sqrt{d_k}$ scaling.** For large dot products, softmax acts like a step
  function and its Jacobian $\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ goes to
  zero, so the gradient through attention vanishes (book p. 69 gives the gradient
  argument; the Jacobian form is a wiki addition). See [[scaled-dot-product-attention]].
- **Embedding gradients are sparse.** An [[embedding-layer]] equals one-hot × matrix
  (NOTE in §2.7, book p. 43), so $\partial L/\partial W = X^\top\,\partial L/\partial E$: only the
  rows of looked-up token IDs receive gradient (wiki derivation).
- **Symmetry breaking needs backprop's view.** If all weights started equal, every unit
  in a layer would get the same gradient and stay identical forever (§A.5, book p. 268) —
  see [[weight-initialization]].
- **LoRA's zero init and the first backward pass.** With $\Delta W \approx \frac{\alpha}{r}AB$
  and $B=0$ at start, $\partial L/\partial A \propto (\cdot)B^\top = 0$ on step one, but
  $\partial L/\partial B \ne 0$, so learning still starts (§E.4, book p. 329; wiki
  derivation).
- **Memory cost.** Reverse mode must cache forward activations; that's why
  `torch.no_grad()` saves memory at inference, and why the book notes that for larger
  models, where backprop is costly, LoRA usually makes training faster (§E.4, book p. 335).

## Connections
- Builds on: [[chain-rule]], [[jacobian]], [[gradient]], [[partial-derivative]], [[derivative]], [[computational-graph]], [[forward-pass]], [[loss-function]]
- Leads to: [[automatic-differentiation]], [[gradient-descent]], [[stochastic-gradient-descent]], [[adamw-optimizer]], [[gradient-clipping]], [[gradient-accumulation]]
- Used by: training deep neural networks, autoencoders, [[llm-training-loop]], [[training-loop]], [[pretraining]], [[fine-tuning]], [[low-rank-adaptation]], [[embedding-layer]]
- Related: [[vanishing-and-exploding-gradients]], [[shortcut-connections]], [[scaled-dot-product-attention]], [[weight-initialization]], [[layer-freezing]], [[pytorch]], [[logistic-regression]], [[sigmoid-function]], [[binary-cross-entropy]]
- People: [[geoffrey-hinton]]

## Sources
- `raw/mml-book.pdf` §5.6.1 (5.111–5.118), printed pp. 159–161 (PDF pp. 165–167).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7 (embeddings need backprop, p. 42; embedding = one-hot × matrix NOTE, weights optimized via backpropagation, `EmbeddingBackward0`, p. 43), book pp. 41–43; §2.8 (positional embeddings, also trained), book pp. 43–49.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4 (√d_k scaling avoids small gradients; `requires_grad`), book pp. 64–74.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4 (vanishing gradients, `print_gradients`, Listing 4.5, Fig. 4.12), book pp. 109–113.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1 (Backpropagation sidebar, Fig. 5.7), book pp. 128–146; §5.2 (training loop), book pp. 146–151.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7 (classification training loop), book pp. 195–200; §6.8 (`no_grad` inference), book pp. 200–203.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1 (autograd overview), book pp. 251–257; §A.3 (computation graphs, Fig. A.7), book pp. 261–262; §A.4 (Fig. A.8, Listing A.3, sidebar "Partial derivatives and gradients"), book pp. 263–264.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (`grad_fn`, `no_grad`, symmetry breaking), book pp. 265–269; §A.7 (`zero_grad`/`backward`/`step`), book pp. 274–278; §A.9 (DDP backward pass and gradient sync), book pp. 279–288.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.3 (gradient clipping), book pp. 317–319; §D.4 (clipping in the training function), book pp. 319–321.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.1 (ΔW learned by backprop), book pp. 322–324; §E.4 (freezing, LoRA init, trainable-parameter counts), book pp. 328–336.
