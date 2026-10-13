---
title: Vanishing and Exploding Gradients
type: concept
tags: [deep-learning, optimization, backpropagation, training-stability, llm]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Vanishing and Exploding Gradients

> **One-liner:** In a deep network the learning signal gets multiplied layer after layer on its way back, so it can fade to almost nothing (vanish) or blow up to huge values (explode) — and either way, training breaks.

*Also known as:* the vanishing-gradient problem, exploding gradients.

## 🟢 Intuition (Dummy)
Think of the game "telephone". The last kid in the line hears what went wrong
(the error) and has to whisper the correction back to the first kid. Every kid
in between repeats it a bit quieter — or, if they are excitable, a bit louder.
With 3 kids that's fine. With 50 kids, a whisper becomes silence and a shout
becomes a scream.

That's exactly what happens to the **[[gradient]]** — the "here's how to fix
yourself" message — when [[backpropagation]] sends it backward through many
layers:

- **Vanishing:** the message shrinks at every layer, so the *early* layers get
  almost no instructions and barely learn.
- **Exploding:** the message grows at every layer, so the weight updates become
  wild jumps and the loss shoots up (or becomes `NaN`).

The book introduces this as the reason deep networks are hard to train, and then
shows the three main fixes used in GPT-style models: [[layer-normalization]]
(keep activations at a sensible scale), [[shortcut-connections]] (give the
gradient an express lane), and [[gradient-clipping]] (put a speed limit on the
gradient) (§4.2, §4.4, §D.3).

## 🟡 Mechanics (Practitioner)

### Why it happens: a product of many factors
A deep network is a chain of layers $\mathbf{x}_{\ell+1} = f_\ell(\mathbf{x}_\ell)$.
By the [[chain-rule]], the gradient of the loss with respect to an early layer is

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_1}
= \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{L}}\,
\frac{\partial \mathbf{x}_{L}}{\partial \mathbf{x}_{L-1}}\cdots
\frac{\partial \mathbf{x}_{2}}{\partial \mathbf{x}_{1}}
$$

Reading it symbol by symbol:
- $\mathcal{L}$ = the loss (one number saying how wrong the model is).
- $\mathbf{x}_\ell$ = the vector coming *into* layer $\ell$; $L$ = the number of layers.
- $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_1}$ = "how much does the loss change if I wiggle the first layer's input?" — the signal the early layers learn from.
- Each $\frac{\partial \mathbf{x}_{\ell+1}}{\partial \mathbf{x}_\ell}$ = one layer's local [[jacobian]] (its "amplification factor").

So the early-layer gradient is a **product of $L-1$ factors**. If those factors are
typically a bit smaller than 1, the product decays like $0.5^{L}$ → vanishing.
If they are typically a bit bigger than 1, it grows like $2^{L}$ → exploding.
(This product view is the standard explanation; the book states the effect in
words, p. 99, and shows it numerically, p. 112.)

### The book's experiment: watch a gradient vanish (§4.4, pp. 110–113)
A toy 5-layer network, every layer = `Linear` + [[gelu|GELU]],
`layer_sizes = [3, 3, 3, 3, 3, 1]`, input `[[1., 0., -1.]]`, target `0`,
loss = [[mean-squared-error]], seed 123. After `loss.backward()` the book prints,
for each layer, the **mean absolute gradient** of its weight matrix:

$$
\bar g = \frac{1}{mk}\sum_{i=1}^{m}\sum_{j=1}^{k}\left|\frac{\partial \mathcal{L}}{\partial W_{ij}}\right|
$$

- $W$ = the layer's $m \times k$ weight matrix ($3\times 3$, so $mk = 9$ numbers, for the first four layers; $1\times 3$ for the last one).
- $\frac{\partial \mathcal{L}}{\partial W_{ij}}$ = the gradient of one weight; $|\cdot|$ = absolute value.
- $\bar g$ = their average size — one number per layer, easy to compare.

| Layer | No shortcut | With shortcut |
|---|---|---|
| `layers.4` (last) | 0.00505 | 1.3259 |
| `layers.3` | 0.00140 | 0.2666 |
| `layers.2` | 0.00072 | 0.3290 |
| `layers.1` | 0.00012 | 0.2069 |
| `layers.0` (first) | 0.00020 | 0.2217 |

Without shortcuts the gradient shrinks about 25× from the last layer to the first
layer (not perfectly monotonically: `layers.1` is smaller than `layers.0`) — the book calls this the **vanishing gradient problem** (p. 112). With
[[shortcut-connections]] the first layer's gradient is ~1,100× larger
(0.2217 / 0.000202, our arithmetic) and stays stable (p. 113).

```python
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()(output, target)
    loss.backward()                       # backpropagation
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
```

### Fix 1 — shortcut (residual) connections
Instead of $\mathbf{x}_{\ell+1} = f_\ell(\mathbf{x}_\ell)$, use
$\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + f_\ell(\mathbf{x}_\ell)$. Then

$$
\frac{\partial \mathbf{x}_{\ell+1}}{\partial \mathbf{x}_\ell} = I + \frac{\partial f_\ell}{\partial \mathbf{x}_\ell}
$$

- $I$ = the identity matrix ("pass it through unchanged").
- $\frac{\partial f_\ell}{\partial \mathbf{x}_\ell}$ = the layer's own Jacobian.

Because of the $I$, there is always a path where the gradient is *not* shrunk.
Residual connections came from ResNets in computer vision, created "to mitigate
the challenge of vanishing gradients" (p. 109). Every [[transformer-block]] has two of them.

### Fix 2 — layer normalization
[[layer-normalization]] rescales each token's activations to mean 0 and variance 1.
The book motivates LayerNorm directly with vanishing/exploding gradients:
unstable gradients make it hard for the network to find weights that minimize the
[[loss-function]] (p. 99).

### Fix 3 — scaled attention (avoid a saturated softmax)
In [[scaled-dot-product-attention]] the scores are divided by $\sqrt{d_k}$. The
book's reason: with embedding sizes above 1,000, raw dot products get large, the
[[softmax]] becomes almost a step function, and its gradients go to nearly zero. Such small
gradients "can drastically slow down learning or cause training to stagnate" (§3.4, p. 69). That is a local
vanishing-gradient problem inside attention.

### Fix 4 — gradient clipping (against exploding gradients)
[[gradient-clipping]] caps the size of the whole gradient vector (§D.3, pp. 317–319):

$$
\mathbf{g} \leftarrow \mathbf{g}\cdot\min\!\left(1,\ \frac{\text{max\_norm}}{\lVert\mathbf{g}\rVert_2}\right),
\qquad \lVert\mathbf{g}\rVert_2 = \sqrt{g_1^2 + \cdots + g_n^2}
$$

- $\mathbf{g}$ = all parameter gradients stacked into one long vector.
- $\lVert\mathbf{g}\rVert_2$ = its length (the L2 [[norm]]).
- max_norm = the allowed maximum (the book uses 1.0).
- $\min(1, \cdot)$ = "only shrink, never grow".

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
On a fresh `GPTModel(GPT_CONFIG_124M)` the largest single gradient goes from
`0.0411` to `0.0185` after clipping (p. 319). In the full training loop (§D.4,
p. 320) clipping is applied only after the [[learning-rate-warmup]] phase, "to
avoid exploding gradients".

## 🔴 Theory (Expert)
- **Norm bound.** By submultiplicativity of operator norms,
  $\big\lVert \prod_{\ell} J_\ell \big\rVert \le \prod_\ell \lVert J_\ell\rVert$.
  If every $\lVert J_\ell\rVert \le \rho < 1$, the gradient reaching layer 1 is at
  most $\rho^{L-1}$ times the output gradient — exponential decay in depth. The
  reverse (exploding) direction is governed by the *largest* singular values: if
  the Jacobians share a direction with singular value $> 1$, that component grows
  geometrically (connect to [[singular-value-decomposition]]).
- **Activation choice matters.** For a pointwise activation $\phi$, the Jacobian
  contains $\operatorname{diag}(\phi'(\mathbf{z}))$. Sigmoid has $\phi' \le 1/4$, so
  stacked sigmoids vanish quickly; ReLU/GELU have derivatives near 1 for positive
  inputs, which is one reason modern nets prefer them (standard background, not
  stated in the book).
- **Softmax saturation, made precise.** The softmax Jacobian is
  $\frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$.
  When $\mathbf{p}$ approaches one-hot, every entry goes to 0 — this is the "step
  function" behaviour the book describes (p. 69). If $q_i, k_i$ are independent
  with mean 0 and variance 1, then $\operatorname{Var}(\mathbf{q}\cdot\mathbf{k}) = d_k$;
  dividing by $\sqrt{d_k}$ restores unit variance (Vaswani et al. 2017; background).
- **Residual networks as perturbations of the identity.** With
  $\mathbf{x}_{L} = \mathbf{x}_1 + \sum_{\ell} f_\ell(\mathbf{x}_\ell)$, the
  Jacobian $\prod_\ell (I + J_\ell)$ expands into a sum over all subsets of layers,
  including the bare identity term — gradients can always reach early layers through
  "short" paths.
- **Clipping keeps the direction.** Norm clipping multiplies $\mathbf{g}$ by a
  positive scalar, so the update still points downhill; only the step length is
  capped. It is a *heuristic* — it changes the effective [[learning-rate]] on
  large-gradient steps, not the descent direction.
- **Pre-LN vs Post-LN.** Putting LayerNorm *before* each sublayer (GPT-2 style)
  keeps an un-normalized identity path from input to output, which is the usual
  explanation for Pre-LN's more stable training (see
  [[pre-layernorm-vs-post-layernorm]]; book p. 115; Xiong et al. 2020, appendix B).

> ⚠️ Book erratum (§D.3, p. 318): the worked clipping example computes the norm of
> $\begin{bmatrix}1&2\\3&4\end{bmatrix}$ as $\sqrt{1^2+2^2+2^2+4^2}=\sqrt{25}=5$ and gives $G' = \tfrac15 G$. The correct
> value is $\sqrt{30}\approx 5.477$, so the scaling factor should be
> $1/\sqrt{30}\approx 0.183$, not $1/5$. The method is right; the arithmetic has a typo.

## Connections
- Builds on: [[gradient]], [[chain-rule]], [[backpropagation]], [[jacobian]], [[norm]], [[deep-learning]]
- Leads to: [[shortcut-connections]], [[layer-normalization]], [[gradient-clipping]], [[scaled-dot-product-attention]], [[pre-layernorm-vs-post-layernorm]]
- Used by: [[transformer-block]], [[llm-training-loop]], [[learning-rate-warmup]], [[gpt-architecture]]
- Related: [[gelu]], [[softmax]], [[weight-initialization]], [[automatic-differentiation]], [[numerical-stability]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4 (sidebar "The rationale behind scaled-dot product attention"), book p. 69
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book p. 99 (motivation for layer normalization)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4, book pp. 109–113 (Listing 4.5, Figure 4.12, gradient printouts)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7 / Chapter 4 summary, book p. 127
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.3, book pp. 317–319; §D.4, book p. 320
