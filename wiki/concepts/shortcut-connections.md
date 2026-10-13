---
title: Shortcut (Residual) Connections
type: concept
tags: [deep-learning, architecture, transformer, training, llm]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Shortcut (Residual) Connections

> **One-liner:** A shortcut connection adds a layer's input straight onto its output, so information (and, during training, the learning signal) always has an express lane that skips the layer.

*Also called:* **skip connection**, **residual connection** — three names for the same idea (book p. 109). The "residual network" (ResNet) idea is covered on this page too.

## 🟢 Intuition (Dummy)
Picture a long line of people playing "telephone". Each person whispers the message to
the next one, and each person garbles it a bit. After 50 people, the message is mush.
Now give every person a photocopy of the message they *received*, and let them hand on
"the original **plus** my small edit". Even after 50 people, the original is still
there — each person only adds a correction on top.

A **shortcut connection** does exactly that inside a neural network. Instead of
"output = what the layer computed", it says "output = input **+** what the layer
computed". The layer only has to learn a small *change* (a "residual") to the input.

Why does this matter? Training works by sending blame (the [[gradient]]) **backward**
through all the layers ([[backpropagation]]). In a deep network without shortcuts that
blame is multiplied by many small numbers and fades to almost nothing before it reaches
the first layers — the [[vanishing-and-exploding-gradients|vanishing gradient problem]].
The shortcut is an express lane: the blame can flow straight back along it, so even the
earliest layers keep learning. This is why every [[transformer-block]] in GPT has two of
them (book pp. 109, 116).

## 🟡 Mechanics (Practitioner)

### The formula
A plain layer (no shortcut), for layer $\ell$:
$$\mathbf{x}_{\ell+1} = f_\ell(\mathbf{x}_\ell)$$
The same layer **with** a shortcut:
$$\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + f_\ell(\mathbf{x}_\ell)$$
Reading it symbol by symbol:
- $\mathbf{x}_\ell$ = the [[vector]] that goes **into** layer $\ell$ (in the book's toy example, 3 numbers).
- $f_\ell(\cdot)$ = whatever the layer computes. In the toy example it is a [[linear-layer]] followed by [[gelu]]: $f_\ell(\mathbf x)=\mathrm{GELU}(W_\ell\mathbf x+\mathbf b_\ell)$, where $W_\ell$ is a weight [[matrix]] and $\mathbf b_\ell$ a [[bias-vector]].
- $+$ = ordinary element-wise vector addition. **This plus sign is the shortcut.**
- $\mathbf{x}_{\ell+1}$ = what comes **out**, and goes into the next layer.

One rule: you can only add two vectors of the **same shape**. So a shortcut is only
possible when the layer keeps the size of its input (book p. 111). That is one big
reason the transformer block is designed to keep the shape `[batch, tokens, 768]`
unchanged (book p. 116).

### The book's experiment (§4.4, Listing 4.5, book pp. 110–113)
A 5-layer toy network; each layer is `nn.Linear` + `GELU`. A flag switches the shortcut on or off:
```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i+1]), GELU())
            for i in range(5)])          # (book writes the five layers out one by one)
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output     # the shortcut
            else:
                x = layer_output
        return x
```
Setup: `layer_sizes = [3, 3, 3, 3, 3, 1]`, `sample_input = torch.tensor([[1., 0., -1.]])`,
`torch.manual_seed(123)`. The last layer goes 3 → 1, so its shape changes and **no
shortcut is applied on layer 5** (only layers 1–4 get one). A helper `print_gradients`
runs a forward pass, computes a [[mean-squared-error]] loss against the target 0, calls
`loss.backward()`, and prints, for each weight matrix, the **mean absolute gradient**:
$$\bar g = \frac{1}{mk}\sum_{i=1}^{m}\sum_{j=1}^{k}\left|\frac{\partial \mathcal{L}}{\partial W_{ij}}\right|$$
- $W$ = one layer's weight matrix, of size $m\times k$ (here $3\times3$, so $mk=9$).
- $\partial\mathcal L/\partial W_{ij}$ = the [[partial-derivative]] of the loss with respect to one weight — "how much the loss changes if I nudge this weight".
- $|\cdot|$ = absolute value; $\frac{1}{mk}\sum\sum$ = average over all entries. So $\bar g$ is one number per layer that says "how strong is the learning signal here?"

The results (book pp. 112–113):

| Layer (code name) | Without shortcut | With shortcut |
|---|---|---|
| `layers.0` (first) | 0.000202 | 0.2217 |
| `layers.1` | 0.000120 | 0.2069 |
| `layers.2` | 0.000715 | 0.3290 |
| `layers.3` | 0.001399 | 0.2666 |
| `layers.4` (last) | 0.005050 | 1.3259 |

Without shortcuts the signal shrinks by about 25× from the last layer to the first — the
vanishing gradient. With shortcuts, the last layer still has the largest gradient, but
the others **stabilize** around 0.2–0.3 instead of collapsing (book p. 113). (Our
arithmetic: the first layer's gradient is ≈1,100× bigger with shortcuts, 0.2217 / 0.000202.)
Figure 4.12 (book p. 110) shows the same numbers, truncated (e.g. 0.001399 → 0.0013,
0.3290 → 0.32, 1.3259 → 1.32). Figure layer $k$ = code name `layers.(k-1)`.

### Shortcuts inside the GPT transformer block (§4.5, Listing 4.6, book p. 115)
Each [[transformer-block]] has **two** shortcuts — one around attention, one around the
[[feed-forward-network]]:
```python
def forward(self, x):
    shortcut = x                 # shortcut for the attention sub-layer
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut             # add the original input back
    shortcut = x                 # shortcut for the feed-forward sub-layer
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut
    return x
```
In math:
$$\mathbf h = \mathbf x + \operatorname{Dropout}\big(\operatorname{MHA}(\operatorname{LN}_1(\mathbf x))\big),\qquad
\mathbf y = \mathbf h + \operatorname{Dropout}\big(\operatorname{FFN}(\operatorname{LN}_2(\mathbf h))\big)$$
- $\mathbf x$ = block input, shape `[batch, tokens, 768]`; $\mathbf h$ = result after the attention half; $\mathbf y$ = block output (same shape).
- $\operatorname{LN}_1,\operatorname{LN}_2$ = the two [[layer-normalization]] modules (`norm1`, `norm2`).
- $\operatorname{MHA}$ = masked [[multi-head-attention]]; $\operatorname{FFN}$ = feed-forward network.
- $\operatorname{Dropout}$ = `drop_shortcut`, the [[dropout]] on each sub-layer's output (rate 0.1 in `GPT_CONFIG_124M`).
- The two "$\mathbf x+$" and "$\mathbf h+$" terms are the shortcuts.

Note where the LayerNorm sits: **inside** the side branch, before attention/FFN, and
*not* on the shortcut path. That is the [[pre-layernorm-vs-post-layernorm|Pre-LayerNorm]]
design (book p. 115). The book checks the shape: `torch.rand(2, 4, 768)` in →
`torch.Size([2, 4, 768])` out (book p. 116). Because every block preserves shape,
the GPT can stack 12 of them (GPT-2 124M) or 48 (GPT-2 XL) (book p. 117).
Figures 4.13 (book p. 114), 4.15 (book p. 118) and 5.17 (book p. 163) all draw the two
⊕ nodes per block.

## 🔴 Theory (Expert)
- **Why the gradient survives (chain rule).** Our addition, standard and consistent with
  the book's claim (p. 109). Differentiate $\mathbf x_{\ell+1}=\mathbf x_\ell+f_\ell(\mathbf x_\ell)$:
  $$\frac{\partial \mathbf{x}_{\ell+1}}{\partial \mathbf{x}_\ell} = I + \frac{\partial f_\ell}{\partial \mathbf{x}_\ell}$$
  - $\partial\mathbf x_{\ell+1}/\partial\mathbf x_\ell$ = the [[jacobian]] of one block; $I$ = the identity matrix.
  Through $L$ blocks, by the [[chain-rule]],
  $\frac{\partial\mathbf x_L}{\partial\mathbf x_0}=\prod_{\ell}\big(I+J_\ell\big)$. Expanding the product gives
  $I + \sum_\ell J_\ell + \sum_{\ell<m}J_mJ_\ell+\dots$ — there is always a pure identity
  term, so the gradient never *has* to pass through a long product of small Jacobians.
  Without shortcuts you only get $\prod_\ell J_\ell$, whose norm can shrink (or blow up)
  geometrically with depth ([[vanishing-and-exploding-gradients]]).
- **Residual learning.** He, Zhang, Ren & Sun (2015/2016), "Deep Residual Learning for
  Image Recognition" (ResNet) — the computer-vision origin the book mentions without a
  citation (p. 109). The argument: if the ideal block is close to the identity, it is easier
  to learn $f\approx 0$ than to learn $f\approx\mathrm{id}$ from scratch. At random
  initialization, a residual network starts near the identity map.
- **Ensemble view.** The expansion of $\prod_\ell(I+J_\ell)$ shows a residual net of depth
  $L$ behaves like a sum over $2^L$ paths of different lengths; most gradient flows
  through the short paths (Veit et al., 2016 — external).
- **The residual stream.** In a Pre-LN transformer, the vector $\mathbf x$ that flows along
  the shortcuts is a running sum: $\mathbf x_L=\mathbf x_0+\sum_\ell(\text{attention and FFN updates})$.
  Each sub-layer *reads* a normalized copy and *writes* an additive update. The book adds
  one more [[layer-normalization|final LayerNorm]] before the output head, "standardizing
  the outputs from the transformer blocks to stabilize the learning process" (book pp. 117–119).
  Our reading (not the book's): this is needed because the un-normalized sum's scale grows with depth.
- **Variants.** Appendix B lists *ResiDual* (Tie et al., 2023), which combines Pre-LN and
  Post-LN residual paths, and *Simplifying Transformer Blocks* (He & Hofmann, 2023), which
  revisits which block parts (value matrix, output projection) are needed (book p. 292).
- **Shape constraint.** When dimensions differ (e.g. the toy net's 3 → 1 layer), ResNets use
  a learned projection $W_s\mathbf x + f(\mathbf x)$; the book simply skips the shortcut there.

## Connections
- Builds on: [[vector]], [[gradient]], [[chain-rule]], [[jacobian]], [[backpropagation]], [[neural-network]], [[linear-layer]], [[gelu]]
- Leads to: [[transformer-block]], [[gpt-architecture]], [[pre-layernorm-vs-post-layernorm]]
- Fixes / related: [[vanishing-and-exploding-gradients]], [[layer-normalization]], [[dropout]], [[deep-learning]]
- Used by: [[gpt-2]], [[transformer-architecture]], [[decoder-only-transformer]], [[llm-training-loop]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4, book pp. 109–113 (Fig. 4.11, Fig. 4.12, Listing 4.5, `print_gradients` results).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book pp. 113–117 (Fig. 4.13, Listing 4.6, shape-preservation discussion).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 117–118 (Fig. 4.15, 12 / 48 stacked blocks); §4.7 chapter summary, book p. 127.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book p. 163 (Fig. 5.17, two shortcuts per block).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 4 and Ch. 3 references, book p. 292 (ResiDual; Simplifying Transformer Blocks).
- Chain-rule/Jacobian argument, ResNet citation, ensemble view: standard results, not in the book.
