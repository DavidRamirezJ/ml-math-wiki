---
title: Layer Normalization
type: concept
tags: [deep-learning, normalization, statistics, llm, training-stability]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Layer Normalization

> **One-liner:** Layer normalization takes each token's vector and re-centers it to average 0 and re-scales it to spread 1 — then lets the model learn its own preferred scale and offset — so the numbers flowing through a deep network never drift too big or too small.

*Also known as:* LayerNorm.

## 🟢 Intuition (Dummy)
Imagine students from different schools sending in test scores: one school grades
out of 10, another out of 1,000. Before comparing, you convert everyone's scores
to "how far above or below average, in units of typical spread" — a
[[standardization-z-score|z-score]]. Now every list is on the same scale.

LayerNorm does that to every token's vector inside the network, at many points.
Without it, the numbers coming out of one layer can slowly get huge or tiny as
they pass through dozens of layers, and training becomes unstable — the
[[vanishing-and-exploding-gradients]] problem the book uses to motivate it
(§4.2, p. 99). After standardizing, LayerNorm also has two learnable dials (a
**scale** and a **shift**), so if the model *wants* a different spread, it can
learn it.

## 🟡 Mechanics (Practitioner)

### The formula (one token vector $\mathbf{x} = (x_1,\dots,x_n)$)
$$
\mu = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2
$$
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad
y_i = \gamma_i\,\hat{x}_i + \beta_i
$$

Symbol by symbol:
- $n$ = number of features in the vector (the [[embedding-dimension]], 768 in GPT-2 small).
- $\sum_{i=1}^{n}$ = "add up over all $n$ features".
- $\mu$ (mu) = the [[expected-value-and-mean|mean]] of this one vector.
- $\sigma^2$ (sigma squared) = its [[covariance-and-variance|variance]]; $\sqrt{\sigma^2} = \sigma$ is the standard deviation.
- $\varepsilon$ (epsilon) = a tiny constant, `1e-5`, added "to prevent division by zero" (p. 103).
- $\hat{x}_i$ (x-hat) = the standardized feature: mean 0, variance ≈ 1.
- $\gamma_i$ (gamma) = learned **scale**, starts at 1. $\beta_i$ (beta) = learned **shift**, starts at 0.
- $y_i$ = the final output.

Key point: the mean and variance are computed **across the features of one token**
(the last dimension), not across the batch.

### Worked example from the book (§4.2, pp. 100–104)
A `Linear(5, 6)` + ReLU layer on two random inputs (seed 123) gives:
```
out = [[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
       [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]]
mean = [[0.1324], [0.2170]]      var = [[0.0231], [0.0398]]
```
Standardize, `(out - mean) / torch.sqrt(var)`:
```
[[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
 [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]]
mean ≈ [[-5.9605e-08], [1.9868e-08]]    var = [[1.], [1.]]
```
Check one entry by hand: $(0.3470 - 0.1324)/\sqrt{0.0231} = 0.2146 / 0.1520 \approx 1.412$ ✓.
The tiny nonzero means like $-5.96\times10^{-8}$ come from the finite precision of
[[floating-point-precision|floating-point]] numbers (p. 102).

> ⚠️ Book inconsistency: Figure 4.5 (p. 100) labels the same row "Variance = 0.39",
> but the code prints 0.0231 (p. 101). The code value matches the normalized outputs;
> 0.39 appears to be a typo in the figure.

### The PyTorch class (Listing 4.2, p. 103)
```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # beta

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```
- `dim=-1` = "the last dimension" — the embedding dimension, whether the tensor is
  2-D `[rows, features]` or 3-D `[batch, tokens, emb_dim]` (p. 102).
- `keepdim=True` keeps the result as shape `[..., 1]` instead of `[...]`, so
  `x - mean` [[broadcasting|broadcasts]] correctly (p. 101).
- `LayerNorm(emb_dim=6)` on the example gives mean ≈ 0 and variance **0.9995** and
  **0.9997** — just under 1 because of $\varepsilon$ (p. 104).

### Biased variance: why $1/n$ and not $1/(n-1)$ (sidebar, p. 103)
`unbiased=False` divides by $n$, **without** [[bessels-correction-biased-variance|Bessel's correction]]:

$$
\sigma^2_{\text{biased}} = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2
\qquad\text{vs.}\qquad
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\mu)^2
$$

- $s^2$ = the "unbiased" sample variance (PyTorch's default for `.var()`).

For $n = 768$ the difference is negligible ($767/768 \approx 0.9987$). The book
uses $1/n$ to match GPT-2's original TensorFlow implementation and so the
pretrained weights load cleanly (p. 103).

> ⚠️ Book cross-reference error: the sidebar says the weights are loaded "in
> chapter 6"; they are actually downloaded and loaded in §5.5 (pp. 161–166). Chapter 6 reuses them.

### Where LayerNorm sits in GPT-2
The book's overview (p. 99) says LayerNorm is "typically applied before and after the
multi-head attention module" and before the final output layer; the code makes this
precise:
- **Two per [[transformer-block]]:** `norm1` before attention, `norm2` before the
  feed-forward network — **Pre-LayerNorm** (§4.5, p. 115).
- **One final LayerNorm** after the last block, right before the output head (§4.6, p. 119).
- So GPT-2 small has $12 \times 2 + 1 = 25$ LayerNorms, each with $2 \times 768 = 1{,}536$
  parameters (our arithmetic).
- When loading OpenAI's weights, their `g` (gain) and `b` (bias) map to our `scale`
  and `shift`: `ln_1`, `ln_2` per block and the top-level `g`, `b` for the final
  norm (§5.5, p. 166).
- In classification fine-tuning, the final LayerNorm is unfrozen together with the
  last block (§6.5, p. 186).

### LayerNorm vs. BatchNorm (sidebar, pp. 104–105)
[[batch-normalization]] normalizes each feature **across the batch**; LayerNorm
normalizes each example **across its features**. Because LayerNorm doesn't depend
on batch size, it is more flexible and stable for LLMs, especially in distributed
training or on limited hardware.

## 🔴 Theory (Expert)
- **Centering is an orthogonal projection.** Subtracting the mean is
  $\mathbf{x} - \mu\mathbf{1} = \big(I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top\big)\mathbf{x}$,
  the [[orthogonal-projection]] onto the hyperplane $\{\mathbf{v}: \mathbf{1}^\top\mathbf{v} = 0\}$
  (the [[orthogonal-complement]] of $\mathbf{1}$). Then
  $\sigma^2 = \tfrac{1}{n}\lVert\mathbf{x}-\mu\mathbf{1}\rVert^2$, so (with $\varepsilon = 0$)
  $\lVert\hat{\mathbf{x}}\rVert_2 = \sqrt{n}$. **Before** $\gamma, \beta$, every
  normalized vector lies on the sphere of radius $\sqrt{n}$ inside that hyperplane.
- **Invariances.** For $a > 0$ and any $b$:
  $\operatorname{LN}(a\mathbf{x} + b\mathbf{1}) = \operatorname{LN}(\mathbf{x})$
  (ignoring $\varepsilon$). The network's output is therefore insensitive to the
  overall scale and offset of a token's activations — which is exactly why
  activations can't drift out of control.
- **Jacobian.** Differentiating these invariances: the Jacobian $J$ of
  $\mathbf{x}\mapsto\hat{\mathbf{x}}$ satisfies $J\mathbf{1} = \mathbf{0}$ and
  $J\mathbf{x} = \mathbf{0}$. Explicitly,
  $J = \tfrac{1}{\sigma}\big(I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top - \tfrac{1}{n}\hat{\mathbf{x}}\hat{\mathbf{x}}^\top\big)$
  — a scaled projection. Large inputs (big $\sigma$) give proportionally smaller
  gradients, which stabilizes training.
- **Why $\varepsilon$ shrinks the variance.** With $\varepsilon > 0$,
  $\operatorname{Var}(\hat{\mathbf{x}}) = \sigma^2/(\sigma^2+\varepsilon)$. In the
  book's example the biased variances are about 0.0193 and 0.0332, giving 0.9995 and
  0.9997 — exactly the printed values (our check).
- **Pre-LN vs Post-LN.** The original transformer applied LayerNorm *after* each
  residual add (Post-LN), which "often leads to worse training dynamics"
  (p. 115). Pre-LN leaves the residual stream un-normalized until the final norm,
  giving an identity path for gradients (Xiong et al. 2020; Tie et al. 2023 —
  appendix B, p. 292). See [[pre-layernorm-vs-post-layernorm]].
- **RMSNorm.** Many modern LLMs replace LayerNorm with [[rmsnorm]]
  (Zhang & Sennrich 2019): $\hat{x}_i = x_i / \sqrt{\tfrac{1}{n}\sum_j x_j^2 + \varepsilon}$
  — no mean subtraction, only scale invariance, and cheaper to compute (appendix B,
  pp. 292–293).
- **Origin.** LayerNorm was introduced by Ba, Kiros & Hinton (2016) to stabilize
  hidden-state dynamics and reduce training time (appendix B, p. 292).

## Connections
- Builds on: [[expected-value-and-mean]], [[covariance-and-variance]], [[standardization-z-score]], [[bessels-correction-biased-variance]], [[broadcasting]], [[tensor]]
- Leads to: [[transformer-block]], [[pre-layernorm-vs-post-layernorm]], [[rmsnorm]], [[gpt-architecture]]
- Compared with: [[batch-normalization]]
- Solves: [[vanishing-and-exploding-gradients]], [[numerical-stability]]
- Related: [[orthogonal-projection]], [[orthogonal-complement]], [[norm]], [[affine-space]], [[gpt-2]], [[floating-point-precision]]
- People: [[jimmy-lei-ba]], [[geoffrey-hinton]], [[friedrich-wilhelm-bessel]], [[biao-zhang]], [[rico-sennrich]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 96–99 (`DummyLayerNorm` placeholder, `eps=1e-5`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 99–105 (Figures 4.5–4.7, Listing 4.2, sidebars "Biased variance" and "Layer normalization vs. batch normalization")
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book pp. 113–116 (norm1/norm2, Pre-LN vs Post-LN)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 117–119 (final LayerNorm); §4.7 / Chapter 4 summary, book p. 127
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 165–166 (loading `ln_1`, `ln_2`, `g`, `b`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book p. 186 (unfreezing the final LayerNorm)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 292–293 (Ba et al. 2016; Xiong et al. 2020; Tie et al. 2023; Zhang & Sennrich 2019)
