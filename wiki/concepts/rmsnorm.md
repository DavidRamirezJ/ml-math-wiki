---
title: RMSNorm
type: concept
tags: [deep-learning, normalization, transformer, llm]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# RMSNorm

> **One-liner:** RMSNorm is a cheaper cousin of LayerNorm: it shrinks or stretches each token's vector to a standard size, but it doesn't bother shifting it to have average zero.

## 🟢 Intuition (Dummy)
Inside a deep network, the numbers flowing from layer to layer can drift: some get huge, some get tiny, and training gets wobbly. [[layer-normalization]] fixes this with two moves on each token's vector: (1) **recentre** it (subtract its average, so the average is 0), and (2) **rescale** it (divide by its spread, so the spread is 1).

RMSNorm asks: do we really need move (1)? Its answer is "mostly no." It only does move (2): it measures the "typical size" of the numbers (their **root mean square**) and divides by that. Fewer steps, less work, and in practice the model trains just as well. The book's appendix B puts it this way: RMSNorm is "a popular variant of LayerNorm used in modern LLMs ... due to its improved computing efficiency," and it "does not center the data before computing the scale" (book pp. 292–293).

The book's own GPT-2 model uses LayerNorm (to stay compatible with OpenAI's weights). RMSNorm is the upgrade you'd meet in newer open models.

## 🟡 Mechanics (Practitioner)

### The formula
The book describes RMSNorm only in words. Here is the standard formula from the original paper (Zhang & Sennrich, 2019):
$$\mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon},\qquad \mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})}\odot \boldsymbol{\gamma}$$

Symbol by symbol:
- $\mathbf{x} = (x_1,\dots,x_d)$ = the [[vector]] of activations for **one token** (the same thing LayerNorm normalizes).
- $d$ = its length, the [[embedding-dimension]] (`emb_dim` = 768 for GPT-2 124M).
- $x_i^2$ = the $i$-th entry squared (squaring makes everything positive, so big negatives count as "big" too).
- $\sum_{i=1}^{d}$ = add over all $d$ entries; $\frac{1}{d}$ = turn the sum into an average. So $\frac{1}{d}\sum x_i^2$ is the **mean of the squares**.
- $\sqrt{\cdot}$ = square root, bringing it back to the scale of the original numbers. "Root, mean, square" read inside-out.
- $\epsilon$ = a tiny constant (like LayerNorm's `eps = 1e-5`) so we never divide by zero ([[numerical-stability]]).
- $\frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})}$ = divide every entry by that one number.
- $\boldsymbol{\gamma}$ = a learned scale vector of length $d$ (starts at all ones); $\odot$ = multiply entry by entry.

### Side by side with LayerNorm
$$\mathrm{LayerNorm}(\mathbf{x}) = \boldsymbol\gamma \odot \frac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\epsilon}} + \boldsymbol\beta \qquad\text{vs.}\qquad \mathrm{RMSNorm}(\mathbf{x}) = \boldsymbol\gamma \odot \frac{\mathbf{x}}{\sqrt{\tfrac1d\sum_i x_i^2+\epsilon}}$$
- $\mu$ = the mean of the entries ([[expected-value-and-mean]]); $\sigma^2$ = their variance ([[covariance-and-variance]]); $\boldsymbol\beta$ = LayerNorm's learned shift.
- RMSNorm drops **$\mu$** (no recentring) and **$\boldsymbol\beta$** (no shift). It keeps only the scale.

| | LayerNorm (book's GPT-2) | RMSNorm |
|---|---|---|
| Subtract the mean? | yes | **no** |
| Divides by | $\sqrt{\sigma^2+\epsilon}$ (std. dev.) | $\sqrt{\text{mean}(x^2)+\epsilon}$ (RMS) |
| Learned params per layer | `scale` + `shift` = $2d$ | $\boldsymbol\gamma$ only = $d$ |
| Output after norm | mean 0, variance ≈ 1 | mean of squares ≈ 1, mean **not** forced to 0 |

### Worked example (using the book's numbers)
§4.2 builds a `Linear(5, 6) + ReLU` layer whose first output row is (book p. 100):
$$\mathbf{x} = [0.2260,\ 0.3470,\ 0.0000,\ 0.2216,\ 0.0000,\ 0.0000]$$
The book normalizes it with LayerNorm (mean $0.1324$). Let's apply RMSNorm instead (our calculation, $\epsilon$ ignored, $\boldsymbol\gamma = \mathbf{1}$):
- Squares: $0.0511,\ 0.1204,\ 0,\ 0.0491,\ 0,\ 0$. Sum $= 0.2206$.
- Mean of squares: $0.2206/6 = 0.03677$. RMS $= \sqrt{0.03677} \approx 0.1917$.
- Divide: $\mathrm{RMSNorm}(\mathbf{x}) \approx [1.179,\ 1.810,\ 0,\ 1.156,\ 0,\ 0]$.

Compare the book's LayerNorm output for the same row: `[0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719]` (book p. 102). LayerNorm pushes the zeros *negative* to centre the row. RMSNorm leaves zeros at zero and only rescales. Its output has mean of squares $= 1$, but its mean is about $0.69$, not $0$.

A handy identity explains the link: $\mathrm{RMS}^2 = \sigma^2 + \mu^2$. Here $0.03677 \approx 0.01923 + 0.13243^2$, where $0.01923$ is the *biased* variance (divide by $d=6$). The book prints $0.0231$ for this row (p. 101) because that first manual example uses PyTorch's default unbiased `var` (divide by $d-1=5$): $0.01923 \times 6/5 \approx 0.0231$. When the mean is already near 0, the two norms almost agree.

### A short PyTorch version
Written in the style of the book's `LayerNorm` class (Listing 4.2); this code is ours, not the book's:

```python
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))   # gamma; no shift

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)
```

Recent PyTorch also ships `torch.nn.RMSNorm`. To swap it into the book's [[transformer-block]], you'd replace `norm1`, `norm2` and `final_norm`. But then the OpenAI GPT-2 [[pretrained-weights]] no longer fit, since they were trained with LayerNorm.

**Parameter saving (our calculation for GPT-2 124M):** 12 blocks × 2 norms + 1 final norm = 25 norm layers. LayerNorm uses $25 \times 2 \times 768 = 38{,}400$ parameters; RMSNorm would use $19{,}200$. That's tiny next to 124M. The real win is **compute**: there's no mean to calculate and subtract.

## 🔴 Theory (Expert)
- **Invariances.** Ignoring $\epsilon$, $\mathrm{RMSNorm}(\alpha\mathbf{x}) = \mathrm{RMSNorm}(\mathbf{x})$ for any $\alpha > 0$. So it is **re-scaling invariant** (for both inputs and the weight matrix feeding it), like LayerNorm. Unlike LayerNorm, it is **not re-centering invariant**: $\mathrm{RMSNorm}(\mathbf{x}+c\mathbf{1}) \ne \mathrm{RMSNorm}(\mathbf{x})$. Zhang & Sennrich's hypothesis is that re-scaling invariance is what makes LayerNorm help, and re-centering is dispensable.
- **Geometry.** $\mathrm{RMS}(\mathbf{x}) = \lVert\mathbf{x}\rVert_2/\sqrt{d}$ (see [[norm]]). So, before $\boldsymbol\gamma$, RMSNorm projects $\mathbf{x}$ onto the sphere of radius $\sqrt{d}$. LayerNorm does the same *after* first projecting onto the hyperplane $\sum_i x_i = 0$ (the orthogonal complement of $\mathbf{1}$). So LayerNorm = centring (a [[linear-mapping]], a projection) followed by RMSNorm in $d-1$ dimensions, and RMSNorm = LayerNorm without the projection.
- **Relation to statistics.** With $\mu = \frac1d\sum x_i$ and the biased variance $\sigma^2 = \frac1d\sum (x_i-\mu)^2$ (the book's `unbiased=False`, see [[bessels-correction-biased-variance]]), $\frac1d\sum x_i^2 = \sigma^2 + \mu^2$. RMSNorm divides by the *second raw moment*, LayerNorm by the *second central moment*. They coincide exactly when $\mu = 0$. See [[standardization-z-score]].
- **Jacobian.** For $\mathbf{y} = \mathbf{x}/r$ with $r = \mathrm{RMS}(\mathbf{x})$ (no $\epsilon$): $\frac{\partial\mathbf{y}}{\partial\mathbf{x}} = \frac{1}{r}\left(I - \frac{\mathbf{x}\mathbf{x}^\top}{d\,r^2}\right)$. The matrix in brackets removes the component along $\mathbf{x}$, so gradients that only change the vector's length are cancelled. This keeps gradient sizes steady with depth and links to [[vanishing-and-exploding-gradients]] and [[backpropagation]].
- **Empirical claims (paper, not book).** The 2019 paper reports quality comparable to LayerNorm on translation and other tasks, with running-time reductions of roughly 7%–64% depending on the model. It also proposes *partial* RMSNorm (pRMSNorm), which estimates the RMS from only the first $p\%$ of entries.
- **In modern LLMs (general knowledge, not in the book).** RMSNorm in the [[pre-layernorm-vs-post-layernorm|pre-norm]] position, together with [[swiglu]] feed-forward layers and rotary [[relative-positional-embedding|relative position embeddings]], is the common recipe of open models such as Llama. The book names RMSNorm only in its further-reading notes.

## Connections
- Builds on: [[layer-normalization]], [[norm]], [[vector]], [[expected-value-and-mean]], [[covariance-and-variance]], [[standardization-z-score]], [[numerical-stability]]
- Leads to: [[transformer-block]], [[pre-layernorm-vs-post-layernorm]]
- Related: [[batch-normalization]], [[bessels-correction-biased-variance]], [[swiglu]], [[vanishing-and-exploding-gradients]], [[embedding-dimension]], [[gpt-2]]
- Used by: [[large-language-model]], [[decoder-only-transformer]]
- People: [[biao-zhang]], [[rico-sennrich]] (RMSNorm authors); [[jimmy-lei-ba]], [[geoffrey-hinton]] (LayerNorm, which it simplifies)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 4 references, book pp. 292–293 (RMSNorm is a popular LayerNorm variant in modern LLMs due to improved computing efficiency; normalizes using only the root mean square, without subtracting the mean; cites "Root Mean Square Layer Normalization" (2019) by Zhang and Sennrich, arXiv 1910.07467).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 100–103 (the `Linear(5, 6)+ReLU` example row, the LayerNorm outputs, Listing 4.2 `LayerNorm` with `eps = 1e-5`, `scale` and `shift`), used for the side-by-side example.
- The RMSNorm formula, the PyTorch class, the worked RMSNorm numbers, the parameter count, invariances, geometry, Jacobian and the paper's speed claims are standard results or our own calculations, not printed in the book.
