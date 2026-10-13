---
title: Pre-LayerNorm vs Post-LayerNorm
type: concept
tags: [transformer, normalization, architecture, training-stability, llm]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Pre-LayerNorm vs Post-LayerNorm

> **One-liner:** It's the question of *where* to put the layer-normalization step in a transformer block — before each sub-layer (Pre-LN, used by GPT-2 and modern LLMs) or after the residual add (Post-LN, the original 2017 transformer).

## 🟢 Intuition (Dummy)
Think of a relay team carrying a bucket of water (the information) down a long track.
At each station, a worker can add a little water (the sub-layer's work).

- **Post-LN**: after each worker adds water, someone **re-measures and resizes the whole
  bucket** to a standard size. The bucket itself keeps getting reshaped at every station.
  With many stations, that constant reshaping makes things wobbly, especially at the start
  of training.
- **Pre-LN**: the bucket is **never** touched on the main track. Each worker takes a
  *tidied-up copy* of what's in the bucket, does their work on the copy, and pours their
  small result back in. The main track stays a clean, straight line from start to finish.

The clean straight line is the [[shortcut-connections|shortcut]] path. The book says
Post-LN "often leads to worse training dynamics" (book p. 115). Our explanation (not the
book's): Pre-LN keeps normalization **off** the main path, so the signal and the
gradient travel along it untouched.

## 🟡 Mechanics (Practitioner)

### The two formulas
Let $\operatorname{Sublayer}$ be either the attention part or the feed-forward part of a
[[transformer-block]], and $\operatorname{LN}$ be [[layer-normalization]].

**Post-LayerNorm** (original transformer, Vaswani et al. 2017):
$$\mathbf y = \operatorname{LN}\big(\mathbf x + \operatorname{Sublayer}(\mathbf x)\big)$$

**Pre-LayerNorm** (GPT-2, the book's `TransformerBlock`):
$$\mathbf y = \mathbf x + \operatorname{Sublayer}\big(\operatorname{LN}(\mathbf x)\big)$$

Symbol by symbol:
- $\mathbf x$ = the input to the sub-layer (for GPT-2 small, one 768-number [[vector]] per [[token]]).
- $\operatorname{Sublayer}(\cdot)$ = masked [[multi-head-attention]] or the [[feed-forward-network]] (plus its [[dropout]]).
- $\operatorname{LN}(\mathbf z)=\gamma\odot\frac{\mathbf z-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ = layer norm: subtract the mean $\mu$ of the vector's entries, divide by their standard deviation $\sqrt{\sigma^2+\epsilon}$, then rescale by learnable $\gamma$ (`scale`) and shift by $\beta$ (`shift`) ([[standardization-z-score]]).
- $\mathbf x + \dots$ = the residual / shortcut addition.
- $\mathbf y$ = the output, same shape as $\mathbf x$.

The only difference is **where the parentheses go**. In Post-LN, the LN wraps the sum (so
it sits *on* the main path). In Pre-LN, the LN is *inside* the branch (so the main path is
a pure addition).

### In the book's code (Listing 4.6, book p. 115)
```python
def forward(self, x):
    shortcut = x
    x = self.norm1(x)        # LN BEFORE attention  -> Pre-LN
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut         # plain add, no LN after it
    shortcut = x
    x = self.norm2(x)        # LN BEFORE feed-forward
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut
    return x
```
The book's description: "Layer normalization (`LayerNorm`) is applied before each of
these two components, and dropout is applied after them to regularize the model and
prevent overfitting. This is also known as *Pre-LayerNorm*. Older architectures, such as
the original transformer model, applied layer normalization after the self-attention and
feed forward networks instead, known as *Post-LayerNorm*, which often leads to worse
training dynamics" (book p. 115).

A Post-LN version of the same block would instead be:
```python
x = self.norm1(x + self.drop_shortcut(self.att(x)))
x = self.norm2(x + self.drop_shortcut(self.ff(x)))
```
(our illustration, not in the book).

### The extra final LayerNorm
`GPTModel` applies one more `self.final_norm = LayerNorm(cfg["emb_dim"])` after the last
block and before the output head, "standardizing the outputs from the transformer blocks
to stabilize the learning process" (book p. 119; Fig. 4.15, book p. 118). Our explanation
(not the book's): because Pre-LN never normalizes the main path, the output of the last
block is an un-normalized running sum. Post-LN models don't need this, because each block
already ends with an LN.

### Further reading the book points to (appendix B, book p. 292)
The book's summary: Post-LayerNorm, used in the original transformer, applies layer
normalization after the self-attention and feed forward networks; Pre-LayerNorm, "as adopted
in models like GPT-2 and newer LLMs", applies it before them, "which can lead to more stable
training dynamics and has been shown to improve performance in some cases", as discussed in:
- Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture" (arXiv 2002.04745).
- Tie et al. (2023), "ResiDual: Transformer with Dual Residual Connections" (arXiv 2304.14802).

The same page also cites Ba, Kiros & Hinton (2016), "Layer Normalization", and notes that
[[rmsnorm]] is a popular, cheaper LayerNorm variant in modern LLMs (book pp. 292–293).

## 🔴 Theory (Expert)
(The results below summarise the cited papers; they go beyond what the book itself states.)
- **Gradient at initialization (Xiong et al., 2020).** With Post-LN, the expected gradient
  norm of the parameters near the *output* is large at initialization and does not shrink
  with depth, so a large learning rate blows up early training. This is why the original
  transformer needs a [[learning-rate-warmup]]. With Pre-LN, the gradient norm scales
  like $O(1/\sqrt L)$ per layer and is well behaved, so warmup can often be shortened or
  removed. (The book still uses warmup + [[cosine-decay]] in appendix D.)
- **Identity path.** In Pre-LN, $\mathbf x_L=\mathbf x_0+\sum_{\ell}F_\ell(\operatorname{LN}(\mathbf x_{\ell}))$,
  so $\partial\mathbf x_L/\partial\mathbf x_\ell$ contains an exact identity term
  (see [[shortcut-connections]]). In Post-LN, every block's Jacobian is multiplied by the
  LN Jacobian $\partial\operatorname{LN}/\partial\mathbf z$, which rescales by $1/\sigma$
  and projects out the mean direction — the clean identity path is lost.
- **The trade-off.** Pre-LN's residual stream grows in norm with depth, so later sub-layers
  contribute relatively less ("representation collapse" / under-used deep layers); some
  studies find well-tuned Post-LN reaches slightly better final quality. ResiDual
  (Tie et al., 2023) keeps both a Pre-LN and a Post-LN residual stream to get the
  stability of one and the expressiveness of the other.
- **Modern practice.** Most current LLMs use Pre-LN with [[rmsnorm]] (no mean
  subtraction, cheaper; Zhang & Sennrich 2019, appendix B p. 292–293) instead of
  LayerNorm.

## Connections
- Builds on: [[layer-normalization]], [[shortcut-connections]], [[transformer-block]], [[standardization-z-score]]
- Leads to: [[gpt-architecture]], [[rmsnorm]], [[learning-rate-warmup]]
- Related: [[vanishing-and-exploding-gradients]], [[batch-normalization]], [[dropout]], [[transformer-architecture]]
- Used by: [[gpt-2]], [[decoder-only-transformer]]
- People: [[ashish-vaswani]] (original Post-LN transformer), [[jimmy-lei-ba]] & [[geoffrey-hinton]] (LayerNorm), [[biao-zhang]] & [[rico-sennrich]] (RMSNorm)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book pp. 113–116 (Fig. 4.13, Listing 4.6, Pre-LN vs Post-LN paragraph on p. 115).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 117–119 (final LayerNorm in `GPTModel`, Fig. 4.15, Listing 4.7).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 4 references, book pp. 292–293 (Xiong et al. 2020; Tie et al. 2023; Zhang & Sennrich 2019).
- Theory-tier gradient-scaling and trade-off claims: from the cited papers, not stated in the book.
