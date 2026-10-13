---
title: Transformer Block
type: concept
tags: [llm, transformer, deep-learning, model-architecture, attention]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Transformer Block

> **One-liner:** A transformer block is the repeating "Lego brick" of a GPT model: it lets every token look at the earlier tokens (attention), then thinks about each token on its own (feed-forward), and hands back vectors of exactly the same shape so the next brick can stack on top.

*Also covers:* shape preservation in transformers.

## 🟢 Intuition (Dummy)
Imagine a reading group working through a sentence. Each round has two steps:
1. **Discuss** — every word "asks around" the earlier words to gather context
   ("'it' probably refers to 'the cat'"). That's [[multi-head-attention]].
2. **Reflect** — each word then quietly updates its own notes, alone. That's the
   [[feed-forward-network]].

Before each step, everyone's notes are tidied to a standard scale
([[layer-normalization]]). After each step, the new ideas are *added on top* of
the old notes instead of replacing them ([[shortcut-connections]]) — so nothing
important gets lost, and the learning signal flows back easily.

One round = one transformer block. GPT-2 small runs **12 rounds**; GPT-2 XL runs
**48** (§4.6, p. 117). Because each round hands back notes of the same size
(768 numbers per word), you can stack as many rounds as you like.

## 🟡 Mechanics (Practitioner)

### The layout (Figure 4.13, p. 114)
Bottom to top: **LayerNorm 1 → masked multi-head attention → dropout → (+ shortcut)
→ LayerNorm 2 → feed forward (Linear → GELU → Linear) → dropout → (+ shortcut)**.

### The formulas
For an input $X \in \mathbb{R}^{B\times T\times d}$:

$$
H = X + \operatorname{Dropout}\big(\operatorname{MHA}(\operatorname{LN}_1(X))\big)
$$
$$
Y = H + \operatorname{Dropout}\big(\operatorname{FFN}(\operatorname{LN}_2(H))\big)
$$

Symbol by symbol:
- $X$ = the block's input: $B$ texts, $T$ tokens each, every token a vector of $d = 768$ numbers.
- $\operatorname{LN}_1, \operatorname{LN}_2$ = two separate [[layer-normalization]] layers (each with its own learned scale and shift).
- $\operatorname{MHA}$ = masked [[multi-head-attention]] — mixes information **across positions** (each token only sees itself and earlier tokens, via the [[causal-attention-mask]]).
- $\operatorname{FFN}(\mathbf{z}) = W_2\,\operatorname{GELU}(W_1\mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2$ = the [[feed-forward-network]], applied to **each position separately**; it expands $768 \to 3072 \to 768$.
- $\operatorname{Dropout}$ = [[dropout]] at rate `drop_rate` (0.1), active only in training. The block reuses one `drop_shortcut` module for both sub-blocks; `MultiHeadAttention` also applies its own dropout (same `drop_rate`) to the attention weights.
- $X + \dots$ and $H + \dots$ = the two **shortcut (residual) connections**.
- $H$ = the halfway result; $Y$ = the block output, **same shape as $X$**.

### The code (Listing 4.6, p. 115)
```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x                 # shortcut for the attention sub-block
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut             # add the original input back

        shortcut = x                 # shortcut for the feed-forward sub-block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
```
Note `d_in = d_out = emb_dim`: attention returns vectors as wide as it receives.

### Shape preservation — the book's test (p. 116)
```python
torch.manual_seed(123)
x = torch.rand(2, 4, 768)            # [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
# Input shape:  torch.Size([2, 4, 768])
# Output shape: torch.Size([2, 4, 768])
```
The book stresses that keeping the shape is "not incidental but a crucial aspect
of its design" (p. 116): each output vector corresponds one-to-one to an input
vector, but its *content* has been re-encoded as a [[context-vector]] that mixes
in information from the whole (visible) sequence. In math, the block is a map

$$
\text{TransformerBlock}: \mathbb{R}^{B\times T\times d} \to \mathbb{R}^{B\times T\times d}
$$

- $\to$ = "maps to"; same space in, same space out — so blocks can be **composed**:
  $\text{Block}_{12}\circ\cdots\circ\text{Block}_1$.

### Stacking into GPT (§4.6, p. 119)
```python
self.trf_blocks = nn.Sequential(
    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
```
The 12 blocks have identical structure but **separate weights**. When the book
loads OpenAI's GPT-2, it loops `for b in range(len(params["blocks"]))` and fills
each block's attention, feed-forward, and two LayerNorms (§5.5, pp. 165–166).

### Parameter budget per block (GPT-2 small, `qkv_bias=False`)
| Part | Parameters |
|---|---|
| Multi-head attention ($3\cdot768^2 + 768^2 + 768$) | 2,360,064 |
| Feed-forward ($768\cdot3072 + 3072 + 3072\cdot768 + 768$) | 4,722,432 |
| Two LayerNorms ($2\times 2\times768$) | 3,072 |
| **Total per block** | **7,085,568** |

So the FFN holds about **2×** the attention parameters — the answer to Exercise 4.1
(p. 121; appendix C, p. 302). 12 blocks = 85,026,816 parameters.

### Using a block in fine-tuning
In Ch. 6 the whole model is frozen, then the **last** transformer block and the
final LayerNorm are made trainable again with `requires_grad = True` (§6.5, p. 186).
The book's reasoning (pp. 185–186): lower blocks capture general language structure
that transfers across tasks, while the last blocks hold more task-specific
features — and the author found that fine-tuning more than just the new output layer
"can noticeably improve the predictive performance". See [[layer-freezing]].

## 🔴 Theory (Expert)
- **Pre-LN vs Post-LN.** GPT-2 uses **Pre-LayerNorm**: $y = x + F(\operatorname{LN}(x))$.
  The original transformer used **Post-LayerNorm**: $y = \operatorname{LN}(x + F(x))$,
  which "often leads to worse training dynamics" (p. 115). In Pre-LN the residual
  stream $x$ is never normalized inside the block, so there is a clean identity
  path from the embeddings to the final LayerNorm (Xiong et al. 2020, appendix B,
  p. 292). See [[pre-layernorm-vs-post-layernorm]].
- **Gradient flow.** The block's Jacobian is
  $\frac{\partial Y}{\partial X} = \big(I + J_{\text{FFN}}\big)\big(I + J_{\text{MHA}}\big)$,
  where each $J$ includes the LayerNorm in front of the sublayer. Expanding the
  product over 12 blocks always leaves the bare identity term, so gradients can
  reach the embeddings without vanishing ([[vanishing-and-exploding-gradients]]).
- **Residual stream view.** Unrolling the stack:
  $X^{(L)} = X^{(0)} + \sum_{\ell=1}^{L}\big(\Delta^{\text{att}}_\ell + \Delta^{\text{ffn}}_\ell\big)$.
  Every sublayer *reads* from a shared $d$-dimensional stream and *writes* an
  additive update back into it. Shape preservation is exactly what makes this sum
  well-defined (all terms live in $\mathbb{R}^{T\times d}$).
- **Division of labour.** Attention is the only operation that mixes positions
  (a data-dependent [[convex-combination]] of value vectors). The FFN, LayerNorm,
  dropout, and residual add are all *position-wise* — the same function applied to
  each token independently (p. 113). Without attention, a stack of blocks could not
  relate tokens at all.
- **Why expand to $4d$.** $W_2W_1$ alone would have rank $\le d$ — no more
  expressive than one $d\times d$ matrix. The [[gelu|GELU]] between them is what
  turns the $4d$ hidden layer into extra capacity (§4.3; link [[basis-and-rank]]).
- **Permutation behaviour.** Apart from the causal mask, attention is
  permutation-equivariant over positions, and the FFN is too. Word order enters
  only through the [[positional-embedding|positional embeddings]] added before
  the first block.

## Connections
- Builds on: [[multi-head-attention]], [[causal-attention]], [[layer-normalization]], [[feed-forward-network]], [[gelu]], [[shortcut-connections]], [[dropout]]
- Leads to: [[gpt-architecture]], [[gpt-2]], [[decoder-only-transformer]], [[large-language-model]]
- Configured by: [[gpt-model-configuration]] (`emb_dim`, `n_heads`, `n_layers`, `drop_rate`, `qkv_bias`)
- Related: [[pre-layernorm-vs-post-layernorm]], [[context-vector]], [[vanishing-and-exploding-gradients]], [[layer-freezing]], [[tensor]], [[transformer-architecture]]
- People: [[ashish-vaswani]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 93–97 (role in `DummyGPTModel`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book pp. 105–109 (`FeedForward`, $4\times$ expansion)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4, book pp. 109–113 (shortcut connections)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book pp. 113–117 (Figure 4.13, Listing 4.6, Pre-LN vs Post-LN, shape preservation)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 117–121 (stacking 12 / 48 blocks, Exercise 4.1)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7 / Chapter 4 summary, book p. 127
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 163–166 (per-block weight loading)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book p. 186 (unfreezing the last block)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book p. 292 (Xiong et al. 2020); appendix C, book p. 302 (Exercise 4.1 solution)
