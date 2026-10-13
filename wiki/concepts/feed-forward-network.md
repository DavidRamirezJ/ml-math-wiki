---
title: Position-wise Feed-Forward Network
type: concept
tags: [deep-learning, transformer, llm, gpt, neural-networks]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Position-wise Feed-Forward Network

> **One-liner:** Inside every transformer block, a small two-layer network takes each token's vector on its own, blows it up to 4× the size, bends it with GELU, and squeezes it back down.

*Also called:* FFN, feed forward module, MLP block (OpenAI's GPT-2 weights call it `mlp`).

## 🟢 Intuition (Dummy)
A [[transformer-block]] has two jobs. First, the tokens **talk to each other** — that's
[[multi-head-attention]], where each word looks at the earlier words. Second, each token **thinks on its
own** about what it just heard — that's the feed-forward network.

The "thinking" step works like this: take the token's 768 numbers, spread them out into a much bigger
workspace of 3,072 numbers (more room to work), apply a smooth bend ([[gelu]]), then pack the result back
into 768 numbers so the next block gets the same size it expects. The *same* little network is used for
every token in the sentence — like one teacher giving every student the same personal exercise, one at a
time. That's why it's called **position-wise**.

## 🟡 Mechanics (Practitioner)
**Formula** (math form of Listing 4.4, applied to each token vector separately):
$$\mathrm{FFN}(\mathbf x)=W_2\,\mathrm{GELU}(W_1\mathbf x+\mathbf b_1)+\mathbf b_2$$
Symbol by symbol:
- $\mathbf x\in\mathbb R^{d}$ = one token's vector; $d$ = `emb_dim` = **768** for GPT-2 124M,
- $W_1\in\mathbb R^{4d\times d}$ = the "expand" weight [[matrix]] ($3072\times768$), $\mathbf b_1\in\mathbb R^{4d}$ = its [[bias-vector]],
- $W_1\mathbf x+\mathbf b_1$ = first [[linear-layer]]: 768 numbers → 3,072 numbers,
- $\mathrm{GELU}(\cdot)$ = the [[activation-function]], applied to each of the 3,072 entries,
- $W_2\in\mathbb R^{d\times4d}$ = the "contract" matrix ($768\times3072$), $\mathbf b_2\in\mathbb R^{d}$ = its bias,
- output = 768 numbers again — same size as the input.

**Listing 4.4 (book p. 107):**
```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

**Shape walk-through (Fig. 4.9, book p. 107)** — batch of 2 texts, 3 tokens each:

| Step | Shape | What changed |
|---|---|---|
| input | `(2, 3, 768)` | batch, tokens, embedding |
| Linear 1 | `(2, 3, 3072)` | embedding × 4 |
| GELU | `(2, 3, 3072)` | elementwise, no shape change |
| Linear 2 | `(2, 3, 768)` | embedding ÷ 4 |

```python
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)        # torch.Size([2, 3, 768])   (book p. 108)
```
`nn.Linear` acts only on the **last** dimension, so the batch and token dimensions pass straight through
(see [[broadcasting]], [[batched-matrix-multiplication]]). That's the code-level reason the FFN is
position-wise. The Fig. 4.9 caption says it well: the network accommodates any batch size and number of
tokens, but the embedding size per token "is determined and fixed when initializing the weights."

**Why expand then contract? (book p. 108, Fig. 4.10)** The book says the module "plays a crucial role in
enhancing the model's ability to learn from and generalize the data." It "internally expands the embedding
dimension into a higher-dimensional space" (768 → 3,072), applies the nonlinear GELU, then contracts back
to 768. The book says this "allows for the exploration of a richer representation space," and that keeping
input and output sizes equal lets us stack many blocks "without the need to adjust dimensions between
them," which makes the model more scalable.

**Attention vs. FFN — division of labor (book §4.5, p. 113).** Self-attention "identifies and analyzes
relationships between elements in the input sequence," while the feed forward network "modifies the data
individually at each position." In the Pre-LayerNorm GPT block (book pp. 114–115) the FFN sits in the
second half:
`x → LayerNorm 2 → FeedForward → Dropout → (+ shortcut) → out`.
```python
shortcut = x                     # shortcut connection for feed forward block
x = self.norm2(x)
x = self.ff(x)
x = self.drop_shortcut(x)
x = x + shortcut
```

**Parameter count — Exercise 4.1 (book p. 121; solution p. 302):**
$$\underbrace{768\cdot3072+3072}_{\text{Linear 1}=2{,}362{,}368}+\underbrace{3072\cdot768+768}_{\text{Linear 2}=2{,}360{,}064}=4{,}722{,}432$$
versus 2,360,064 for the multi-head attention module of the same block. So the FFN holds **about twice as
many parameters as attention**. Over the 12 blocks of GPT-2 124M that's 56,669,184 FFN parameters vs.
28,320,768 attention parameters (×12 computed here, not in the book). Appendix B (p. 293) also cites a blog
post (Harm de Vries, "In the Long (Context) Run") noting that for contexts under ~32,000 tokens, most of an
LLM's compute goes to the feed forward layers, not to attention.

**Loading GPT-2 weights (book §5.5, p. 166).** OpenAI's checkpoint calls the FFN `mlp`:
`ff.layers[0] ← mlp.c_fc` (768 → 3072) and `ff.layers[2] ← mlp.c_proj` (3072 → 768), with the weight
matrices transposed (`.T`) because OpenAI stores them the other way round. `ff.layers[1]` is the GELU and
has nothing to load.

## 🔴 Theory (Expert)
- **Position-wise = shared weights over tokens.** For an input matrix $X\in\mathbb R^{T\times d}$ (rows =
  tokens), the FFN is $\mathrm{GELU}(XW_1^\top+\mathbf 1\mathbf b_1^\top)W_2^\top+\mathbf 1\mathbf b_2^\top$:
  the same map applied to each row. It is permutation-equivariant over tokens and mixes **no** information
  across positions — all cross-token mixing comes from attention. Equivalently it is a 1×1 convolution over
  the sequence.
- **Why the nonlinearity is essential.** Without GELU, $W_2W_1$ is a $768\times768$ matrix of rank
  $\le768$, so the 3,072-wide middle would add nothing beyond a single linear map ([[basis-and-rank]],
  [[linear-mapping]]). GELU makes the expansion meaningful: the FFN becomes a one-hidden-layer
  [[multilayer-perceptron]], a universal approximator per token.
- **Key–value memory view (Geva et al. 2021 — outside the book).** Write $W_1$ rows as "keys" $\mathbf k_i$
  and $W_2$ columns as "values" $\mathbf v_i$: $\mathrm{FFN}(\mathbf x)=\sum_{i=1}^{4d}\mathrm{GELU}(\mathbf k_i^\top\mathbf x+b_i)\,\mathbf v_i+\mathbf b_2$.
  Each hidden unit detects a pattern and adds a value vector — a soft lookup table, which is one reason FFNs
  are thought to store much of an LLM's factual knowledge.
- **FLOPs.** Per token the FFN costs about $2\cdot(2\cdot d\cdot4d)=16d^2$ FLOPs; attention projections cost
  about $8d^2$, plus roughly $4Td$ for the scores and the weighted sum. So the FFN dominates until the
  context length $T$ grows past about $2d$ (and large models have $d$ in the thousands), which fits the ~32k-token
  remark the book cites.
- **Variants.** The original Transformer (Vaswani et al. 2017) used ReLU with $d=512$, inner size 2048 (also
  4×) and Post-LN ([[pre-layernorm-vs-post-layernorm]]). Modern LLMs often use a gated [[swiglu]] FFN with
  inner size ≈ $\tfrac83 d$ and no biases; mixture-of-experts models replace one FFN with many routed FFNs.

## Connections
- Builds on: [[linear-layer]], [[activation-function]], [[gelu]], [[matrix-multiplication]], [[bias-vector]], [[multilayer-perceptron]]
- Leads to: [[transformer-block]], [[gpt-architecture]], [[parameter-count-and-memory]]
- Used by: [[gpt-2]], [[decoder-only-transformer]], [[transformer-architecture]]
- Related: [[multi-head-attention]], [[layer-normalization]], [[shortcut-connections]], [[dropout]], [[swiglu]], [[pretrained-weights]], [[gpt-model-configuration]]
- People: [[ashish-vaswani]] (original Transformer FFN), [[alec-radford]] (GPT-2)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book pp. 105–108 (Listing 4.4, Figs. 4.9–4.10, shape check).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4, book p. 109 (Fig. 4.11 roadmap: FFN done).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book pp. 113–117 (role of FFN vs. attention; FFN in the transformer block, Fig. 4.13, Listing 4.6).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book p. 121 (Exercise 4.1).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book p. 166 (mapping `mlp.c_fc` / `mlp.c_proj` weights).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book p. 293 (de Vries blog on FFN compute share).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book p. 302 (Exercise 4.1 solution: 4,722,432 vs. 2,360,064).
