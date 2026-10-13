---
title: Multi-Head Attention (Heads and Head Dimension)
type: concept
tags: [attention, transformer, llm, deep-learning, linear-algebra]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Multi-Head Attention (Heads and Head Dimension)

> **One-liner:** Multi-head attention runs several small attention mechanisms side by side — each "head" looks at the sentence in its own way — and then glues their results together.

## 🟢 Intuition (Dummy)
Suppose you ask one friend to read a sentence and tell you which words relate to
which. They'll notice *some* patterns — maybe who-did-what — but miss others. Now ask
twelve friends, each told to focus on whatever they find useful: one tracks grammar,
one tracks pronouns, one tracks topic words… Then you staple their notes together.
That's **multi-head attention**: many [[causal-attention]] "heads", each with its
own learned way of looking, running in parallel.

The smart engineering twist: you don't actually hire twelve full-sized friends. You
take one big notebook and cut it into twelve narrow columns — each head works in a
slice of size **head dimension** = total width ÷ number of heads. For GPT-2 small,
768 ÷ 12 = 64.

## 🟡 Mechanics (Practitioner)
**Definition (book pp. 51, 74, 82).** Multi-head attention divides the attention
mechanism into multiple **heads**, each operating independently, so the model can
attend to information from **different representation subspaces at different
positions** at the same time. A single `CausalAttention` module is **single-head
attention** — one set of attention weights (book p. 82). The main idea: run attention
multiple times in parallel with **different, learned linear projections** — a linear
projection being the input multiplied by a weight matrix (book p. 83).

### Version 1 — stack heads (Listing 3.4, book p. 84)
```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```
In math, for head $i=1,\dots,h$:
$$Z_i=\operatorname{dropout}\!\left(\operatorname{softmax}\!\left(\frac{\operatorname{mask}(Q_iK_i^\top)}{\sqrt{d_k}}\right)\right)V_i,\qquad Z=[\,Z_1\,\|\,Z_2\,\|\cdots\|\,Z_h\,]$$
- $h$ = number of heads (`num_heads`).
- $Q_i=XW_q^{(i)}$, $K_i=XW_k^{(i)}$, $V_i=XW_v^{(i)}$ = head $i$'s queries, keys, values, each from its **own** weight matrices ([[query-key-value]]).
- $X$ = the input embeddings — the same for all heads.
- $\operatorname{mask}$ = the causal mask ($-\infty$ above the diagonal, see [[causal-attention]]).
- $\sqrt{d_k}$ = square root of the key size ([[scaled-dot-product-attention]]).
- $Z_i$ = head $i$'s [[context-vector|context vectors]]; $\|$ = concatenate side by side (`torch.cat(..., dim=-1)`).

Output width = $d_{out}\times h$. Book demo: `d_in=3, d_out=2, num_heads=2` on a batch
of shape `(2, 6, 3)` → `torch.Size([2, 6, 4])`; first row
`[-0.4519, 0.2216, 0.4772, 0.1063]` — first two numbers from head 1, last two from
head 2 (book pp. 84–85). In Fig. 3.24, the 5th token's head outputs `[-0.7, -0.1]`
and `[0.7, 0.4]` concatenate to `[-0.7, -0.1, 0.7, 0.4]` (book p. 83).
*Exercise 3.2:* to get 2-D outputs with 2 heads, set `d_out=1` (2 × 1 = 2) (book pp. 85, 301).

Downside: heads run **one after another** in the list comprehension (book p. 85).

### Version 2 — weight splits (Listing 3.5, book pp. 86–87) — the one used in the GPT
Use **one** big $W_q$ (size $d_{in}\times d_{out}$), do **one** matrix multiplication,
then *split* the result into heads (Fig. 3.26, book p. 88).
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out, self.num_heads = d_out, num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = self.dropout(torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1))
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)
```
(The book writes each `.view`/`.transpose` on its own line; condensed here.)

**Shape trace** ($b$ = batch, $T$ = tokens, $h$ = heads, $d_h$ = `head_dim`):

| Step | Shape |
|---|---|
| `x` | $(b, T, d_{in})$ |
| `W_key(x)` | $(b, T, d_{out})$ |
| `.view(b, T, h, d_h)` — split the last dim into heads | $(b, T, h, d_h)$ |
| `.transpose(1, 2)` — heads first | $(b, h, T, d_h)$ |
| `queries @ keys.transpose(2, 3)` — scores per head | $(b, h, T, T)$ |
| mask, softmax, dropout | $(b, h, T, T)$ |
| `@ values` | $(b, h, T, d_h)$ |
| `.transpose(1, 2)` then `.contiguous().view(b, T, d_out)` — merge heads | $(b, T, d_{out})$ |
| `out_proj` | $(b, T, d_{out})$ |

**The head dimension.**
$$d_h=\frac{d_{out}}{h}$$
- $d_h$ = `head_dim` = how many numbers each head works with.
- $d_{out}$ = total output width; $h$ = number of heads. $d_{out}$ must divide evenly by $h$ (the `assert`).

Note the scaling now uses $\sqrt{d_h}$ (`keys.shape[-1]` is the head size after the split), not $\sqrt{d_{out}}$ — visible in Listing 3.5's code (book p. 87); the book does not comment on it.

**The full formula** (our compact restatement of Listing 3.5):
$$\operatorname{MultiHead}(X)=[\,Z_1\,\|\cdots\|\,Z_h\,]\,W_o+\mathbf b_o$$
- $W_o,\mathbf b_o$ = the **output projection** `out_proj` ($d_{out}\times d_{out}$, with bias). The book says it's not strictly necessary but common in LLM architectures (book pp. 89–90; see [[output-projection]]).

**Batched matmul demo (book pp. 88–89).** With a tensor `a` of shape `(1, 2, 3, 4)`
(1 batch, 2 heads, 3 tokens, head_dim 4), `a @ a.transpose(2, 3)` multiplies the last
two dims for each head. Head 0 gives
`[[1.3208, 1.1631, 1.2879], [1.1631, 2.2150, 1.8424], [1.2879, 1.8424, 2.0402]]`, the
same as `a[0,0] @ a[0,0].T`. Check: $0.2745^2+0.6584^2+0.2775^2+0.8573^2\approx1.3208$
([[batched-matrix-multiplication]], [[gram-matrix]]).

**Demo output (book p. 90).** `MultiHeadAttention(3, 2, 6, 0.0, num_heads=2)` (so
`head_dim=1`) → `torch.Size([2, 6, 2])`, first row `[0.3190, 0.4858]`. Output width is
now controlled directly by `d_out`.

**GPT-2 numbers.** In GPT models $d_{in}=d_{out}$ (book p. 90). `GPT_CONFIG_124M` uses
`n_heads: 12`, `emb_dim: 768`, `context_length: 1024` (book p. 95). §3.6 also gives the
largest GPT-2 (1.5B) as 25 heads and embedding size 1,600 (book p. 90). All four GPT-2
sizes keep $d_h=64$ (`model_configs`, book p. 164, also §6.4 p. 182; Fig. 5.17 p. 163;
$d_h$ is our arithmetic):

| Model | `emb_dim` | `n_heads` | $d_h$ |
|---|---|---|---|
| gpt2-small (124M) | 768 | 12 | 64 |
| gpt2-medium (355M) | 1024 | 16 | 64 |
| gpt2-large (774M) | 1280 | 20 | 64 |
| gpt2-xl (1558M) | 1600 | 25 | 64 |

*Exercise 3.3* builds the GPT-2-small module: `MultiHeadAttention(768, 768, 1024, 0.0, 12)`
(book pp. 91, 301). Its parameters (derived): $3\cdot768^2=1{,}769{,}472$ for Q/K/V
(no bias) plus $768^2+768=590{,}592$ for `out_proj` = **2,360,064** per block — about
half of the feed-forward module's 4,722,432 (Exercise 4.1 posed on book p. 121;
printed totals in the solution, book p. 302). To load
OpenAI's GPT-2 weights the book sets `qkv_bias=True` (book p. 164).

In the GPT, each [[transformer-block]] is LayerNorm → masked multi-head attention →
[[dropout]] → shortcut (book p. 114).

> ⚠️ Book inconsistency: the Fig. 3.25 caption says the output columns "correspond to the embedding dimension specified via d_out=4", but the argument passed is `d_out=2` per head; 4 is the concatenated width (book p. 84).
>
> ⚠️ Book inconsistency: §3.6 calls the smallest GPT-2 "117 million parameters" (book p. 90), while §4.1 uses 124M and notes the original 117M figure "was later corrected" (book p. 93).

## 🔴 Theory (Expert)
- **Weight split = block-matrix product.** With
  $W_q=[\,W_q^{(1)}\,\|\cdots\|\,W_q^{(h)}\,]$ (column blocks),
  $XW_q=[\,XW_q^{(1)}\,\|\cdots\|\,XW_q^{(h)}\,]$. So one large matmul *is* $h$ small
  ones — Version 2 computes exactly the same function family as Version 1 with
  per-head size $d_h$, just batched (book p. 87: "the same concept"). The wrapper
  repeats the most expensive step — the matmul — once per head (book p. 90).
- **Subspaces.** Head $i$ computes scores $\mathbf q_i^\top\mathbf k_i$ in a
  $d_h$-dimensional projection. Each head's bilinear form
  $W_q^{(i)}W_k^{(i)\top}$ has rank $\le d_h$ ([[basis-and-rank]],
  [[low-rank-approximation]]); multiple heads give $h$ different low-rank
  similarity measures instead of one full-rank one. Concatenation is a direct sum
  $\mathbb R^{d_h}\oplus\cdots\oplus\mathbb R^{d_h}\cong\mathbb R^{d_{out}}$
  ([[vector-space]]), and $W_o$ mixes the subspaces back together.
- **Cost.** FLOPs for projections are the same as one head of width $d_{out}$;
  attention scores cost $O(h\,T^2 d_h)=O(T^2 d_{out})$. Memory for the $h$ score
  matrices is $O(hT^2)$ — the term [[flashattention]] attacks. Appendix B points to
  PyTorch's `scaled_dot_product_attention` and `MultiHeadAttention` as fused,
  FlashAttention-backed implementations (book p. 292).
- **Is `out_proj` (or even $W_v$) needed?** He & Hofmann (2023), cited in appendix
  B, show good performance without the value matrix and output projection
  (book p. 292).
- **`.contiguous()`.** `transpose` returns a strided view; `.view` requires
  contiguous memory, hence the copy before merging heads (background; not explained
  in the book).

## Connections
- Builds on: [[causal-attention]], [[self-attention]], [[scaled-dot-product-attention]], [[query-key-value]], [[linear-layer]], [[matrix-multiplication]]
- Uses: [[tensor-reshaping-and-transposing]], [[batched-matrix-multiplication]], [[weight-splitting-multi-head-attention]], [[output-projection]], [[dropout]]
- Leads to: [[transformer-block]], [[gpt-architecture]], [[gpt-2]], [[gpt-model-configuration]], [[parameter-count-and-memory]]
- Related math: [[gram-matrix]], [[basis-and-rank]], [[vector-space]], [[inner-product]]
- People: [[ashish-vaswani]], [[tri-dao]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1 intro, book p. 51 (Fig. 3.2); §3.5, book p. 74; §3.6.1–3.6.2, book pp. 82–91 (Figs. 3.24–3.26, Listings 3.4–3.5, Exercises 3.2–3.3).
- §4.1, book pp. 93–95 (`GPT_CONFIG_124M`); §4.5, book pp. 114–115; §4.6, book p. 121 (Exercise 4.1); §5.5, book pp. 163–164 (GPT-2 sizes, `qkv_bias`); §6.4, book p. 182.
- Appendix B Ch. 3 (PyTorch `scaled_dot_product_attention` / `MultiHeadAttention`; He & Hofmann 2023), book p. 292; appendix C solutions 3.2–3.3, book p. 301; solution 4.1, book p. 302.
