---
title: Weight Splits and Fused QKV Projection
type: concept
tags: [attention, transformer, linear-algebra, block-matrices, pytorch, llm]
difficulty: advanced
sources: [llm-ch03-coding-attention-mechanisms, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Weight Splits and Fused QKV Projection

> **One-liner:** Instead of giving every attention head its own small weight matrix, you glue them side by side into one big matrix, do one multiplication, and then cut the result back into per-head pieces.

## 🟢 Intuition (Dummy)
Imagine a bakery with 12 bakers, each making their own small batch of dough in their own small bowl.
That works, but it means 12 separate mixing jobs. A smarter bakery uses **one giant mixer**, makes all
the dough in a single run, and then **slices** the big lump into 12 portions. Every baker still gets
exactly their own portion with their own recipe, but the kitchen did one big job instead of twelve
small ones.

[[multi-head-attention|Multi-head attention]] does the same. Each head needs its own query, key and
value projections. Rather than doing 12 small matrix multiplications, the model stacks all the
heads' weights into one wide matrix, multiplies once, and slices the output into heads. That is a
**weight split**.

GPT-2 goes one step further: it glues the query, key and value matrices themselves together into a
single matrix (a **fused QKV** matrix). When the book loads OpenAI's weights, it has to slice that one
matrix back into three.

## 🟡 Mechanics (Practitioner)

### Two ways to build multi-head attention (§3.6, pp. 82–90)
**Way 1 — the wrapper (Listing 3.4, p. 84).** Keep a list of independent `CausalAttention` heads and concatenate their outputs:
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
Each head has its own $W_q^{(i)}, W_k^{(i)}, W_v^{(i)}$, and the heads run **one after another** in the list
comprehension (p. 85). Output width: `d_out * num_heads` (2 × 2 = 4 in the book's demo, p. 84).

**Way 2 — weight splits (Listing 3.5, pp. 86–87).** One big $W_q$ instead of $W_{q1}, W_{q2}$; **one** matrix
multiplication gives $Q$, which is then split into $Q_1, Q_2$; same for keys and values (Fig. 3.26, p. 88).
```python
assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
self.head_dim = d_out // num_heads
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)   # one big matrix for all heads
...
keys = self.W_key(x)                                          # (b, T, d_out)
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # (b, T, h, d_h)  split
keys = keys.transpose(1, 2)                                    # (b, h, T, d_h)  heads first
attn_scores = queries @ keys.transpose(2, 3)                   # (b, h, T, T)    all heads at once
...
context_vec = (attn_weights @ values).transpose(1, 2)          # (b, T, h, d_h)
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # merge heads
context_vec = self.out_proj(context_vec)                       # optional output projection
```
Here the total output width is `d_out` itself, **not** `d_out * num_heads` (pp. 86–87, 90).

### The key formula: one big multiply = many small ones
$$
XW_q \;=\; X\big[\,W_q^{(1)} \,\big|\, W_q^{(2)} \,\big|\,\cdots\,\big|\, W_q^{(h)}\,\big]
\;=\; \big[\,XW_q^{(1)} \,\big|\, XW_q^{(2)} \,\big|\,\cdots\,\big|\, XW_q^{(h)}\,\big]
\;=\; \big[\,Q_1 \,|\, Q_2 \,|\,\cdots\,|\, Q_h\,\big].
$$
Symbol by symbol:
- $X \in \mathbb R^{T\times d_{in}}$ = the input tokens (one per row).
- $W_q \in \mathbb R^{d_{in}\times d_{out}}$ = the one big query matrix.
- $[\,A \mid B\,]$ = "put matrix $A$ and matrix $B$ side by side" (column-wise concatenation).
- $W_q^{(i)} \in \mathbb R^{d_{in}\times d_h}$ = the block of columns that belongs to head $i$; $d_h = d_{out}/h$ = `head_dim`; $h$ = `num_heads`.
- $Q_i = XW_q^{(i)} \in \mathbb R^{T\times d_h}$ = head $i$'s queries.

In words: multiplying $X$ by a matrix whose columns are split into blocks gives an output whose columns
are split into the same blocks. `.view(b, T, h, d_h)` is just the code for "cut the last axis into $h$
consecutive chunks of length $d_h$" — chunk $i$ is columns $i\,d_h$ to $(i+1)\,d_h-1$. (The block-matrix
algebra is our addition; the book shows it as Fig. 3.26.)

Then each head runs [[scaled-dot-product-attention]] on its slice, scaling by $\sqrt{d_h}$ (`keys.shape[-1]`, p. 87):
$$
Z_i = \operatorname{dropout}\!\left(\operatorname{softmax}\!\left(\frac{\operatorname{mask}(Q_iK_i^\top)}{\sqrt{d_h}}\right)\right)V_i,\qquad
\operatorname{MultiHead}(X) = [\,Z_1\,|\,\cdots\,|\,Z_h\,]\,W_o + \mathbf b_o .
$$
- $\operatorname{mask}$ = the [[causal-attention-mask]] (future positions set to $-\infty$).
- $W_o \in \mathbb R^{d_{out}\times d_{out}}$, $\mathbf b_o$ = the weights and bias of `out_proj`, the [[output-projection]]. The book calls it "not strictly necessary" but "commonly used in many LLM architectures" (pp. 89–90).

### Why bother? (p. 90)
In the wrapper, the projection matmul — "computationally one of the most expensive steps" — is repeated
once per head. With weight splits it happens **once** for keys, once for queries, once for values. Then
[[batched-matrix-multiplication]] (`@` on 4-D tensors multiplies the last two axes and repeats over the
batch and head axes, p. 89) computes all heads' scores together. The `.view`/`.transpose` code "looks
mathematically complicated", but it implements the same concept as the wrapper (p. 87). See
[[tensor-reshaping-and-transposing]].

### Book demo numbers
- `MultiHeadAttention(d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2)` → `head_dim = 1`;
  output shape `torch.Size([2, 6, 2])`, first rows `[0.3190, 0.4858]`, `[0.2943, 0.3897]` (seed 123, p. 90).
- GPT-2 small (p. 90): 12 heads, embedding size 768 → $d_h = 768/12 = 64$ (our arithmetic).
- **Exercise 3.3** (p. 91): build a GPT-2-size module (`d_in = d_out = 768`, 12 heads, `context_length = 1024`).
  Parameter count (derived, not in the book): $3\cdot768^2 = 1{,}769{,}472$ for Q/K/V (no bias) plus
  $768^2 + 768 = 590{,}592$ for `out_proj`, total **2,360,064**.

### Fused QKV: loading GPT-2's weights (§5.5, pp. 164–166)
OpenAI stores Q, K and V **concatenated** in one array, `params["blocks"][b]["attn"]["c_attn"]["w"]`.
The book splits it into three equal parts along the last axis (Listing 5.5, p. 165):
```python
q_w, k_w, v_w = np.split(
    (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
gpt.trf_blocks[b].att.W_query.weight = assign(
    gpt.trf_blocks[b].att.W_query.weight, q_w.T)
# ... same for W_key (k_w.T) and W_value (v_w.T)

q_b, k_b, v_b = np.split(
    (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
gpt.trf_blocks[b].att.W_query.bias = assign(
    gpt.trf_blocks[b].att.W_query.bias, q_b)
# ... same for W_key (k_b) and W_value (v_b)
```
The math:
$$
W_{qkv} = [\,W_q \mid W_k \mid W_v\,] \in \mathbb R^{768\times 2304},\qquad
\mathbf b_{qkv} = [\,\mathbf b_q \mid \mathbf b_k \mid \mathbf b_v\,] \in \mathbb R^{2304}.
$$
- $W_{qkv}$ = the fused matrix: the three $768\times768$ matrices side by side ($2304 = 3\cdot768$, derived from the shapes).
- `np.split(..., 3, axis=-1)` = cut the last axis into 3 equal blocks.
- `.T` on the weights only: PyTorch's `nn.Linear` stores the weight as (out, in) and computes $\mathbf xW^\top + \mathbf b$, while OpenAI's layout is (in, out). The 1-D biases need no transpose. (The book shows the `.T` without explaining it; the reason is the standard PyTorch convention.)
- `qkv_bias=True` must be set, because OpenAI's model *does* use biases in these layers, even though bias vectors are "not commonly used in LLMs anymore" (p. 164). With bias the projection is affine: $\mathbf q = \mathbf xW_q + \mathbf b_q$ (see [[bias-vector]]).
- `assign` raises an error if shapes don't match — a safety net, because "a lot of guesswork" went into matching OpenAI's names (pp. 165–166).

The output projection is loaded from a separate array, `c_proj`: `out_proj.weight` gets `c_proj["w"].T` and `out_proj.bias` gets `c_proj["b"]` (pp. 165–166).

So the stack of splits is: **fused QKV → Q, K, V** (split into 3, done at load time) and then **Q → heads** (split into 12 inside `forward`).

## 🔴 Theory (Expert)
- **Block-matrix multiplication.** If $W = [W^{(1)}\mid\cdots\mid W^{(m)}]$ is partitioned by columns, then $XW = [XW^{(1)}\mid\cdots\mid XW^{(m)}]$ — each column of $XW$ only depends on the matching column of $W$. This single identity justifies both the per-head split and the fused QKV matrix. It is the column version of [[matrix-multiplication]] seen as a [[linear-mapping]] applied to each basis direction.
- **Same parameter count, same expressiveness.** A wrapper with $h$ heads of width $d_h$ has $3\cdot h\cdot d_{in}d_h = 3\,d_{in}d_{out}$ projection parameters — exactly the same as three $d_{in}\times d_{out}$ matrices. The two parameterizations represent the same family of functions; they differ only in memory layout and number of kernel launches. (The book's own wrapper demo uses `d_out` *per head*, so its output is $h$ times wider; that is a choice of sizes, not a different idea. Note also the Fig. 3.25 caption calls the concatenated width "d_out=4" though the argument passed is `d_out=2` — a book inconsistency.)
- **Heads see disjoint subspaces of the projection.** Head $i$ only uses the columns $W_q^{(i)}, W_k^{(i)}, W_v^{(i)}$, so its scores $Q_iK_i^\top = X\,W_q^{(i)}W_k^{(i)\top}X^\top$ come from a rank-$\le d_h$ bilinear form. Splitting $d_{out}$ into $h$ heads trades one rank-$d_{out}$ score pattern for $h$ different rank-$d_h$ patterns. The book's phrase "different representation subspaces" (p. 74) is this decomposition (see [[basis-and-rank]], [[vector-space]] — direct sum $\mathbb R^{d_{out}} = \bigoplus_i\mathbb R^{d_h}$).
- **The output projection mixes heads.** Concatenation alone keeps heads in separate coordinates. $W_o$ is the only place where information from different heads is linearly combined before the [[feed-forward-network]]. He & Hofmann (2023) report good performance without the value matrix and the output projection (Appendix B, p. 292).
- **Why `.contiguous()`** (our explanation; the book does not say): `.transpose` only changes strides, leaving the data non-contiguous in memory, and `.view` requires a contiguous layout, so the code makes a contiguous copy first.
- **Fusing in other code** (outside the book): GPT-2's original code computes all of Q, K, V with one matmul against `c_attn` and splits the output, and PyTorch's `nn.MultiheadAttention` stores a fused `in_proj_weight`. The book instead keeps three separate `nn.Linear` layers and splits the *stored weights* when loading.

## Connections
- Builds on: [[matrix-multiplication]], [[linear-mapping]], [[linear-layer]], [[scaled-dot-product-attention]], [[query-key-value]], [[tensor]]
- Leads to: [[multi-head-attention]], [[transformer-block]], [[gpt-architecture]], [[pretrained-weights]]
- Related: [[batched-matrix-multiplication]], [[tensor-reshaping-and-transposing]], [[output-projection]], [[bias-vector]], [[parameter-count-and-memory]], [[gpt-2]], [[gpt-model-configuration]], [[weight-tying]]
- Used by: [[gpt-2]], [[large-language-model]]
- People: [[sebastian-raschka]], [[ashish-vaswani]], [[alec-radford]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6.1 (Listing 3.4, Figs. 3.24–3.25), book pp. 82–85; §3.6.2 (Listing 3.5, Fig. 3.26, batched matmul, `out_proj`, GPT-2 sizes, Exercise 3.3), book pp. 86–91.
- Same PDF §5.5 (`qkv_bias=True`, `assign`, Listing 5.5 `load_weights_into_gpt` with `np.split` of `c_attn`), book pp. 164–166.
- Same PDF §3.4 (preview of "representation subspaces"), book p. 74; Appendix B ch. 3 refs (He & Hofmann 2023), book p. 292.
