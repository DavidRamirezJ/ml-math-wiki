---
title: FlashAttention
type: concept
tags: [attention, efficiency, gpu, numerical-stability, transformer]
difficulty: advanced
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# FlashAttention

> **One-liner:** FlashAttention computes exactly the same attention as the normal formula, just much faster and with far less memory, by working in small tiles that fit in the GPU's fastest memory.

## 🟢 Intuition (Dummy)
Think of a chef with a tiny countertop and a big pantry down the hall. The slow way
to cook is: bring *every* ingredient from the pantry, chop it, carry the whole pile
back to the pantry, then fetch the pile again for the next step. Most of the time
goes into walking, not cooking.

FlashAttention is the smart chef. They bring a small basket of ingredients, do
**all** the steps on that basket right there on the counter, keep a little running
note of the results, and only walk back once. Same dish, same taste — **exactly**
the same — but a lot less walking.

On a GPU, the "pantry" is the big but slow main memory (HBM) and the "countertop" is
the small but super-fast on-chip memory (SRAM). Ordinary attention builds a huge
table with one score for every pair of tokens and writes it all to the pantry.
FlashAttention never builds the whole table at once.

## 🟡 Mechanics (Practitioner)

### What the book says
The book mentions FlashAttention only in its references (appendix B, chapter 3):
> "FlashAttention is a highly efficient implementation of a self-attention mechanism,
> which accelerates the computation process by optimizing memory access patterns.
> FlashAttention is mathematically the same as the standard self-attention mechanism
> but optimizes the computational process for efficiency" (book p. 291).

It cites two papers: Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness" (arXiv 2205.14135), and Dao (2023),
"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
(arXiv 2307.08691). It also notes that "PyTorch implements a function for
self-attention and causal attention that supports FlashAttention for efficiency"
(`scaled_dot_product_attention`, "beta and subject to change"), and an efficient
`MultiHeadAttention` class built on it (book p. 292). Everything below that isn't
from those lines is background from the FlashAttention papers.

### The thing it computes (unchanged)
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
Symbol by symbol:
- $Q$ = the query matrix, one row per token ($N$ rows, $d_k$ columns). → [[query-key-value]]
- $K$ = the key matrix; $K^\top$ = $K$ [[matrix-transpose|transposed]].
- $QK^\top$ = an $N\times N$ table of [[attention-scores]]: entry $(i,j)$ is the [[inner-product|dot product]] of query $i$ with key $j$.
- $\sqrt{d_k}$ = square root of the key size; dividing by it keeps scores from getting too large. → [[scaled-dot-product-attention]]
- $\operatorname{softmax}$ = applied to each row, turns scores into [[attention-weights]] that are positive and add up to 1. → [[softmax]]
- $V$ = the value matrix; multiplying by it gives the [[context-vector|context vectors]].

The book's own `MultiHeadAttention` (§3.6) does exactly this, and it materializes the
full `attn_scores` tensor of shape $(b, h, N, N)$. For GPT-2's $N = 1{,}024$ tokens
(book p. 91) and 12 heads, that's $12 \times 1024^2 \approx 12.6$ million scores **per
text, per layer** — all written to and read back from slow memory. (Arithmetic ours.)

### The trick: tiles + an "online" softmax
The problem: softmax needs the **whole row** (its max and its sum) before you can
normalize. FlashAttention gets around that by keeping three running numbers per
query row while it walks through the keys block by block (block $j$ = a few keys):

$$m^{\text{new}} = \max\!\big(m,\ \max_k s_k^{(j)}\big)$$
$$\ell^{\text{new}} = e^{\,m - m^{\text{new}}}\,\ell \;+\; \sum_k e^{\,s_k^{(j)} - m^{\text{new}}}$$
$$\mathbf{o}^{\text{new}} = e^{\,m - m^{\text{new}}}\,\mathbf{o} \;+\; \sum_k e^{\,s_k^{(j)} - m^{\text{new}}}\,\mathbf{v}_k$$
and at the very end, $\text{output} = \mathbf{o}/\ell$.

Symbol by symbol:
- $s_k^{(j)}$ = the score of this query against key $k$ in block $j$.
- $m$ = the biggest score seen so far (the running max). Subtracting it keeps $e^{(\cdot)}$ from overflowing. → [[numerical-stability]]
- $\ell$ = the running softmax denominator (sum of exponentials, measured relative to $m$).
- $\mathbf{o}$ = the running weighted sum of value vectors $\mathbf{v}_k$, not yet divided by $\ell$.
- $e^{\,m - m^{\text{new}}}$ = a correction factor: if a bigger max shows up, everything collected so far is scaled down to the new reference point.

### Tiny worked example (ours)
One query, three keys, scores $[1, 2 \mid 3]$ (two blocks), scalar values $v = [10, 20 \mid 30]$.

**Normal softmax:** $e^1, e^2, e^3 = 2.718, 7.389, 20.086$; sum $= 30.193$; weights $= 0.0900, 0.2447, 0.6652$; output $= 0.900 + 4.895 + 19.957 = 25.752$.

**Online, block 1** (scores 1, 2): $m = 2$, $\ell = e^{-1} + e^{0} = 1.3679$, $o = e^{-1}\cdot10 + 1\cdot20 = 23.679$.

**Online, block 2** (score 3): $m^{\text{new}} = 3$, correction $e^{2-3} = 0.3679$.
$\ell = 0.3679\cdot1.3679 + 1 = 1.5032$, $o = 0.3679\cdot23.679 + 1\cdot30 = 38.711$.
Output $= 38.711 / 1.5032 = 25.752$. ✔ Same answer, but we never needed all three scores at once.

### Using it from PyTorch (not from the book)
```python
import torch.nn.functional as F
# queries, keys, values: (b, num_heads, num_tokens, head_dim), as in Listing 3.5
context_vec = F.scaled_dot_product_attention(
    queries, keys, values, dropout_p=0.0, is_causal=True)
```
This one call replaces the score / mask / softmax / dropout / `@ values` lines of the
book's `MultiHeadAttention`. PyTorch picks a FlashAttention kernel when the GPU,
dtype, and arguments allow it. → [[pytorch]], [[causal-attention-mask]], [[gpu-computing]]

## 🔴 Theory (Expert)
- **Exactness.** The online update is an exact algebraic identity: for any split of a row into blocks, $\sum_k e^{s_k}\mathbf{v}_k \big/ \sum_k e^{s_k}$ is invariant to the reference shift $m$, and the rescaling $e^{m-m^{\text{new}}}$ re-expresses the partial sums in the new reference. Results differ from the naive kernel only by floating-point rounding order. → [[floating-point-precision]]
- **IO complexity (Dao et al. 2022).** With sequence length $N$, head dim $d$, and SRAM size $M$ ($d \le M \le Nd$), standard attention makes $\Theta(Nd + N^2)$ HBM accesses; FlashAttention makes $\Theta(N^2 d^2 M^{-1})$, and the paper proves no exact algorithm can do asymptotically better for all $M$ in that range. FLOPs stay $\Theta(N^2 d)$ — the speedup comes from memory traffic, not arithmetic.
- **Memory.** Extra memory drops from $\Theta(N^2)$ (the score/probability matrices) to $\Theta(N)$ (the per-row statistics $m, \ell$, saved as a log-sum-exp). This is what makes long [[context-length|context lengths]] practical.
- **Backward pass.** Instead of storing the $N\times N$ attention probabilities for [[backpropagation]], FlashAttention **recomputes** them tile by tile from $Q, K, V$ and the saved log-sum-exp. It does more FLOPs but is still faster, a classic recompute-vs-memory trade (gradient checkpointing). Dropout masks are regenerated from the saved RNG state. → [[dropout]], [[automatic-differentiation]]
- **Causal masking** ([[causal-attention]]) lets the kernel skip key blocks that lie entirely in the future of a query block, nearly halving the work.
- **FlashAttention-2** (Dao 2023) cuts non-matmul FLOPs (defers the $1/\ell$ rescaling to the end), parallelizes over the sequence dimension as well as batch and heads, and splits work between GPU warps to avoid shared-memory traffic — roughly 2× over v1 per the paper. (Later versions target newer GPU hardware; not in the book.)
- **Not an approximation.** Unlike sparse or low-rank attention variants (e.g. Linformer-style [[low-rank-approximation]]), FlashAttention leaves the model, its [[model-parameters]], and its outputs unchanged, so it is a drop-in replacement for [[multi-head-attention]] in training and inference.

## Connections
- Builds on: [[scaled-dot-product-attention]], [[softmax]], [[numerical-stability]], [[batched-matrix-multiplication]], [[attention-scores]], [[gpu-computing]]
- Leads to: efficient long-context LLM training and inference ([[context-length]], [[large-language-model]])
- Used by: [[multi-head-attention]], [[self-attention]], [[causal-attention]], [[pytorch]] (`scaled_dot_product_attention`)
- Related: memory-saving training tricks such as [[galore]] and [[gradient-accumulation]]; [[parameter-count-and-memory]]
- People: [[tri-dao]], [[ashish-vaswani]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 3 references, book p. 291 (FlashAttention description; Dao et al. 2022; Dao 2023).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 3 references, book p. 292 (PyTorch `scaled_dot_product_attention` supports FlashAttention; PyTorch `MultiHeadAttention`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6.2, book pp. 86–91 (the naive `MultiHeadAttention` that FlashAttention speeds up; GPT-2 context length 1,024).
- Online-softmax equations, IO-complexity bounds, recomputation, FlashAttention-2 details, the toy example and the PyTorch snippet are from the FlashAttention papers / PyTorch docs or our own arithmetic, not from the book.
