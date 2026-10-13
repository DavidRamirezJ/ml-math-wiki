---
title: "LLM Ch.3 — Coding Attention Mechanisms"
type: source
tags: [llm, attention, transformer, linear-algebra, source]
created: 2026-10-13
updated: 2026-10-13
---

# LLM Ch.3 — Coding Attention Mechanisms

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 3, **book pp. 50–91** (PDF pages 72–113)
- **Type:** book chapter — *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025)
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13 (every printed tensor checked digit by digit in the staging notes)
- *Page rule:* PDF page = book page + 22.
- **Related:** references for this chapter are in Appendix B (book pp. 291–292); exercise solutions are in Appendix C (book p. 301).

This is the chapter where the LLM gets its "brain". It builds attention in four steps, each one
a small upgrade on the last: simplified self-attention → self-attention with trainable weights →
causal attention → multi-head attention. The final `MultiHeadAttention` class is the exact module
that the GPT model in Ch. 4 plugs in.

## Section coverage (all of Ch. 3 accounted for)
| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Chapter intro ("This chapter covers"; Figs. 3.1–3.2: the four attention variants) | 50–52 | ✅ [[attention-mechanism]], [[self-attention]], [[causal-attention]], [[multi-head-attention]], [[llm-building-roadmap]] |
| 3.1 | The problem with modeling long sequences | 52–54 | ✅ [[machine-translation]], [[encoder-decoder-architecture]], [[recurrent-neural-network]], [[hidden-state]], [[long-range-dependencies]] |
| 3.2 | Capturing data dependencies with attention mechanisms | 54–55 | ✅ [[bahdanau-attention]], [[attention-weights]], [[self-attention]], [[transformer-architecture]], [[decoder-only-transformer]] |
| 3.3 | Attending to different parts of the input with self-attention (incl. sidebar "The 'self' in self-attention") | 55–56 | ✅ [[self-attention]], [[context-vector]] |
| 3.3.1 | A simple self-attention mechanism without trainable weights | 56–61 | ✅ [[context-vector]], [[attention-scores]], [[inner-product]], [[attention-weights]], [[softmax]], [[numerical-stability]], [[convex-combination]] |
| 3.3.2 | Computing attention weights for all input tokens | 61–64 | ✅ [[gram-matrix]], [[matrix-multiplication]], [[matrix-transpose]], [[softmax]], [[context-vector]] |
| 3.4 | Implementing self-attention with trainable weights (Fig. 3.13 roadmap) | 64–65 | ✅ [[scaled-dot-product-attention]] |
| 3.4.1 | Computing the attention weights step by step (sidebars: weight parameters vs attention weights; rationale for scaling; why Q, K, V) | 65–70 | ✅ [[query-key-value]], [[scaled-dot-product-attention]], [[attention-scores]], [[attention-weights]], [[model-parameters]], [[vanishing-and-exploding-gradients]], [[covariance-and-variance]] |
| 3.4.2 | Implementing a compact self-attention Python class (Listings 3.1–3.2, Exercise 3.1) | 70–74 | ✅ [[pytorch-nn-module]], [[linear-layer]], [[weight-initialization]], [[linear-mapping]] |
| 3.5 | Hiding future words with causal attention (Fig. 3.19) | 74–75 | ✅ [[causal-attention]], [[causal-attention-mask]] |
| 3.5.1 | Applying a causal attention mask (sidebar: information leakage) | 75–78 | ✅ [[causal-attention-mask]], [[softmax]], [[broadcasting]] |
| 3.5.2 | Masking additional attention weights with dropout | 78–80 | ✅ [[dropout]], [[regularization]], [[overfitting]], [[expected-value-and-mean]] |
| 3.5.3 | Implementing a compact causal attention class (Listing 3.3) | 80–82 | ✅ [[causal-attention]], [[batched-matrix-multiplication]], [[tensor]], [[mini-batch]] |
| 3.6 | Extending single-head attention to multi-head attention | 82 | ✅ [[multi-head-attention]] |
| 3.6.1 | Stacking multiple single-head attention layers (Listing 3.4, Exercise 3.2) | 82–85 | ✅ [[multi-head-attention]] |
| 3.6.2 | Implementing multi-head attention with weight splits (Listing 3.5, Exercise 3.3) | 86–91 | ✅ [[weight-splitting-multi-head-attention]], [[tensor-reshaping-and-transposing]], [[batched-matrix-multiplication]], [[output-projection]], [[gpt-2]] |
| — | Summary | 91 | — (recap; bullets folded into TL;DR below) |
| App. B | References for Ch. 3 | 291–292 | ✅ [[flashattention]] + person pages (see below) |
| App. C | Solutions to Exercises 3.1–3.3 | 301 | ✅ see "Exercises" below |

> ✅ **Every section and subsection of Ch. 3 (intro, 3.1–3.6.2, summary) maps to at least one
> wiki page.** §3.1, §3.2 and §3.6 have no numbered sub-subsections beyond those listed.

## TL;DR (5 bullets)
- **Why attention exists:** encoder–decoder [[recurrent-neural-network|RNNs]] squeeze a whole sentence into one [[hidden-state]], so they lose context on long sentences. [[bahdanau-attention]] (2014) let the decoder look back at *every* input token; the 2017 [[transformer-architecture]] then dropped RNNs and kept only [[self-attention]].
- **Self-attention in three steps:** (1) [[attention-scores]] = dot products ([[inner-product]]) between tokens, (2) [[softmax]] turns each row into [[attention-weights]] that are positive and sum to 1, (3) each [[context-vector]] is a weighted sum ([[convex-combination]]) of the inputs. In matrix form: $Z = \operatorname{softmax}(XX^\top)X$.
- **Trainable weights = [[scaled-dot-product-attention]]:** three learned matrices project tokens into [[query-key-value|queries, keys and values]]; $Z = \operatorname{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V$. Dividing by $\sqrt{d_k}$ stops softmax from saturating, which would kill the gradients.
- **[[causal-attention]]:** a [[causal-attention-mask]] (fill above-diagonal scores with $-\infty$ before softmax) stops each token from seeing the future, which is needed for next-token prediction. [[dropout]] on the attention weights adds [[regularization]] during training only.
- **[[multi-head-attention]]:** run several causal attention "heads" in parallel and concatenate them. The efficient version uses one big projection, splits it with `.view`/`.transpose` ([[weight-splitting-multi-head-attention]]), does [[batched-matrix-multiplication]], and finishes with an [[output-projection]].

## Key concepts introduced
**Motivation (§3.1–3.2)**
- [[attention-mechanism]] — lets a model weigh how relevant each input token is to the token it is producing
- [[machine-translation]] — the running example; word-by-word translation fails (Fig. 3.3)
- [[encoder-decoder-architecture]] — encoder reads the full input, decoder writes the output
- [[recurrent-neural-network]] — outputs of previous steps feed the current step
- [[hidden-state]] — the internal "memory" vector of an RNN; a bottleneck when it is the only link
- [[long-range-dependencies]] — words that depend on far-away words; where RNNs lose context
- [[bahdanau-attention]] — 2014 RNN attention; decoder selectively reads all input tokens
- [[self-attention]] — each position attends to all positions in the *same* sequence

**Simplified self-attention (§3.3)**
- [[context-vector]] — an enriched embedding that mixes in information from all tokens
- [[attention-scores]] — unnormalized similarity values $\omega_{ij}$
- [[inner-product]] — the dot product as a similarity measure
- [[attention-weights]] — normalized scores $\alpha_{ij}$ (≥ 0, sum to 1)
- [[softmax]] — the normalizer; positive outputs, good gradients, handles extreme values
- [[numerical-stability]] — naive softmax can overflow/underflow; use `torch.softmax`
- [[convex-combination]] — a context vector is a weighted average of inputs
- [[gram-matrix]] — $XX^\top$, the all-pairs dot-product matrix (symmetric here)

**Trainable self-attention (§3.4)**
- [[query-key-value]] — three learned projections, named after database lookup
- [[scaled-dot-product-attention]] — $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$
- [[model-parameters]] — "weight parameters" (learned) vs attention weights (dynamic, per input)
- [[vanishing-and-exploding-gradients]] — why large dot products need scaling
- [[pytorch-nn-module]], [[linear-layer]], [[weight-initialization]] — the PyTorch building blocks

**Causal attention and dropout (§3.5)**
- [[causal-attention]] — masked attention; only previous and current tokens are visible
- [[causal-attention-mask]] — `tril`/`triu` masks; $-\infty$ trick; no information leakage
- [[dropout]] — randomly zero units during training; scale survivors by $1/(1-p)$
- [[regularization]], [[overfitting]] — why dropout is used
- [[expected-value-and-mean]] — inverted dropout keeps the expected value unchanged
- [[batched-matrix-multiplication]], [[tensor]], [[broadcasting]] — 3-D batch shapes (2, 6, 3)

**Multi-head attention (§3.6)**
- [[multi-head-attention]] — several heads, each with its own projections, concatenated
- [[weight-splitting-multi-head-attention]] — one big $W_q$ split into heads (block matrices)
- [[tensor-reshaping-and-transposing]] — `.view`, `.transpose`, `.contiguous`
- [[output-projection]] — final `Linear(d_out, d_out)` that mixes the heads
- [[gpt-2]] — reference sizes for heads and embedding width

**Further reading (Appendix B)**
- [[flashattention]] — exact attention with IO-aware memory access; same math, faster

## Notable results / numbers
- **Running example:** "Your journey starts with one step" — $T = 6$ tokens, $d = 3$ embedding dimension (book p. 57). Query $x^{(2)}$ = "journey" = [0.55, 0.87, 0.66].
- **Simplified attention (§3.3.1):** scores for "journey" = [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]; softmax weights = [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]; context vector $z^{(2)}$ = [0.4419, 0.6515, 0.5683] (book pp. 58–60). "journey" attends most to itself and to the nearly identical "starts".
- **Sum-normalization vs softmax:** plain division by the sum gives [0.1455, 0.2278, …]; softmax is preferred because it is always positive, handles extreme values, and has better gradients (book pp. 59–60).
- **Matrix form (§3.3.2):** `inputs @ inputs.T` replaces nested loops and gives the 6×6 score matrix; with no trainable weights it is **symmetric** (book pp. 62–63). `softmax(dim=-1)` makes every row sum to 1.
- **Trainable attention (§3.4.1):** $d_{in}=3$, $d_{out}=2$, seed 123 → `query_2` = [0.4306, 1.4551]; $\omega_{22} = 1.8524$; weights after $\sqrt{d_k}=\sqrt 2$ scaling = [0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]; $z^{(2)}$ = [0.3061, 0.8210] (book pp. 66–70).
- **Why scale by $\sqrt{d_k}$:** GPT-like embeddings are usually **> 1,000** dimensions; large dot products push softmax towards a step function with near-zero gradients (book p. 69). (Wiki addition: if $q, k$ have independent unit-variance components, $\operatorname{Var}(q\cdot k) = d_k$.)
- **`nn.Linear` stores $W^\top$:** shape $(d_{out}, d_{in})$, computes $xW^\top$; it also has a better initialization than `torch.rand` (book pp. 71–73).
- **Causal mask, two equivalent ways (§3.5.1):** (A) softmax → zero the upper triangle → renormalize rows; (B) fill the upper triangle with $-\infty$ → softmax. Both give rows [1.0], [0.5517, 0.4483], [0.3800, 0.3097, 0.3103], … (book pp. 76–78).
- **No information leakage:** renormalizing after masking equals a softmax over only the unmasked positions, because the full denominator cancels (book p. 77):
  $$\frac{e^{z_{ij}}/S_i}{\sum_{k\le i} e^{z_{ik}}/S_i} = \frac{e^{z_{ij}}}{\sum_{k\le i} e^{z_{ik}}}.$$
- **Dropout (§3.5.2):** demo rate **0.5** (survivors become ×2); GPT training later uses **0.1 or 0.2**. Applied *after* the attention weights (the more common of two placements) (book pp. 78–80). Dropout output can differ across operating systems (PyTorch issue #121595, book p. 80).
- **Shapes:** batch (2, 6, 3) → `CausalAttention` (2, 6, 2) → wrapper with 2 heads (2, 6, 4) → `MultiHeadAttention` with `d_out=2` (2, 6, 2) (book pp. 80–90).
- **Multi-head shape trace:** (b, T, d_out) → `.view` (b, T, h, d_h) → `.transpose(1,2)` (b, h, T, d_h) → scores (b, h, T, T) → back to (b, T, d_out); scaling uses $\sqrt{d_h}$, the *head* dimension (book p. 87).
- **Batched matmul demo:** a (1, 2, 3, 4) tensor times its last-two-axes transpose gives one 3×3 Gram matrix per head, e.g. head 0 entry $[0,0] = 1.3208$ (book p. 89).
- **GPT-2 sizes:** smallest = **117M parameters, 12 heads, 768-dim, context 1,024**; largest = **1.5B parameters, 25 heads, 1,600-dim** (book pp. 90–91). Both give head dim 64 (wiki arithmetic).

> ⚠️ **Book inconsistency (Fig. 3.25, p. 84):** the caption says the output columns correspond
> to "d_out=4". The argument passed is `d_out=2` per head; 4 is the concatenated width
> (`d_out × num_heads`).

## Exercises
| # | Book p. | Task | Solution idea (Appendix C, p. 301) |
|---|---|---|---|
| 3.1 | 73 | Copy weights from `SelfAttention_v2` (uses `nn.Linear`) into `SelfAttention_v1` (uses raw `nn.Parameter`) so both give identical outputs. | `sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)` (same for key and value). The `.T` is needed because `nn.Linear` stores the weight as $(d_{out}, d_{in})$ and computes $xW^\top$. See [[linear-layer]], [[matrix-transpose]]. |
| 3.2 | 85 | Make `MultiHeadAttentionWrapper(num_heads=2)` return 2-D context vectors without changing the class or `num_heads`. | Set `d_out = 1`. The wrapper concatenates heads, so output width = `num_heads × d_out` = 2 × 1 = 2. See [[multi-head-attention]]. |
| 3.3 | 91 | Initialize a `MultiHeadAttention` module the size of the smallest GPT-2. | `MultiHeadAttention(768, 768, 1024, 0.0, 12)` → head_dim 64. Wiki-derived parameter count: $3\cdot768^2 + (768^2 + 768) = 2{,}360{,}064$. See [[gpt-2]], [[parameter-count-and-memory]]. |

## References cited (Appendix B, Ch. 3 group, book pp. 291–292)
1. [[dzmitry-bahdanau]], Kyunghyun Cho, [[yoshua-bengio]] (2014), "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 1409.0473 — [[bahdanau-attention]] for RNN translation.
2. [[ashish-vaswani]] et al. (2017), "Attention Is All You Need", arXiv 1706.03762 — introduced self-attention as [[scaled-dot-product-attention]] (also cited in Ch. 1).
3. [[tri-dao]] et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", arXiv 2205.14135 — [[flashattention]]; same math as standard attention, better memory access.
4. [[tri-dao]] (2023), "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", arXiv 2307.08691 — faster successor.
5. PyTorch `scaled_dot_product_attention` docs (mng.bz/NRJd) — built-in self/causal attention with FlashAttention support (beta).
6. PyTorch `MultiHeadAttention` docs (mng.bz/DdJV) — efficient built-in multi-head class.
7. [[nitish-srivastava]], [[geoffrey-hinton]] et al. (2014), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 15 — [[dropout]].
8. He and Hofmann (2023), "Simplifying Transformer Blocks", arXiv 2311.01906 — good results are possible without the value matrix and the [[output-projection]] (the reason the book calls `out_proj` optional).

## People
- [[dzmitry-bahdanau]] and [[yoshua-bengio]] — attention for neural machine translation (2014).
- [[ashish-vaswani]] — lead author of the transformer paper.
- [[nitish-srivastava]] and [[geoffrey-hinton]] — dropout (2014).
- [[tri-dao]] — FlashAttention 1 and 2.
- No person is named in the chapter body itself; all names come from Appendix B.

## What this changes in the wiki
- **Adds the core of the transformer** to the wiki: 25+ concept pages from [[attention-mechanism]]
  to [[weight-splitting-multi-head-attention]], plus person pages for [[dzmitry-bahdanau]],
  [[yoshua-bengio]], [[nitish-srivastava]] and [[geoffrey-hinton]].
- **Strong links back to the MML pages.** Attention is mostly linear algebra the wiki already
  covers: [[inner-product]] (scores), [[matrix]] and [[linear-mapping]] (projections $XW$),
  [[gram-matrix]] ($XX^\top$), [[convex-combination]] (context vectors), [[softmax]] and
  [[probability-distribution]] (rows of weights), [[expected-value-and-mean]] (inverted dropout).
  The existing pages [[inner-product]], [[regularization]] and [[expected-value-and-mean]] get a
  new "Used by: attention / dropout" connection.
- **New ML-practice pages** that MML does not cover: [[causal-attention-mask]], [[dropout]],
  [[batched-matrix-multiplication]], [[tensor-reshaping-and-transposing]], [[output-projection]],
  [[flashattention]].
- **No contradictions** with existing pages. One book-internal inconsistency (Fig. 3.25 caption,
  "d_out=4") is flagged above.
- **Next:** Ch. 4 wraps `MultiHeadAttention` into the [[transformer-block]] with
  [[layer-normalization]], [[gelu]], a [[feed-forward-network]] and [[shortcut-connections]] to
  build the full [[gpt-architecture]].
