---
title: Tri Dao
type: person
tags: [efficient-attention, gpu, state-space-models, systems]
sources: [llm-ch01-understanding-large-language-models, llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Tri Dao (active 2010s–present)

> **Why they matter to ML:** creator of **FlashAttention**, the fast, memory-saving way to compute exactly the same attention, and co-author of the **Mamba** non-transformer LLM.

## Life (brief)
Tri Dao did his PhD in computer science at Stanford (advised by Christopher Ré and Stefano Ermon). He then became an assistant professor at Princeton and chief scientist at Together AI. (Biographical details are general knowledge. The book names him only in appendix B citations: "Dao et al.", "Dao", and "Gu and Dao.")

## Key contributions
- **FlashAttention** (Dao et al. 2022, arXiv 2205.14135). The book's description: "a highly efficient implementation of a self-attention mechanism, which accelerates the computation process by optimizing memory access patterns. FlashAttention is mathematically the same as the standard self-attention mechanism." → [[flashattention]], [[self-attention]], [[scaled-dot-product-attention]]
  - **What it computes** is the usual attention: $\mathrm{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)V$. Symbol by symbol: $Q$ = the matrix of query vectors (one row per token); $K$ = the key matrix; $K^\top$ = $K$ [[matrix-transpose|transposed]], so $QK^\top$ holds every query–key [[inner-product|dot product]] (the [[attention-scores]]); $\sqrt{d_k}$ = square root of the key dimension, which keeps the scores from getting too big; [[softmax]] = turn each row of scores into [[attention-weights]] that add up to 1; $V$ = the value matrix, which the weights mix into [[context-vector|context vectors]]. → [[query-key-value]]
  - **What it changes** (external context) is *how*: the slow way writes the full $n \times n$ score matrix to GPU main memory. FlashAttention cuts $Q$, $K$, $V$ into small tiles that fit in fast on-chip memory and keeps a running max and running sum so the softmax can be built piece by piece ("online softmax"). Memory use grows linearly with [[context-length]] $n$ instead of with $n^2$, and the answer is exact, not approximate. → [[gpu-computing]], [[numerical-stability]]
  - In PyTorch you usually get it for free through `torch.nn.functional.scaled_dot_product_attention`, which the book's references say "supports FlashAttention" for both self-attention and causal attention (the book notes this function "is beta and subject to change"). PyTorch's efficient `MultiHeadAttention` class builds on it. → [[pytorch]], [[multi-head-attention]]
- **FlashAttention-2** (Dao 2023, arXiv 2307.08691). A faster successor with better parallelism and work partitioning across GPU threads (book appendix B). → [[flashattention]]
- **Mamba** (Gu and Dao 2023, arXiv 2312.00752), with [[albert-gu]]. A selective state-space model that the book lists as an example "that not all LLMs need to be based on the transformer architecture." See [[albert-gu]] for the math.

## Connections
- Concepts: [[flashattention]], [[self-attention]], [[scaled-dot-product-attention]], [[causal-attention]], [[softmax]], [[gpu-computing]], [[context-length]], [[recurrent-neural-network]]
- People: [[albert-gu]] (Mamba co-author), [[ashish-vaswani]] (scaled dot-product attention)
- Appears in sources: [[llm-ch01-understanding-large-language-models]], [[llm-ch03-coding-attention-mechanisms]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 3 references, book pp. 291–292 (FlashAttention and FlashAttention-2 citations and description; PyTorch `scaled_dot_product_attention` supports FlashAttention).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 1 references, book p. 290 (Mamba, Gu and Dao 2023).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book p. 10 (non-transformer LLMs aim for better computational efficiency).
- Biography and the tiling / online-softmax explanation are general knowledge from the FlashAttention papers, not from the book.
