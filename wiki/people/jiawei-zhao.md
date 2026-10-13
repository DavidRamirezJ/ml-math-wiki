---
title: Jiawei Zhao
type: person
tags: [optimization, memory-efficiency, low-rank, pretraining]
sources: [llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Jiawei Zhao (active 2020s–present)

> **Why they matter to ML:** first author of **GaLore**, a way to pretrain big LLMs with much less GPU memory by squeezing the gradients down to a low rank.

## Life (brief)
The book names him only as "Zhao et al." in appendix B, and links his code repository `github.com/jiaweizzhao/GaLore` (book p. 294).

General knowledge, not from the book: Jiawei Zhao did his PhD at Caltech, advised by Anima Anandkumar, and later joined Meta AI (FAIR). GaLore (2024) was written with Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar and Yuandong Tian.

## Key contributions
- **GaLore** (Zhao et al. 2024, "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", arXiv 2403.03507). → [[galore]]
  - **What the book says.** "GaLore is a recent research project that aims to make LLM pretraining more efficient. The required code change boils down to just replacing PyTorch's AdamW optimizer in the training function with the GaLoreAdamW optimizer provided by the galore-torch Python package" (book p. 294). → [[pretraining]], [[adamw-optimizer]], [[llm-training-loop]]
  - **Why memory is the problem** (external context). [[adamw-optimizer|AdamW]] keeps two extra numbers for *every* weight (a running mean of the [[gradient]] and a running mean of the squared gradient). For a weight [[matrix]] of size $m \times n$ that is $2mn$ extra numbers. For a big model this "optimizer state" can take more memory than the weights. → [[parameter-count-and-memory]]
  - **The idea** (external context, from the paper). The gradient of a weight matrix is usually close to low rank, so you can store a much smaller version of it:
    $$R_t = P_t^{\top} G_t, \qquad W_{t+1} = W_t - \eta\, P_t\, \rho(R_t)$$
    Symbol by symbol: $W_t$ = the weight matrix ($m \times n$) at step $t$; $G_t$ = its gradient, also $m \times n$; $P_t$ = a "projection" matrix of size $m \times r$ with $r \ll m$ ("$r$ much smaller than $m$"); $P_t^{\top}$ = $P_t$ [[matrix-transpose|transposed]]; $R_t$ = the small $r \times n$ compressed gradient; $\rho(\cdot)$ ("rho") = the normal AdamW update, but run on the small $R_t$, so its optimizer states are only $r \times n$; multiplying by $P_t$ again blows the update back up to $m \times n$; $\eta$ ("eta") = the [[learning-rate]].
  - **Where $P_t$ comes from.** Every few hundred steps, GaLore takes the [[singular-value-decomposition]] of the current gradient, $G_t = U \Sigma V^{\top}$, and keeps the first $r$ columns of $U$ (the top $r$ left singular vectors) as $P_t$. That is exactly the best rank-$r$ [[low-rank-approximation]] direction (Eckart–Young). Changing $P_t$ over time lets the full weight matrix still learn in all directions, unlike [[low-rank-adaptation|LoRA]], which fixes the low-rank form of the *update* itself. → [[basis-and-rank]], [[parameter-efficient-fine-tuning]]
  - **Small numeric feel** (our example, not from the book): a $4096 \times 4096$ layer with rank $r=128$. AdamW states: $2 \times 4096 \times 4096 \approx 33.6$ million numbers. GaLore states: $2 \times 128 \times 4096 \approx 1.05$ million numbers — about 32 times less.
  - **In code** (the book's description, using the `galore-torch` package): swap `torch.optim.AdamW(...)` for `GaLoreAdamW(...)` in the training function; the rest of the book's training loop stays the same.

## Connections
- Concepts: [[galore]], [[adamw-optimizer]], [[gradient]], [[singular-value-decomposition]], [[low-rank-approximation]], [[basis-and-rank]], [[low-rank-adaptation]], [[pretraining]], [[parameter-count-and-memory]], [[gpu-computing]]
- People: [[sebastian-raschka]] (book author), [[carl-eckart]] and [[gale-young]] (best low-rank approximation theorem)
- Appears in sources: [[llm-ch05-pretraining-on-unlabeled-data]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 5 references, book p. 294 (GaLore citation, GaLoreAdamW drop-in replacement, code repository).
- `raw/mml-book.pdf` §4.5–4.6 for the SVD and Eckart–Young theorem used to explain the projection (see [[singular-value-decomposition]], [[low-rank-approximation]]).
- Biography, co-authors, the update formula, and the memory example are general knowledge from the GaLore paper, not from the book.
