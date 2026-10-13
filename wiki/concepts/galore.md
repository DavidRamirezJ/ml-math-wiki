---
title: GaLore (Gradient Low-Rank Projection)
type: concept
tags: [llm, optimization, linear-algebra, memory-efficiency, pretraining]
difficulty: advanced
sources: [llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# GaLore (Gradient Low-Rank Projection)

> **One-liner:** GaLore saves GPU memory during LLM training by squeezing each weight matrix's gradient into a much smaller "low-rank" summary before the optimizer sees it, while still training every weight.

## 🟢 Intuition (Dummy)
Training with [[adamw-optimizer|AdamW]] is memory-hungry. Next to every weight the optimizer keeps
two extra "notebook" numbers (a running average of the gradient and of its square). So the
optimizer alone needs about **twice the memory of the model**.

GaLore's observation: for a big weight matrix, the gradient (the "which way to nudge every
weight" table) is mostly made of a few dominant patterns. So instead of keeping notebooks for the
whole huge table, GaLore:

1. finds the few main directions of the gradient (like summarizing a 768-column spreadsheet with
   its 128 most important columns),
2. lets the optimizer keep its notebooks only for that small summary,
3. expands the optimizer's suggested step back to full size and applies it to the full weights.

Every now and then it re-finds the main directions, so over time the model can still move in any
direction. The weights stay full-size; only the optimizer's bookkeeping shrinks.

In the book, GaLore appears only in the Chapter 5 reading list, as "a recent research project that
aims to make LLM pretraining more efficient". The code change "boils down to just replacing
PyTorch's AdamW optimizer in the training function with the GaLoreAdamW optimizer provided by the
galore-torch Python package" (book p. 294).

## 🟡 Mechanics (Practitioner)

### The book's one-line change
In the book's Chapter 5 training setup the optimizer is

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
```
(book p. 149). GaLore swaps that for `GaLoreAdamW` from `galore-torch` (book p. 294). The GaLore
repository's usage looks like this (from the repo, not from the book; argument names may change
between versions):

```python
from galore_torch import GaLoreAdamW
param_groups = [
    {"params": regular_params},                       # embeddings, LayerNorm, biases
    {"params": galore_params, "rank": 128,            # 2-D weight matrices
     "update_proj_gap": 200, "scale": 0.25, "proj_type": "std"},
]
optimizer = GaLoreAdamW(param_groups, lr=0.01)
```

### The math of one GaLore step (from Zhao et al. 2024, not in the book)
For a weight matrix $W\in\mathbb R^{m\times n}$ with gradient $G_t$ at step $t$:

$$P_t = U_{[:,\,1:r]} \ \text{ where } \ G_t = U\Sigma V^\top \quad(\text{recomputed every } T \text{ steps})$$

$$R_t = P_t^\top G_t \in \mathbb R^{r\times n}, \qquad N_t = \text{Adam}(R_t), \qquad W_{t+1} = W_t - \eta\,\alpha\,P_t N_t$$

Reading it symbol by symbol:
- $W$ = one weight [[matrix]] with $m$ rows and $n$ columns (e.g. a feed-forward layer);
- $G_t$ = the [[gradient]] of the loss with respect to $W$ (same shape as $W$);
- $G_t = U\Sigma V^\top$ = its [[singular-value-decomposition]]: $U$ holds the gradient's main
  "column directions", ordered from most to least important;
- $U_{[:,\,1:r]}$ = keep only the first $r$ columns of $U$; that is $P_t\in\mathbb R^{m\times r}$,
  the **projector**, with $r\ll m$ (the rank, e.g. 128);
- $P_t^\top G_t$ = "express the gradient in those $r$ directions": a small $r\times n$ matrix $R_t$;
- $\text{Adam}(R_t)$ = the usual Adam moment updates and normalized step, but run on the small
  $R_t$, so the moment buffers are $r\times n$ instead of $m\times n$;
- $P_tN_t$ = project the step back up to full $m\times n$ size;
- $\eta$ = learning rate, $\alpha$ = an extra scale factor (0.25 in the example above);
- $T$ = how often $P_t$ is refreshed (`update_proj_gap`, 200 above).

(Weight decay is applied as in AdamW; omitted here to keep the formula readable.)

### Worked memory numbers on GPT-2 124M (computed here)
GPT-2 small's feed-forward expansion layer is $768\times3072$ ([[feed-forward-network]]). With
plain AdamW the two moment buffers hold $2\times768\times3072=4{,}718{,}592$ numbers. With GaLore,
rank $r=128$ along the 768 side:

- moments on $R_t$: $2\times128\times3072=786{,}432$
- projector $P_t$: $768\times128=98{,}304$
- total: $884{,}736$, i.e. **18.75%** of AdamW's optimizer memory for that matrix.

The weights and their full gradient still exist; GaLore shrinks the *optimizer state*, which is
where AdamW spends $2\times$ the model size ([[parameter-count-and-memory]]).

## 🔴 Theory (Expert)
- **Why the top singular vectors.** By the Eckart–Young–Mirsky theorem, $P_tP_t^\top G_t$ (top-$r$
  left singular subspace) is the best rank-$r$ approximation of $G_t$ in Frobenius and spectral
  norm ([[low-rank-approximation]], [[norm]]). GaLore therefore discards the least possible gradient
  information for a given $r$. $P_t$ has orthonormal columns, so $P_t^\top P_t=I_r$ and the
  projection is an [[orthogonal-projection]].
- **Low-rank gradients, full-rank weights.** The paper argues that gradients of these layers
  become low-rank during training. Because $P_t$ is refreshed every $T$ steps, the accumulated
  update $\sum_t P_tN_t$ can have rank far above $r$. This is the key contrast with
  [[low-rank-adaptation|LoRA]] (book appendix E), which fixes the *weights' update* to the form
  $W_0+BA$ with rank $\le r$ for the whole run ([[basis-and-rank]]).
- **Cost of the SVD.** Computing an SVD of an $m\times n$ matrix costs about $O(mn\min(m,n))$, so it
  is done only every $T$ steps; amortized over $T=200$ steps it is small relative to forward and
  backward passes.
- **Reported results (from the paper's abstract, not the book).** Up to 65.5% less optimizer-state
  memory; with 8-bit optimizer states, pretraining a 7B-parameter LLaMA model on a single 24 GB
  consumer GPU without model parallelism, checkpointing or offloading. Treat these as the authors'
  claims.
- **Caveats.** Moment buffers live in the *current* subspace; when $P_t$ changes, the old moments
  are expressed in the old basis, a known approximation of the method. Rank $r$, $\alpha$ and $T$
  are new [[hyperparameter|hyperparameters]].

## Connections
- Builds on: [[singular-value-decomposition]], [[low-rank-approximation]], [[orthogonal-projection]], [[gradient]], [[adamw-optimizer]], [[matrix]]
- Leads to: memory-efficient [[pretraining]], [[parameter-efficient-fine-tuning]]
- Compare: [[low-rank-adaptation]], [[lora-rank-and-alpha]]
- Related: [[parameter-count-and-memory]], [[gradient-descent]], [[stochastic-gradient-descent]], [[llm-training-loop]], [[basis-and-rank]]
- People: [[jiawei-zhao]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Chapter 5 references, book p. 294 (GaLore description; swap AdamW for `GaLoreAdamW` from `galore-torch`; Zhao et al. 2024, arXiv 2403.03507; github.com/jiaweizzhao/GaLore).
- Same PDF, §5.2, book p. 149 (`AdamW(..., lr=0.0004, weight_decay=0.1)`).
- Projection formulas, repo usage, memory claims: Zhao et al. (2024) and the GaLore repository, not printed in the book. GPT-2 memory arithmetic computed here.
