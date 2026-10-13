---
title: Model Capacity
type: concept
tags: [llm, model-size, generalization, gpt-2]
difficulty: intermediate
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Model Capacity

> **One-liner:** A model's capacity is how much it is able to learn and remember. Bigger models usually have more of it.

## 🟢 Intuition (Dummy)
Think of a model as a notebook. A small notebook can hold a few simple notes. A huge notebook can hold detailed instructions, exceptions, and subtle tricks. If the job is "follow any instruction a person gives you", you need a lot of room for patterns: how to rewrite sentences, convert units, answer trivia, and so on.

That's exactly why the book switches models in Chapter 7. Earlier chapters used the small GPT-2 (124 million parameters). For instruction following, the author says this small model is "too limited in capacity". So the chapter uses GPT-2 medium (355 million parameters) instead (book pp. 226–227). At the end of the chapter, "using a larger pretrained model, which may have greater capacity to capture complex patterns and generate more accurate responses" comes back as one of the ways to improve the results (book p. 247).

Capacity is a double-edged sword, though. More room means the model can learn richer patterns, but it also costs more storage and memory, and with too little data a very roomy model can just memorize ([[overfitting]]).

## 🟡 Mechanics (Practitioner)

### How the book uses the word
The book doesn't give a formula for capacity. It uses the word in plain English: smaller models "lack the necessary capacity to learn and retain the intricate patterns and nuanced behaviors required for high-quality instruction following" (book pp. 226–227). In practice, the knob you turn is **model size**, meaning how many [[model-parameters]] it has.

### The GPT-2 family (Listing 7.7, book p. 227)

| Model | `emb_dim` ($d$) | `n_layers` ($L$) | `n_heads` | Parameters |
|---|---|---|---|---|
| gpt2-small | 768 | 12 | 12 | 124M |
| **gpt2-medium** (chosen) | 1024 | 24 | 16 | 355M |
| gpt2-large | 1280 | 36 | 20 | 774M |
| gpt2-xl | 1600 | 48 | 25 | 1558M |

All four share `vocab_size = 50257` and `context_length = 1024`. Switching sizes is literally one string:

```python
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
```

The medium checkpoint is about **1.42 GB** on disk, roughly **3×** the small one (book p. 227).

### A rule of thumb for parameter count
A handy approximation (my addition, not from §7.5) is:

$$
N \;\approx\; 12\,L\,d^{2} \;+\; V d \;+\; C d
$$

Symbol by symbol:
- $N$ = total number of parameters.
- $L$ = number of [[transformer-block|transformer blocks]] (`n_layers`).
- $d$ = embedding dimension (`emb_dim`).
- $12\,d^2$ = the weights in one block: about $4d^2$ for the attention projections (query, key, value, output) plus $8d^2$ for the [[feed-forward-network]] ($d \to 4d \to d$).
- $V d$ = the token-embedding matrix ($V = 50257$ rows, one per token, each of length $d$).
- $C d$ = the [[positional-embedding]] matrix ($C = 1024$ positions).

Plug in GPT-2 medium: $12 \cdot 24 \cdot 1024^2 \approx 302.0\text{M}$, plus $50257 \cdot 1024 \approx 51.5\text{M}$, plus $1024 \cdot 1024 \approx 1.0\text{M}$, giving about **354.5M ≈ 355M**. For small: $12\cdot 12\cdot 768^2 \approx 84.9\text{M} + 38.6\text{M} + 0.8\text{M} \approx 124.3\text{M}$. The match works because GPT-2 reuses the token embedding as its output layer ([[weight-tying]]). Notice capacity grows with $d^2$: doubling the width roughly quadruples the block parameters. See [[parameter-count-and-memory]].

### What capacity bought (and didn't) in Chapter 7
The book scores models with an LLM judge ([[llm-as-a-judge]]) on the same 110-item test set (book pp. 246–247):

| Model | Average score (0–100) |
|---|---|
| GPT-2 medium 355M, instruction-fine-tuned | 50.32 |
| Llama 3 8B base (no instruction tuning) | 58.51 |
| Llama 3 8B instruct | 82.6 |

Reading the table (my comparison; the book doesn't spell it out): an 8-billion-parameter base model with **no** instruction tuning already beats our fine-tuned 355M model, and the same 8B model with instruction tuning jumps to 82.6. So both capacity *and* tuning matter.

## 🔴 Theory (Expert)
- **Classical capacity.** In learning theory, capacity is the richness of the hypothesis class a model can represent, measured, for example, by VC dimension or Rademacher complexity. Higher capacity lowers approximation error but can raise estimation error: the bias–variance trade-off in [[overfitting-and-underfitting]]. Parameter count is only a rough stand-in for this.
- **Why big nets don't simply overfit.** Modern LLMs often sit in an over-parameterized regime where larger models generalize *better* ("double descent"), helped by implicit regularization from [[stochastic-gradient-descent]], huge [[pretraining]] corpora, and techniques like [[dropout]] and [[weight-decay]]. (Background knowledge, not from the book.)
- **Scaling laws.** Empirically, pretraining loss falls roughly as a power law in parameters $N$ and data $D$ (Kaplan et al. 2020; Hoffmann et al. 2022). That is one reason the 8B Llama baseline is so much stronger. (Background knowledge, not cited in the book.)
- **Capacity vs. knowledge.** Appendix B cites Gekhman et al. (2024): LLMs pick up factual knowledge mainly during pretraining, and fine-tuning mostly improves how they *use* it (book pp. 297–298). So for instruction following, a bigger pretrained model contributes stored knowledge that a small fine-tuning set cannot add.
- **Cost side.** Memory scales with $N \times$ bytes per parameter ([[floating-point-precision]]); training memory is several times larger because of gradients and optimizer state ([[adamw-optimizer]]). Parameter-efficient methods like [[low-rank-adaptation]] cut the *trainable* parameters without cutting the model's capacity (Exercise 7.4, book p. 247).

## Connections
- Builds on: [[model-parameters]], [[gpt-model-configuration]], [[parameter-count-and-memory]], [[transformer-block]], [[embedding-dimension]]
- Leads to: [[overfitting-and-underfitting]], [[instruction-fine-tuning]], [[llm-as-a-judge]], [[low-rank-adaptation]]
- Related: [[gpt-2]], [[overfitting]], [[regularization]], [[pretrained-weights]], [[hyperparameter]], [[baseline-evaluation]]
- People: [[alec-radford]] (GPT-2), [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5 (124M too limited in capacity; Listing 7.7 configs; 1.42 GB), book pp. 226–227.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8 (larger model as an improvement strategy; reference scores 50.32 / 58.51 / 82.6), book pp. 246–247.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, Chapter 7 references (Gekhman et al. 2024), book pp. 297–298.
