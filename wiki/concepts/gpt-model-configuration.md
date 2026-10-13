---
title: GPT Model Configuration (GPT_CONFIG_124M)
type: concept
tags: [llm, hyperparameters, gpt, model-architecture, pytorch]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appD-training-loop-bells-and-whistles, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# GPT Model Configuration (GPT_CONFIG_124M)

> **One-liner:** A GPT model's whole shape is set by a short dictionary of seven numbers — vocabulary size, context length, embedding size, number of heads, number of layers, dropout rate, and a bias switch — and changing those numbers is all you need to go from a tiny GPT-2 to a huge one.

*Also covers:* model-configuration hyperparameters.

## 🟢 Intuition (Dummy)
Think of ordering a pizza: you don't rewrite the recipe, you just fill in a form —
size, number of slices, toppings. The `GPT_CONFIG_124M` dictionary is that order
form for a GPT model. The code (`GPTModel`) is the recipe and never changes. The
dictionary says *how big* to make it:

- how many different words it knows (**vocab_size**),
- how much text it can read at once (**context_length**),
- how many numbers describe each word (**emb_dim**),
- how many "attention spotlights" each layer has (**n_heads**),
- how many layers are stacked (**n_layers**),
- how much random "forgetting" is used to prevent memorizing (**drop_rate**),
- whether the attention layers have extra offset terms (**qkv_bias**).

These are [[hyperparameter|hyperparameters]]: *you* choose them before training.
The model's [[model-parameters|parameters]] (the 124 million weights) are what
training learns.

## 🟡 Mechanics (Practitioner)

### The dictionary (§4.1, p. 95)
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}
```

| Key | Value | What it controls | Where it is used in the code |
|---|---|---|---|
| `vocab_size` | 50,257 | Size of the [[byte-pair-encoding\|BPE]] [[vocabulary]] | rows of `tok_emb`; columns of the [[logits]] (`out_head`) |
| `context_length` | 1,024 | Max tokens the model can see ([[context-length]]) | rows of `pos_emb`; size of the [[causal-attention-mask]] |
| `emb_dim` | 768 | Length $d$ of every token vector ([[embedding-dimension]]) | every layer's width; FFN hidden size $4d=3072$ |
| `n_heads` | 12 | Number of attention heads ([[multi-head-attention]]) | head size $d/h = 64$ |
| `n_layers` | 12 | Number of stacked [[transformer-block\|transformer blocks]] | `nn.Sequential(*[TransformerBlock(cfg) ...])` |
| `drop_rate` | 0.1 | [[dropout]]: drop 10% of hidden units at random (against [[overfitting]]) | embedding, attention weights, shortcut paths |
| `qkv_bias` | False | Whether the Q/K/V `Linear` layers add a [[bias-vector]] | `MultiHeadAttention(..., qkv_bias=...)` |

The bias switch in math (a gloss, not printed in the book):

$$
\mathbf{q} = \mathbf{x}W_q + \mathbf{b}_q \quad(\texttt{qkv\_bias=True}) \qquad\text{vs.}\qquad \mathbf{q} = \mathbf{x}W_q \quad(\texttt{qkv\_bias=False})
$$

- $\mathbf{x}$ = one token's input vector (length 768).
- $W_q$ = the learned query weight [[matrix]] ($768 \times 768$).
- $\mathbf{b}_q$ = a learned bias vector (length 768) — an extra offset added to every query.
- $\mathbf{q}$ = the resulting query vector. Same story for keys ($W_k$) and values ($W_v$).

With bias, the map is [[affine-space|affine]]; without bias, it is a pure [[linear-mapping]].

### Derived numbers worth knowing
- **Head dimension:** $d_{\text{head}} = \dfrac{\texttt{emb\_dim}}{\texttt{n\_heads}} = \dfrac{768}{12} = 64$. `emb_dim` must be divisible by `n_heads` — `MultiHeadAttention` asserts `d_out % num_heads == 0` (§3.6, p. 86).
- **FFN width:** hard-coded as `4 * cfg["emb_dim"]` = 3,072 in `FeedForward` (§4.3, p. 107); it is not a config key.
- **Total parameters:** `GPTModel(GPT_CONFIG_124M)` has 163,009,536 parameters, 124,412,160 with [[weight-tying]] (§4.6, pp. 120–121). See [[gpt-2]] for the formula.

### How the config changes through the book

| Where | Change | Why |
|---|---|---|
| §5.1, p. 130 (pretraining); also §D.1, p. 313 | `context_length: 256` | Cheaper to compute, so training runs on a laptop |
| §5.5, p. 164 (load OpenAI weights) | `context_length: 1024`, `qkv_bias: True`, plus size from `model_configs` | Must match OpenAI's checkpoint shapes exactly |
| §6.4 p. 182, §7.5 p. 227, §E.3 p. 326 (fine-tuning) | `BASE_CONFIG = {vocab 50257, context 1024, drop_rate 0.0, qkv_bias True}` | Reuse the pretrained GPT-2; no dropout |
| Exercise 4.2 (§4.6, p. 122; solution appendix C, p. 302) | `emb_dim`, `n_layers`, `n_heads` → 1600 / 48 / 25 | Build GPT-2 XL with zero code changes |
| Exercise 4.3 (appendix C, pp. 302–304) | Replace `drop_rate` with `drop_rate_emb`, `drop_rate_attn`, `drop_rate_shortcut` | Different dropout per location |

The size presets the book uses (§5.5, p. 164):
```python
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs["gpt2-small (124M)"])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
```

Note the "bias" choice: the book disables `qkv_bias` by default because it
"follows the norms of modern LLMs" (p. 95), then turns it on only to match
OpenAI's GPT-2 weights, noting bias vectors are "not commonly used in LLMs
anymore" (p. 164). (The p. 95 text says this happens "in chapter 6"; in fact the
switch is flipped in §5.5 and reused in chapters 6–7.)

## 🔴 Theory (Expert)
- **Which keys change the parameter count?** `vocab_size`, `context_length`,
  `emb_dim`, `n_layers`, and `qkv_bias` do. **`n_heads` does not**: the Q/K/V
  projections are $d\times d$ regardless of how they are split into heads
  ([[weight-splitting-multi-head-attention]]). Heads change *how* the $d$
  dimensions are partitioned into $h$ subspaces of size $d/h$, not how many
  weights exist. `drop_rate` has no parameters at all.
- **Scaling.** Parameters grow as $\approx 12\,L\,d^2 + (V + T_{\max})\,d$ (plus another
  $V d$ for the book's untied output head), so
  `emb_dim` is quadratic and `n_layers` linear. Attention compute per layer grows
  as $\Theta(T^2 d)$ in the context length $T$ — the reason §5.1 cuts
  `context_length` to 256 (a 16× saving in the $T^2$ term, our arithmetic).
- **Compatibility constraints.** Loading pretrained weights is a strict shape
  match: the `assign` helper raises `ValueError` on any mismatch (p. 165). So the
  config is effectively part of the checkpoint — `context_length` sets the shape
  of `wpe` ($1024\times768$), `qkv_bias` decides whether `c_attn.b` has a home.
- **Dropout at inference.** `drop_rate` only acts in training mode;
  `model.eval()` disables it (see [[train-vs-eval-mode]]). Setting it to 0.0 for
  fine-tuning (§6.4, §7.5, §E.3) is a regularization choice, not an architecture
  change — the weights are identical either way.
- **Config vs. learned state.** This split mirrors the statistical distinction
  between model *class* (fixed by hyperparameters) and the *point* in that class
  (fixed by parameters fitted by [[maximum-likelihood-estimation]] / [[empirical-risk-minimization]]).

## Connections
- Builds on: [[hyperparameter]], [[vocabulary]], [[context-length]], [[embedding-dimension]], [[multi-head-attention]], [[dropout]]
- Leads to: [[gpt-2]], [[gpt-architecture]], [[transformer-block]], [[parameter-count-and-memory]], [[pretrained-weights]]
- Used by: [[llm-training-loop]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]]
- Related: [[bias-vector]], [[weight-tying]], [[model-parameters]], [[pytorch-nn-module]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 93–95 (`GPT_CONFIG_124M`, meaning of each key)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book p. 107 (FFN width $4\times$ `emb_dim`); §3.6, book p. 86 (divisibility assert)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 120–122 (parameter count, Exercise 4.2)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book p. 130 (context length 256); §5.5, book pp. 163–165 (`model_configs`, `NEW_CONFIG`, `assign`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.4, book pp. 181–182; §7.5, book p. 227 (`BASE_CONFIG`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1, book p. 313; §E.3, book pp. 326–327
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book pp. 302–304 (solutions to Exercises 4.2, 4.3)
