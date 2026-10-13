---
title: GPT-2 Model Family
type: concept
tags: [llm, transformer, gpt, pretrained-model, model-architecture]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# GPT-2 Model Family

> **One-liner:** GPT-2 is OpenAI's 2019 family of four decoder-only transformer language models (124M to 1.5B parameters) whose weights are public — so it is the model the book rebuilds from scratch, trains, and fine-tunes.

*Also covers:* GPT-2 model sizes (small / medium / large / XL).

## 🟢 Intuition (Dummy)
Picture one LEGO design that comes in four box sizes. The instructions are the
same; the bigger boxes just have **wider** bricks (bigger vectors per word) and
**taller** towers (more repeated layers). That's GPT-2: one recipe, four sizes.

Why does the book use this "old" model instead of something newer?
1. **The weights are free.** OpenAI published the trained GPT-2 weights, so you
   can load them and skip a training run that would cost "tens to hundreds of
   thousands of dollars" (§5.5, p. 161).
2. **It fits on a laptop.** GPT-3 has the same architecture but 175 billion
   parameters; training it would take an estimated 355 years on one V100 GPU
   (p. 94). GPT-2 small runs on a single laptop.
3. **It's the blueprint for modern LLMs.** GPT-3 is "fundamentally the same
   architecture", just scaled up (p. 94).

> ⚠️ Book cross-reference error: the GPT-2 vs. GPT-3 sidebar (p. 94) and §4.1 (p. 93)
> say the pretrained weights are loaded "in chapter 6"; they are actually loaded in
> §5.5 (pp. 161–166), and chapter 6 reuses them.

## 🟡 Mechanics (Practitioner)

### The four sizes (§5.5, Figure 5.17, p. 163)

| Name | Parameters | Transformer blocks $L$ | Heads $h$ | Embedding dim $d$ | Head dim $d/h$ |
|---|---|---|---|---|---|
| gpt2-small | 124M | 12 | 12 | 768 | 64 |
| gpt2-medium | 355M | 24 | 16 | 1,024 | 64 |
| gpt2-large | 774M | 36 | 20 | 1,280 | 64 |
| gpt2-xl | 1,558M | 48 | 25 | 1,600 | 64 |

(The head-dim column is our arithmetic: every size keeps 64 dimensions per head.)

Shared by all four: [[vocabulary]] of **50,257** [[byte-pair-encoding|BPE]] tokens,
[[context-length]] of **1,024** tokens, learned [[absolute-positional-embedding|absolute positional embeddings]],
[[gelu|GELU]] activations, Pre-[[layer-normalization|LayerNorm]], bias vectors in the
query/key/value layers, and [[weight-tying]] between the token embedding and the
output layer (pp. 163–166).

> ⚠️ Contradiction (inside the book): Chapter 4 (§4.1, p. 93, and the chapter
> summary, p. 127) gives the larger sizes as **345, 762, and 1,542** million
> parameters. Chapter 5 (§5.5, p. 163), the download code, and appendix B (p. 293)
> say **355M, 774M, 1,558M** (appendix B: "1.5B"). The checkpoint names OpenAI
> ships are `355M`, `774M`, `1558M`, so we use those. The book also notes the
> original GPT-2 report said **117M** for the smallest model, "later corrected" to
> 124M (p. 93); §3.6 (p. 90) still quotes 117M.

### Where the architecture comes from
In the book's code, GPT-2 is `GPTModel` (Listing 4.7): token + positional
embeddings → dropout → $L$ × [[transformer-block]] → final LayerNorm → linear
output head producing [[logits]]. Data flow as math:

$$
X^{(0)} = E_{\text{tok}}[\text{ids}] + E_{\text{pos}}[0{:}T],\qquad
X^{(\ell)} = \text{Block}_\ell\big(X^{(\ell-1)}\big),\qquad
\text{logits} = \text{LN}\big(X^{(L)}\big)\,W_{\text{out}}^{\top}
$$

- $\text{ids}$ = the input [[token-id|token IDs]]; $T$ = how many tokens there are.
- $E_{\text{tok}} \in \mathbb{R}^{50257\times d}$ = token-embedding [[matrix]]; $E[\text{ids}]$ = "take those rows".
- $E_{\text{pos}} \in \mathbb{R}^{1024\times d}$ = positional-embedding matrix; $E_{\text{pos}}[0{:}T]$ = its first $T$ rows.
- $\text{Block}_\ell$ = the $\ell$-th transformer block; $L$ = 12 for small.
- $\text{LN}$ = the final [[layer-normalization]].
- $W_{\text{out}} \in \mathbb{R}^{50257\times d}$ = the output head; in GPT-2 it is *the same matrix* as $E_{\text{tok}}$ (weight tying).

### Loading the real weights (§5.5, pp. 161–166)
```python
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
# settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
# params keys: ['blocks', 'b', 'g', 'wpe', 'wte'];  params["wte"].shape == (50257, 768)

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs["gpt2-small (124M)"])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
load_weights_into_gpt(gpt, params)
```
Two settings must change to match OpenAI's checkpoint (p. 164):
- `context_length` → **1024** (the book had trained with 256), so the positional matrix is $1024\times768$ like `wpe`.
- `qkv_bias` → **True**: OpenAI used bias vectors in the Q/K/V layers, although they are "not commonly used in LLMs anymore".

OpenAI stores Q, K, V fused in one `c_attn` weight (and bias); the loader splits it
into three equal parts with `np.split(..., 3, axis=-1)` and transposes the weights
for PyTorch (Listing 5.5, p. 165). For GPT-2 small the fused matrix must therefore be
$768\times 2304$, giving three $768\times768$ pieces (our inference — the book does not
print this shape). With the weights loaded, the prompt
"Every effort moves you" (top-k 50, temperature 1.5) gives coherent text (p. 167).

### How the book uses each size
- **124M (small):** built and pretrained in Ch. 4–5; classification fine-tuning in Ch. 6 (§6.4, p. 182); LoRA in appendix E.
- **355M (medium):** instruction fine-tuning in Ch. 7, because the 124M model "is too limited in capacity" to give good results there; the download is ~**1.42 GB**, about 3× the small model (§7.5, pp. 226–227).
- **1,558M (XL):** Exercise 5.6 — try the largest one and compare its text (p. 167).

Fine-tuning time in Ch. 7 (2 epochs, §7.6, pp. 230–231): medium takes 15.78 min on an
M3 MacBook Air CPU, 1.83 min on an NVIDIA L4, 0.86 min on an A100; small takes
5.74 / 0.69 / 0.39 min.

## 🔴 Theory (Expert)

### Exact parameter count
For the book's `GPTModel` with `qkv_bias=False`, separate (untied) output head,
and $d$ = emb_dim, $L$ = blocks, $V$ = 50,257, $T_{\max}$ = 1,024:

$$
N = \underbrace{L\big[(4d^2 + d) + (8d^2 + 5d) + 4d\big]}_{\text{attention + FFN + 2 LayerNorms}}
+ \underbrace{Vd}_{\text{tok emb}} + \underbrace{T_{\max}d}_{\text{pos emb}} + \underbrace{2d}_{\text{final LN}} + \underbrace{Vd}_{\text{out head}}
$$

- Attention: three $d\times d$ Q/K/V matrices + output projection $d\times d$ with bias $d$ → $4d^2+d$.
- [[feed-forward-network|FFN]]: $d\to4d\to d$ with biases → $8d^2 + 4d + d$.
- Each LayerNorm: scale + shift → $2d$; two per block.

Check for small ($d=768$, $L=12$): per block $2{,}360{,}064 + 4{,}722{,}432 + 3{,}072 = 7{,}085{,}568$;
$\times 12 = 85{,}026{,}816$; plus $38{,}597{,}376 + 786{,}432 + 1{,}536 + 38{,}597{,}376$
gives **163,009,536** — exactly the book's printed total (p. 120). Removing the
duplicate output head (weight tying) gives **124,412,160** (p. 121). For XL
($d=1600$, $L=48$) the same formula gives **1,637,792,000**, and 1,557,380,800 with
tying — the appendix C solution to Exercise 4.2 (p. 302). The block term
$\approx 12Ld^2$ dominates as models grow; the embedding terms are a big share for
small models (about 32% of GPT-2 small's 124M tied parameters, our arithmetic).

### Memory
At 4 bytes per float32 parameter, $163{,}009{,}536 \times 4 / 1024^2 \approx$
**621.83 MB** for the untied small model (p. 122); XL is 6,247.68 MB (appendix C).
See [[parameter-count-and-memory]] and [[floating-point-precision]].

### Scaling pattern
Across the family, depth $L$ and width $d$ grow together while head size stays 64:
$h = d/64$. Since the per-block cost is $\Theta(d^2)$ in parameters and the
attention cost is $\Theta(T^2 d)$ in compute, going from small to XL multiplies
block parameters by $\tfrac{48\cdot1600^2}{12\cdot768^2}\approx 17$.

### Design choices later LLMs changed
The book points out several GPT-2 choices that modern models dropped: Q/K/V bias
vectors (p. 164), weight tying (the author found separate layers train better,
p. 121), and LayerNorm (many models now use [[rmsnorm|RMSNorm]], appendix B).

## Connections
- Builds on: [[transformer-architecture]], [[decoder-only-transformer]], [[gpt-architecture]], [[transformer-block]], [[multi-head-attention]], [[byte-pair-encoding]]
- Configured by: [[gpt-model-configuration]], [[hyperparameter]]
- Leads to: [[gpt-3]], [[pretrained-weights]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]]
- Uses: [[layer-normalization]], [[gelu]], [[weight-tying]], [[logits]], [[absolute-positional-embedding]]
- Related: [[parameter-count-and-memory]], [[model-capacity]], [[large-language-model]]
- People: [[alec-radford]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6, book pp. 90–91 (GPT-2 head counts, Exercise 3.3)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 93–95 (sizes, GPT-2 vs GPT-3 sidebar, config)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 117–122 (Listing 4.7, 163M/124M counts, 621.83 MB, Exercise 4.2)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7 / Chapter 4 summary, book p. 127
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 160–167 (Figure 5.17, loading OpenAI weights)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.4, book pp. 181–182; §7.5, book pp. 226–228; §7.6, book pp. 230–231
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 292–293 (Radford et al. 2019); appendix C, book pp. 301–302, 306 (solutions 3.3, 4.2, 5.6)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.4, book p. 320 (GPT-2 tokenizer + 124M config in the full training loop)
