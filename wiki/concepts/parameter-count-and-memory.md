---
title: Parameter Count and Memory Footprint
type: concept
tags: [deep-learning, parameters, memory, gpt, llm, pytorch]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Parameter Count and Memory Footprint

> **One-liner:** A model's "size" is just how many learnable numbers it has; multiply that count by the bytes each number takes and you get how much memory the model needs.

*Merged here:* model parameter count; model parameter count and memory.

## 🟢 Intuition (Dummy)
A neural network is a giant machine full of tiny dials. Each dial is one **parameter**
— one number the model learns during training ([[model-parameters]]). When people say
"GPT-2 124M", they mean "about 124 million dials".

Counting them is like counting the squares on graph paper: a sheet that's 768 squares
wide and 3,072 tall has $768\times3{,}072$ squares. Every layer is a few such sheets, so
you multiply the sides and add everything up.

Memory is the next step. Each dial is stored as a number in the computer, and a normal
32-bit number takes **4 bytes**. So 163 million dials × 4 bytes ≈ 650 million bytes —
the book's answer is **621.83 MB** for its GPT model (book p. 122). Even a "small" LLM
needs a lot of room.

## 🟡 Mechanics (Practitioner)

### Rule 1 — counting one layer
A [[linear-layer]] `nn.Linear(d_in, d_out)` holds a weight [[matrix]] and a [[bias-vector]]:
$$\#\text{params} = d_{\text{in}}\cdot d_{\text{out}} + d_{\text{out}}$$
- $d_{\text{in}}$ = how many numbers go in; $d_{\text{out}}$ = how many come out.
- $d_{\text{in}}\cdot d_{\text{out}}$ = entries of the weight matrix $W$ (shape $d_{\text{out}}\times d_{\text{in}}$ in PyTorch).
- $+\,d_{\text{out}}$ = one bias per output (drop this term when `bias=False`).

An `nn.Embedding(V, d)` is a $V\times d$ table: $V\cdot d$ parameters ([[embedding-layer]]).
A [[layer-normalization]] over $d$ features has $2d$ (a `scale` and a `shift` per feature).
Activation functions ([[relu]], [[gelu]]) and [[dropout]] have **zero** parameters.

**Small check from appendix A.** A network with layers 50→30→20→3 has
$(50\cdot30+30)+(30\cdot20+20)+(20\cdot3+3)=1530+620+63=2213$ parameters (book p. 267).
Exercise A.3, a 2→30→20→2 network, has $(2\cdot30+30)+(30\cdot20+20)+(20\cdot2+2)=90+620+42=\mathbf{752}$
(book p. 275; solution p. 311).

### Rule 2 — let PyTorch count
```python
total_params = sum(p.numel() for p in model.parameters())
# trainable only:
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
- `model.parameters()` walks over every parameter tensor; `.numel()` = "number of elements" in a tensor (book pp. 120, 267).
- The `requires_grad` filter counts only **trainable** parameters — important once you freeze layers ([[layer-freezing]]) or add LoRA ([[low-rank-adaptation]]).

### The GPT-2 124M count (§4.6, book pp. 120–121)
```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")   # 163,009,536
```
Why 163M and not 124M? Because of [[weight-tying]]: original GPT-2 reuses the token
embedding as its output layer, so the head should not be counted twice:
```python
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# Number of trainable parameters considering weight tying: 124,412,160
```

**Where the 163,009,536 come from** (our breakdown; it reproduces the book's total exactly).
`GPT_CONFIG_124M`: $V=50{,}257$, context $C=1{,}024$, $d=768$, $L=12$ blocks, `qkv_bias=False`.

| Part | Formula | Count |
|---|---|---|
| Token embedding `tok_emb` | $V d$ | 38,597,376 |
| Positional embedding `pos_emb` | $C d$ | 786,432 |
| One block: attention (Q, K, V without bias + `out_proj` with bias) | $3d^2 + (d^2+d)$ | 2,360,064 |
| One block: feed-forward ($d\to4d\to d$) | $(4d^2+4d)+(4d^2+d)$ | 4,722,432 |
| One block: two LayerNorms | $2\cdot2d$ | 3,072 |
| **One block total** | | **7,085,568** |
| 12 blocks | $\times12$ | 85,026,816 |
| Final LayerNorm | $2d$ | 1,536 |
| Output head `out_head` (no bias) | $V d$ | 38,597,376 |
| **Total** | | **163,009,536** |
| Total minus head (tied) | | **124,412,160** |

The attention and feed-forward numbers are exactly the answer to Exercise 4.1:
feed-forward 4,722,432 vs. attention 2,360,064 — the feed-forward part has **about twice**
the parameters (book p. 121; solution p. 302). Also striking: the two vocabulary-sized
tables are ≈ 47% of the untied model.

**Cross-checks elsewhere in the book (our arithmetic).**
- Loading OpenAI's weights needs `qkv_bias=True` (book p. 164), adding $3d$ biases per block: $124{,}412{,}160 + 12\cdot3\cdot768 = 124{,}439{,}808$.
- In appendix E the output head is swapped for a 2-class head ($768\cdot2+2=1{,}538$): $124{,}439{,}808+1{,}538 = 124{,}441{,}346$ — exactly the "trainable parameters before freezing" printed on book p. 332.

### Memory footprint (§4.6, book p. 122)
```python
total_size_bytes = total_params * 4            # assumes float32
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")   # 621.83 MB
```
$$\text{size}_{\text{MB}} = \frac{N_{\text{params}}\times b}{1024^2}$$
- $N_{\text{params}}$ = number of parameters (163,009,536).
- $b$ = bytes per parameter; a 32-bit float ([[floating-point-precision]]) is $b=4$.
- $1024^2 = 1{,}048{,}576$ bytes per MB (strictly, a mebibyte, MiB).
- $163{,}009{,}536\times4/1{,}048{,}576 \approx \mathbf{621.83\ MB}$.

The book's conclusion: the result illustrates "the relatively large storage capacity
required to accommodate even relatively small LLMs" (book p. 122). Consistency check (ours): the downloaded OpenAI checkpoint
`model.ckpt.data-00000-of-00001` is 498M (book p. 162) ≈ $124{,}439{,}808\times4$ bytes = 497.8 million bytes — the tied model in float32.

### The GPT-2 family
| Model | $d$ | Blocks $L$ | Heads | Name | Untied count / size (fp32) |
|---|---|---|---|---|---|
| gpt2-small | 768 | 12 | 12 | 124M | 163,009,536 / 621.83 MB |
| gpt2-medium | 1,024 | 24 | 16 | 355M | 406,212,608 / 1,549.58 MB* |
| gpt2-large | 1,280 | 36 | 20 | 774M | 838,220,800 / 3,197.56 MB* |
| gpt2-xl | 1,600 | 48 | 25 | 1558M | **1,637,792,000 / 6,247.68 MB** |

Configs from §5.5 (Fig. 5.17, book p. 163; `model_configs`, book p. 164) and Exercise 4.2 (book p. 122). The XL numbers are the
book's answer to Exercise 4.2: total **1,637,792,000**, with weight tying
**1,557,380,800**, size **6,247.68 MB** (appendix C, book p. 302). *Medium/large rows
are our arithmetic with the same code (`qkv_bias=False`). The names (124M, 355M, 774M,
1558M) are the tied counts with QKV biases, e.g. 354,823,168 and 774,030,080.

## 🔴 Theory (Expert)
- **Closed form.** For a GPT with width $d$, depth $L$, vocabulary $V$, context $C$, FFN
  width $4d$:
  $$N = \underbrace{L\,(12d^2 + 10d)}_{\text{blocks}} + \underbrace{Vd + Cd}_{\text{embeddings}} + 2d \;\;(+\,Vd \text{ if untied})$$
  Per block: attention $4d^2+d$, FFN $8d^2+5d$, two LNs $4d$ (with `qkv_bias=False`).
  For large models the $12Ld^2$ term dominates — so parameters grow **linearly with depth**
  and **quadratically with width**. GPT-2 small: $12\cdot12\cdot768^2 = 84.9$M of its
  85.0M block parameters.
- **Where the compute goes.** FFN holds $\tfrac{8}{12}=\tfrac23$ of block parameters, matching
  Exercise 4.1's 2:1 ratio. Appendix B cites de Vries: below ~32,000 tokens of context
  most compute is in the feed-forward layers, not attention (book p. 293). A forward pass
  costs roughly $2N$ FLOPs per token (external rule of thumb).
- **Precision.** Memory scales with bytes per number: fp32 = 4, bf16/fp16 = 2, int8 = 1,
  4-bit ≈ 0.5. GPT-2 small (tied) is ≈ 475 MiB in fp32 and ≈ 237 MiB in bf16 (our
  arithmetic; the book only uses fp32 here).
- **Training needs more than the weights.** With [[adamw-optimizer|AdamW]] in fp32 you
  store weights, gradients and two moment estimates: ≈ 16 bytes/parameter, ≈ 2 GB for
  124M — before activations, which grow with batch size × context length × $d$ × $L$
  (external, standard estimate). The book makes the related point that AdamW "store[s] additional parameters for each model
  weight", so it saves and restores the optimizer state together with the model (§5.4,
  book p. 160; Exercise 5.4 solution, book p. 305). This is also why
  [[parameter-efficient-fine-tuning]] (LoRA: 2,666,528 trainable vs. 124,441,346, book
  p. 332) and [[gradient-accumulation]] matter.
- **Tied vs untied.** Tying removes $Vd$ parameters (23.7% of small, 4.9% of XL — our
  arithmetic); see [[weight-tying]].
- **Capacity.** Parameter count is a crude proxy for [[model-capacity]]; with too many
  parameters for too little data you risk [[overfitting]].

### From llm-ch03-coding-attention-mechanisms
Exercise 3.3 (book p. 91) asks for a `MultiHeadAttention` module the size of the smallest GPT-2: `MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0.0, num_heads=12)` (solution: Appendix C, book p. 301). Its parameter count (wiki arithmetic, with `qkv_bias=False`):
$$3 \cdot 768^2 + (768^2 + 768) = 1{,}769{,}472 + 590{,}592 = 2{,}360{,}064 .$$
Read it as: "three weight matrices $W_q, W_k, W_v$, each $768 \times 768$, plus the [[output-projection]] (a $768 \times 768$ weight and a bias of 768)". The causal mask is a buffer, not a parameter, so it adds nothing to the count. The book states the GPT-2 sizes: smallest = 117M parameters, 12 heads, 768 dimensions; largest = 1.5B parameters, 25 heads, 1,600 dimensions (book pp. 90–91).

## Connections
- Builds on: [[model-parameters]], [[linear-layer]], [[matrix]], [[embedding-layer]], [[layer-normalization]], [[pytorch-nn-module]]
- Leads to: [[weight-tying]], [[gpt-model-configuration]], [[floating-point-precision]], [[parameter-efficient-fine-tuning]], [[low-rank-adaptation]]
- Related: [[gpt-2]], [[transformer-block]], [[feed-forward-network]], [[multi-head-attention]], [[model-capacity]], [[gpu-computing]], [[layer-freezing]]
- Used by: [[gpt-architecture]], [[pretrained-weights]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 117–122 (163,009,536; 124,412,160; 621.83 MB; Exercises 4.1 and 4.2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 162–164 (498M checkpoint file; GPT-2 size table, Fig. 5.17; `qkv_bias=True`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book p. 267 (2,213 parameters, `numel`); §A.7, book p. 275 (Exercise A.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.4, book p. 160 (AdamW stores extra state per weight).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book pp. 302, 305, 311 (Ex. 4.1, 4.2, 5.4, A.3 solutions).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book p. 332 (124,441,346 and 2,666,528 trainable parameters); appendix B, book p. 293 (de Vries).
- Breakdown table, closed form, medium/large rows, precision and optimizer-memory estimates: our arithmetic / standard results, not printed in the book.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6.2 Exercise 3.3 and GPT-2 sizes, book pp. 90–91; Appendix C, book p. 301 (count is wiki arithmetic).
