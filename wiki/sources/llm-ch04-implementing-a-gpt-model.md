---
title: "LLM Ch.4 — Implementing a GPT Model from Scratch to Generate Text"
type: source
tags: [llm, gpt, transformer, normalization, activation-functions, source]
created: 2026-10-13
updated: 2026-10-13
---

# LLM Ch.4 — Implementing a GPT Model from Scratch to Generate Text

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 4, **book pp. 92–127** (PDF pages 114–149)
- **Type:** book chapter — *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025)
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13 (every listing, printed tensor, parameter count and figure value checked in the staging notes)
- *Page rule:* PDF page = book page + 22.
- **Related:** references for this chapter are in Appendix B (book pp. 292–293); exercise solutions are in Appendix C (book pp. 302–304).

Ch. 3 built the attention "brain". This chapter builds the rest of the body around it. It works
top-down: first a placeholder `DummyGPTModel` that shows the whole data flow, then the real parts
one by one — [[layer-normalization]], the [[gelu]] activation, the [[feed-forward-network]],
[[shortcut-connections]] — then the [[transformer-block]], and finally the full 124M-parameter
[[gpt-2]] model. The chapter ends by making the (still untrained) model write text one token at a
time with [[greedy-decoding]]. The output is gibberish; Ch. 5 fixes that with training.

## Section coverage (all of Ch. 4 accounted for)
| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Chapter intro ("This chapter covers"; Fig. 4.1: three-stage roadmap, this is step 3 of stage 1) | 92–93 | ✅ [[llm-building-roadmap]], [[gpt-architecture]] |
| 4.1 | Coding an LLM architecture (GPT = generative pretrained transformer; [[model-parameters]] defined; GPT-2 vs GPT-3 sidebar; `GPT_CONFIG_124M`; Fig. 4.2–4.4; Listing 4.1 `DummyGPTModel`; tiktoken batch; logits of shape [2, 4, 50257]) | 93–99 | ✅ [[gpt-2]], [[gpt-3]], [[gpt-model-configuration]], [[model-parameters]], [[logits]], [[bias-vector]], [[token-embedding]], [[positional-embedding]], [[absolute-positional-embedding]], [[embedding-layer]], [[dropout]], [[byte-pair-encoding]], [[context-length]], [[embedding-dimension]], [[broadcasting]], [[pytorch-nn-module]], [[autoregressive-text-generation]] |
| 4.2 | Normalizing activations with layer normalization (motivation; Fig. 4.5–4.7; mean/var with `dim`, `keepdim`; Listing 4.2 `LayerNorm`; sidebars "Biased variance" and "Layer normalization vs. batch normalization") | 99–105 | ✅ [[layer-normalization]], [[vanishing-and-exploding-gradients]], [[standardization-z-score]], [[covariance-and-variance]], [[expected-value-and-mean]], [[bessels-correction-biased-variance]], [[batch-normalization]], [[relu]], [[activation-function]], [[linear-layer]], [[numerical-stability]], [[floating-point-precision]], [[random-seed-reproducibility]] |
| 4.3 | Implementing a feed forward network with GELU activations (ReLU vs GELU vs SwiGLU; exact GELU and tanh approximation, Listing 4.3; Fig. 4.8 plot; Listing 4.4 `FeedForward`; Fig. 4.9–4.11: 768 → 3072 → 768) | 105–109 | ✅ [[gelu]], [[relu]], [[swiglu]], [[activation-function]], [[gaussian-distribution]], [[cumulative-distribution-function]], [[feed-forward-network]], [[linear-mapping]], [[derivative]], [[sigmoid-function]] |
| 4.4 | Adding shortcut connections (residual/skip connections; Fig. 4.12; Listing 4.5 `ExampleDeepNeuralNetwork`; `print_gradients` with MSE loss; gradients with vs without shortcuts) | 109–113 | ✅ [[shortcut-connections]], [[vanishing-and-exploding-gradients]], [[mean-squared-error]], [[backpropagation]], [[automatic-differentiation]], [[chain-rule]], [[gradient]], [[loss-function]] |
| 4.5 | Connecting attention and linear layers in a transformer block (Fig. 4.13; Listing 4.6 `TransformerBlock`; Pre-LN vs Post-LN; shape preservation; Fig. 4.14) | 113–117 | ✅ [[transformer-block]], [[pre-layernorm-vs-post-layernorm]], [[multi-head-attention]], [[self-attention]], [[context-vector]], [[dropout]], [[regularization]], [[overfitting]] |
| 4.6 | Coding the GPT model (Fig. 4.15; Listing 4.7 `GPTModel`; 163,009,536 parameters; weight tying → 124,412,160; 621.83 MB; Exercises 4.1–4.2) | 117–122 | ✅ [[gpt-architecture]], [[decoder-only-transformer]], [[weight-tying]], [[parameter-count-and-memory]], [[logits]], [[gpt-2]], [[inner-product]], [[matrix-multiplication]] |
| 4.7 | Generating text (Fig. 4.16–4.18; softmax → argmax; Listing 4.8 `generate_text_simple`; context cropping; `.eval()`; gibberish output; Exercise 4.3) | 122–127 | ✅ [[autoregressive-text-generation]], [[next-token-prediction]], [[greedy-decoding]], [[argmax]], [[softmax]], [[monotonic-function]], [[probability-distribution]], [[decoding-strategies]], [[train-vs-eval-mode]], [[context-length]], [[tokenizer-encode-decode]] |
| — | Summary | 127 | — (recap; bullets folded into TL;DR below) |
| App. B | References for Ch. 4 | 292–293 | ✅ [[rmsnorm]], [[pre-layernorm-vs-post-layernorm]], [[gelu]] + person pages (see below) |
| App. C | Solutions to Exercises 4.1–4.3 | 302–304 | ✅ see "Exercises" below |

> ✅ **Every section of Ch. 4 (intro, 4.1–4.7, summary) maps to at least one wiki page.**
> None of §4.1–4.7 has numbered subsections (there is no 4.x.1 anywhere in this chapter), so the
> table lists the main topics inside each section instead.

## TL;DR (5 bullets)
- **A GPT model is a short recipe repeated many times:** token + positional embeddings → [[dropout]] → $L$ identical [[transformer-block|transformer blocks]] → final [[layer-normalization]] → a linear output head that gives one [[logits|logit]] per vocabulary token. For GPT-2 small, $L = 12$, $d = 768$, $V = 50{,}257$ ([[gpt-model-configuration]]).
- **[[layer-normalization]] keeps training stable:** each token vector is standardized to mean 0 and variance 1 over its features ([[standardization-z-score]]), then scaled and shifted by learned $\gamma, \beta$. It uses the *biased* variance ($1/n$, no [[bessels-correction-biased-variance|Bessel correction]]) to match GPT-2. Unlike [[batch-normalization]], it does not depend on the batch size.
- **The [[feed-forward-network]] expands then contracts:** $768 \to 3072 \to 768$ with a [[gelu]] in the middle. GELU $= x\,\Phi(x)$ uses the Gaussian [[cumulative-distribution-function]]; it is a smooth version of [[relu]] that still has a gradient for most negative inputs.
- **[[shortcut-connections]] stop the gradient from vanishing:** adding the input back, $x + f(x)$, gives the gradient a direct path. In the 5-layer demo the first-layer gradient goes from 0.0002 to 0.22 (about 1100× larger). The [[transformer-block]] uses the **Pre-LN** pattern $x + \text{Dropout}(\text{Sublayer}(\text{LN}(x)))$ twice: once for attention, once for the FFN.
- **Counting and generating:** the model has 163M parameters; with [[weight-tying]] (reuse the token-embedding matrix as the output head) it is 124M, the official GPT-2 size, and needs 621.83 MB in float32 ([[parameter-count-and-memory]]). Text comes out one token at a time: take the last row of logits → [[softmax]] → [[argmax]] → append ([[greedy-decoding]]). Untrained, it writes nonsense.

## Key concepts introduced
**Architecture overview (§4.1)**
- [[gpt-architecture]] — "generative pretrained transformer"; embeddings + stacked blocks + output head
- [[gpt-2]] — the reference model: 124M (first reported as 117M), 345M/355M, 762M/774M, 1,542M sizes
- [[gpt-3]] — same architecture, scaled to 175B parameters; weights not public
- [[gpt-model-configuration]] — the `GPT_CONFIG_124M` dictionary (vocab, context, emb_dim, heads, layers, dropout, qkv_bias)
- [[model-parameters]] — the trainable weights; a $2048 \times 2048$ matrix has 4,194,304 of them
- [[logits]] — raw output scores, one per vocabulary token, shape [batch, tokens, 50257]
- [[bias-vector]] — the $\mathbf{b}$ in $W\mathbf{x} + \mathbf{b}$; `qkv_bias=False` turns it off for the Q/K/V projections

**Normalization (§4.2)**
- [[vanishing-and-exploding-gradients]] — why deep networks are hard to train
- [[layer-normalization]] — per-token standardization + learned scale and shift
- [[standardization-z-score]] — $(x - \mu)/\sigma$
- [[covariance-and-variance]], [[expected-value-and-mean]] — the row statistics
- [[bessels-correction-biased-variance]] — $1/n$ vs $1/(n-1)$; negligible when $n = 768$
- [[batch-normalization]] — normalizes across the batch instead of across features
- [[relu]], [[activation-function]] — $\max(0, z)$, used in the 5→6 demo layer

**Feed forward and GELU (§4.3)**
- [[gelu]] — $x\,\Phi(x)$ and its tanh approximation used by GPT-2
- [[swiglu]] — sigmoid-gated alternative used in newer LLMs (named only)
- [[gaussian-distribution]], [[cumulative-distribution-function]] — where $\Phi$ comes from
- [[feed-forward-network]] — position-wise $W_2\,\text{GELU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$

**Shortcut connections (§4.4)**
- [[shortcut-connections]] — residual / skip connections, from ResNets in computer vision
- [[mean-squared-error]] — the toy loss in `print_gradients`
- [[chain-rule]], [[backpropagation]], [[automatic-differentiation]] — `loss.backward()` and why the identity term helps

**Transformer block and the full model (§4.5–4.6)**
- [[transformer-block]] — LN → masked MHA → dropout → add; LN → FFN → dropout → add
- [[pre-layernorm-vs-post-layernorm]] — GPT-2 normalizes *before* each sub-layer; the original transformer did it *after*
- [[weight-tying]] — share the token-embedding matrix with the output head
- [[parameter-count-and-memory]] — `numel()` sums and bytes = params × 4

**Generation (§4.7)**
- [[autoregressive-text-generation]] — predict one token, append it, repeat
- [[greedy-decoding]] — always take the most likely token
- [[argmax]] — the index of the largest value = the token ID
- [[monotonic-function]] — softmax keeps the order, so argmax(softmax(z)) = argmax(z)
- [[train-vs-eval-mode]] — `model.eval()` turns dropout off for inference

**Further reading (Appendix B)**
- [[rmsnorm]] — LayerNorm without the mean subtraction; cheaper, used by modern LLMs

## Notable results / numbers
- **`GPT_CONFIG_124M`** (book p. 95): `vocab_size` 50,257 · `context_length` 1,024 · `emb_dim` 768 · `n_heads` 12 · `n_layers` 12 · `drop_rate` 0.1 · `qkv_bias` False.
- **GPT-3 training cost** (Lambda Labs, book p. 94): 355 years on one V100 GPU; 665 years on one RTX 8000.
- **Dummy model output** (book p. 99): input token IDs `[[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]` ("Every effort moves you", "Every day holds a") → logits `torch.Size([2, 4, 50257])`.
- **LayerNorm demo** (book pp. 100–104): a Linear(5→6) + ReLU layer gives row means 0.1324 and 0.2170 and variances 0.0231 and 0.0398. After normalizing, means are about $-5.96 \times 10^{-8}$ (floating-point noise, not exactly 0) and variances 1. The `LayerNorm` class gives variances 0.9995 and 0.9997 — slightly under 1 because $\varepsilon = 10^{-5}$ sits in the denominator.
  $$\hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad y_i = \gamma_i \hat x_i + \beta_i, \qquad \sigma^2 = \tfrac{1}{n}\textstyle\sum_i (x_i - \mu)^2$$
- **GELU** (book p. 105):
  $$\text{GELU}(x) = x\,\Phi(x) \approx 0.5\,x\left(1 + \tanh\!\left[\sqrt{2/\pi}\,\big(x + 0.044715\,x^3\big)\right]\right)$$
  The curve has a shallow minimum (about $-0.17$) near $x \approx -0.75$, the only place where its slope is 0 (book p. 106).
- **FFN shapes** (book pp. 107–108): (2, 3, 768) → (2, 3, 3072) → (2, 3, 3072) → (2, 3, 768). Expansion factor 4.
- **Shortcut demo** (book pp. 112–113), mean absolute gradient per layer, `layer_sizes = [3, 3, 3, 3, 3, 1]`, seed 123:

  | Layer | Without shortcut | With shortcut |
  |---|---|---|
  | layers.0 | 0.000202 | 0.2217 |
  | layers.1 | 0.000120 | 0.2069 |
  | layers.2 | 0.000715 | 0.3290 |
  | layers.3 | 0.001399 | 0.2666 |
  | layers.4 | 0.005050 | 1.3259 |

  (Wiki explanation, not in the book: with a residual, $\partial \mathbf{x}_{\ell+1}/\partial \mathbf{x}_\ell = I + \partial f_\ell / \partial \mathbf{x}_\ell$, so the backward signal always has an identity path.)
- **Transformer block** (book pp. 115–116): input and output are both `torch.Size([2, 4, 768])`. The shape stays the same, but each output row is a [[context-vector]] that now holds information from the whole sequence.
  $$h = x + \text{Dropout}\big(\text{MHA}(\text{LN}_1(x))\big), \qquad y = h + \text{Dropout}\big(\text{FFN}(\text{LN}_2(h))\big)$$
- **Parameter count** (book pp. 120–122): total **163,009,536**. Token embedding and output head are each $50{,}257 \times 768 = 38{,}597{,}376$. Removing the head (weight tying) gives **124,412,160**. Memory: $163{,}009{,}536 \times 4 / 1024^2 =$ **621.83 MB**.
- **Generation** (book pp. 124–127): "Hello, I am" = `[15496, 11, 314, 716]`. Figure 4.18 shows a trained-looking run ("a" 257, "model" 2746, "ready" 3492, … "Hello, I am a model ready to help."). The actual untrained run gives `[15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267]` → **"Hello, I am Featureiman Byeswickattribute argue"**.
- **Chapter sizes** (summary, book p. 127): GPT models of 124, 345, 762 and 1,542 million parameters all come from the same `GPTModel` class.

> ⚠️ **Book inconsistencies (flagged in the staging notes):**
> - Fig. 4.5 (p. 100) says the pre-normalization variance is **0.39**, but the code for the same row prints **0.0231**. The code is right (the normalized values match 0.0231).
> - The "Biased variance" sidebar (p. 103) says the pretrained weights are loaded in "chapter 6". They are actually loaded in **§5.5** (book p. 161); Ch. 6 reuses them. The same applies to the weight-tying remark on p. 121.
> - p. 106 refers to "figure 4.18, right" for ReLU's corner; it means **Fig. 4.8**.
> - Fig. 4.17 labels the last vocabulary index 50257; the real last index is **50256** (indices start at 0).
> - Appendix B gives the GPT-3 paper the year 2023 (correct: 2020) and spells the GELU author "Hendricks" (correct: Hendrycks).
> - Appendix C's Exercise 4.3 solution ends with `return logitss` (typo for `logits`).

## Exercises
| # | Book p. | Task | Solution idea (Appendix C, pp. 302–304) |
|---|---|---|---|
| 4.1 | 121 | Count and compare the parameters in the feed forward module and the multi-head attention module of one transformer block. | `sum(p.numel() for p in block.ff.parameters())` → **4,722,432**; same for `block.att` → **2,360,064**. The FFN has about **2×** the parameters of attention. Check: FFN $= (768 \cdot 3072 + 3072) + (3072 \cdot 768 + 768)$; attention $= 3 \cdot 768^2$ (Q, K, V, no bias) $+ (768^2 + 768)$ (output projection). See [[parameter-count-and-memory]], [[feed-forward-network]], [[multi-head-attention]]. |
| 4.2 | 122 | Build GPT-2 medium (1,024-dim, 24 blocks, 16 heads), large (1,280, 36, 20) and XL (1,600, 48, 25) by changing only the config; bonus: count their parameters. | Copy `GPT_CONFIG_124M` and change `emb_dim`, `n_layers`, `n_heads`. For XL: total **1,637,792,000** parameters; with weight tying **1,557,380,800** (minus $50{,}257 \times 1{,}600$); size **6,247.68 MB**. See [[gpt-model-configuration]], [[gpt-2]], [[weight-tying]]. |
| 4.3 | 127 | Use a separate dropout rate for each kind of dropout layer instead of one global `drop_rate`. | Replace `drop_rate` with `drop_rate_emb`, `drop_rate_shortcut`, `drop_rate_attn` (all 0.1). Use `drop_rate_attn` in `MultiHeadAttention`, `drop_rate_shortcut` for `drop_shortcut` in `TransformerBlock`, and `drop_rate_emb` for `drop_emb` in `GPTModel`. See [[dropout]], [[transformer-block]]. |

## References cited (Appendix B, Ch. 4 group, book pp. 292–293)
1. [[jimmy-lei-ba]], Jamie Ryan Kiros, [[geoffrey-hinton]] (2016), "Layer Normalization", arXiv 1607.06450 — [[layer-normalization]]; stabilizes hidden-state dynamics and cuts training time.
2. Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture", arXiv 2002.04745 — [[pre-layernorm-vs-post-layernorm]]; Pre-LN can train more stably.
3. Tie et al. (2023), "ResiDual: Transformer with Dual Residual Connections", arXiv 2304.14802 — a second view on the Pre-LN vs Post-LN question.
4. [[biao-zhang]] and [[rico-sennrich]] (2019), "Root Mean Square Layer Normalization", arXiv 1910.07467 — [[rmsnorm]]; normalizes by the root mean square only, no mean subtraction.
5. [[dan-hendrycks]] and Kevin Gimpel (2016), "Gaussian Error Linear Units (GELUs)", arXiv 1606.08415 — [[gelu]]; combines ReLU-like gating with the normal [[cumulative-distribution-function]].
6. [[alec-radford]] et al. (2019), "Language Models Are Unsupervised Multitask Learners" — the [[gpt-2]] paper (124M, 355M, 774M, 1.5B sizes).
7. [[tom-brown]] et al. (2020; book prints 2023), "Language Models are Few-Shot Learners", arXiv 2005.14165 — [[gpt-3]]; same architecture as GPT-2, 175B parameters, 100× the largest GPT-2.
8. Lambda Labs, "OpenAI's GPT-3 Language Model: A Technical Overview" — the 665-year single-GPU training estimate.
9. [[andrej-karpathy]], "NanoGPT" (github.com/karpathy/nanoGPT) — minimal GPT-2 code that inspired the book's split into small submodules.
10. Harm de Vries, "In the Long (Context) Run" — below about 32,000 tokens of context, most LLM compute is in the [[feed-forward-network]] layers, not in attention.

## People
- [[jimmy-lei-ba]] and [[geoffrey-hinton]] — layer normalization (2016).
- [[dan-hendrycks]] — GELU (2016).
- [[biao-zhang]] and [[rico-sennrich]] — RMSNorm (2019).
- [[friedrich-wilhelm-bessel]] — named through Bessel's correction ($n - 1$) in the "Biased variance" sidebar.
- [[alec-radford]] (GPT-2), [[tom-brown]] (GPT-3), [[andrej-karpathy]] (nanoGPT).
- [[sebastian-raschka]] — the author's own advice: separate (untied) embedding and output layers train better (p. 121).
- The chapter body names almost no people; most names come from Appendix B. The ResNet origin of shortcut connections (He et al., 2015) is mentioned without a citation.

## What this changes in the wiki
- **Completes the GPT architecture.** With Ch. 3's [[multi-head-attention]], the wiki now has
  every part of a working GPT-2: [[layer-normalization]], [[gelu]], [[feed-forward-network]],
  [[shortcut-connections]], [[transformer-block]], [[weight-tying]], and the generation loop
  ([[autoregressive-text-generation]], [[greedy-decoding]]). About 25 concept pages plus person
  pages for [[jimmy-lei-ba]], [[dan-hendrycks]], [[biao-zhang]] and [[friedrich-wilhelm-bessel]].
- **Strong links back to the MML pages.** LayerNorm is plain statistics: [[expected-value-and-mean]]
  and [[covariance-and-variance]] (now with a biased-vs-unbiased estimator use case through
  [[bessels-correction-biased-variance]]). GELU is built on the [[gaussian-distribution]] CDF.
  Shortcut connections are an application of the [[chain-rule]] and [[backpropagation]].
  `nn.Linear` is a [[linear-mapping]] plus a [[bias-vector]] (an affine map, see [[affine-space]]).
  With tied weights, each logit is an [[inner-product]] between the final hidden vector and a
  token's embedding row.
- **New ML-practice pages** that MML does not cover: [[batch-normalization]], [[rmsnorm]],
  [[pre-layernorm-vs-post-layernorm]], [[swiglu]], [[parameter-count-and-memory]],
  [[train-vs-eval-mode]], [[monotonic-function]] (as used for argmax/softmax).
- **No contradictions** with existing wiki pages. The book-internal inconsistencies (Fig. 4.5
  variance, "chapter 6" weight-loading references, Fig. 4.8/4.18 typo, index 50257) are flagged
  above.
- **Next:** Ch. 5 trains this model — [[cross-entropy-loss]], [[perplexity]],
  [[training-and-validation-loss]], [[adamw-optimizer]] — and replaces greedy decoding with
  [[temperature-scaling]] and [[top-k-sampling]] (see [[decoding-strategies]]).
