---
title: Transformer Architecture
type: concept
tags: [llm, deep-learning, attention, neural-architecture, nlp]
difficulty: intermediate
aliases: [transformer, transformers, the transformer]
sources: [llm-ch01-understanding-large-language-models, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Transformer Architecture

> **One-liner:** The transformer is a neural-network design where every word gets to "look at" every other word (attention) to decide what it means in context — no reading word-by-word in a chain.

## 🟢 Intuition (Dummy)
Think about the word "bank". In "river bank" and "bank account" it means
different things. To understand a word, you glance at the *other* words around
it. A transformer does exactly that, for every word at once: each word asks
"which of the other words matter for me?" and mixes in information from them.
That glancing is called **attention**.

Older models ([[recurrent-neural-network|RNNs]]) read a sentence one word at a
time and had to squeeze the *whole* sentence into one small memory before
translating it. Long sentences overflowed that memory and details got lost. The
transformer threw out that chain: every word can look directly at every other
word, however far away.

The original transformer (2017) was built for translation and has two halves:
- an **encoder** that reads the whole input sentence and turns it into vectors
  full of context;
- a **decoder** that writes the translation **one word at a time**, using those
  vectors.

Book example (Fig. 1.4): input "This is an example", partial output
"Das ist ein" → the decoder produces the last word, "Beispiel".

Later models kept just one half: [[bert]] keeps the encoder (great at
understanding/classifying), GPT keeps the decoder (great at writing). Today's
chat LLMs are GPT-style, decoder-only transformers.

## 🟡 Mechanics (Practitioner)

### The big picture
Introduced in "Attention Is All You Need" (2017, arXiv 1706.03762) for
English→German and English→French [[machine-translation]] (§1.4, p. 7).

| Part | What it does (book wording) |
|---|---|
| **Encoder** | processes the input text and encodes it into a series of numerical representations, or vectors, that capture its contextual information |
| **Decoder** | takes the encoded vectors and generates the output text |
| **Self-attention** (connects the many layers) | lets the model weigh the importance of different words/tokens relative to each other, capturing [[long-range-dependencies]] |

In the original model the encoder and decoder blocks are each repeated **6 times**
(§1.6, p. 13). See [[encoder-decoder-architecture]].

### The heart: scaled dot-product self-attention (§3.3–3.4)
Each input token is an [[token-embedding|embedding]] row vector $x^{(i)}$. Three
trainable matrices turn it into a **query**, **key** and **value**
([[query-key-value]]):

$$q^{(i)} = x^{(i)} W_q,\qquad k^{(i)} = x^{(i)} W_k,\qquad v^{(i)} = x^{(i)} W_v$$

- $x^{(i)}$ = embedding of token $i$ (a row vector of length $d_{in}$).
- $W_q, W_k, W_v$ = learned weight [[matrix|matrices]] of size $d_{in}\times d_{out}$.
- $q$ = "what am I looking for?", $k$ = "what do I offer?", $v$ = "what I actually pass on" (terms borrowed from databases, p. 70).

Stacking all $T$ tokens as rows of $X$, the whole layer is one line:

$$Z = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V, \quad Q=XW_q,\ K=XW_k,\ V=XW_v$$

Symbol by symbol:
- $QK^\top$ = a $T\times T$ table of [[inner-product|dot products]]: entry $(i,j)$ says how well query $i$ matches key $j$ ([[attention-scores]]). $^\top$ = [[matrix-transpose|transpose]].
- $\sqrt{d_k}$ = square root of the key dimension. Dividing by it keeps scores from getting huge, which would make softmax act like a step function with near-zero gradients (p. 69).
- $\mathrm{softmax}(\cdot)$ = applied row by row, so each row becomes positive weights that sum to 1 ([[attention-weights]]).
- $\cdot\,V$ = each output row is a weighted average of value vectors.
- $Z$ = the [[context-vector|context vectors]]: "enriched" embeddings that include information from all other tokens.

**Book numbers** ("Your journey starts with one step", $d_{in}=3$, $d_{out}=2$,
`torch.manual_seed(123)`, query = "journey", pp. 66–70):

| Quantity | Value |
|---|---|
| $q^{(2)}$ | `[0.4306, 1.4551]` |
| scores $\omega_2 = q^{(2)}K^\top$ | `[1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]` |
| weights $\mathrm{softmax}(\omega_2/\sqrt2)$ | `[0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]` |
| context $z^{(2)}$ | `[0.3061, 0.8210]` |

### Making it work for generation: causal attention (§3.5)
For a decoder (GPT), token $i$ must not see tokens after it. Scores above the
diagonal are set to $-\infty$ before softmax, so they get weight 0:

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
```

See [[causal-attention]] and [[causal-attention-mask]]. [[dropout]] may also be
applied to the attention weights to reduce overfitting.

### Several attentions in parallel: multi-head attention (§3.6)
Run $h$ attention "heads", each with its own $W_q,W_k,W_v$, then glue the
results side by side:

$$Z = [\,Z_1 \,\|\, Z_2 \,\|\, \cdots \,\|\, Z_h\,]$$

- $Z_i$ = context vectors from head $i$; $\|$ = concatenate along the feature
  dimension. The efficient version splits one big projection into heads
  (`head_dim = d_out // num_heads`) and adds an [[output-projection]] `out_proj`.
- Smallest GPT-2: **12 heads**, embedding size **768** (so 64 dims per head);
  largest GPT-2: **25 heads**, embedding size **1,600** (p. 90). See [[multi-head-attention]].

### The GPT (decoder-only) stack (§4.1)
A GPT model = embedding layers → a stack of identical [[transformer-block|transformer
blocks]] (each with masked multi-head attention) → output layers that produce
[[logits]] over the vocabulary (Fig. 4.2, p. 94). The book builds a block from
[[layer-normalization]], [[gelu]] activation, a [[feed-forward-network]] and
[[shortcut-connections]] plus the attention module (Fig. 4.3, p. 96). The 124M
GPT-2 uses 12 blocks, 12 heads, 768-dim embeddings and a 1,024-token context;
[[gpt-3]] has 96 layers and 175B parameters. See [[gpt-architecture]] and
[[decoder-only-transformer]].

## 🔴 Theory (Expert)
- **Why attention beat RNNs.** An encoder–decoder RNN compresses the whole input
  into its final hidden state; the decoder cannot directly access earlier encoder
  states, so long-range context is lost (§3.1, p. 53). [[bahdanau-attention]]
  (2014) let the decoder selectively access all input positions; three years
  later the transformer showed RNNs are not required at all, using
  *self*-attention inspired by Bahdanau's mechanism (§3.2, pp. 54–55).
- **"Self" vs. traditional attention.** Self-attention relates positions within a
  *single* sequence; traditional seq-to-seq attention relates two different
  sequences (input and output) (sidebar, p. 56).
- **Structure of the operation.** Without trainable weights, scores are the
  [[gram-matrix]] $XX^\top$ (symmetric). With $W_q\neq W_k$, $QK^\top =
  XW_qW_k^\top X^\top$ is a learned bilinear form and generally *not*
  symmetric. Each row of the attention matrix is a probability vector, so every
  context vector is a [[convex-combination]] of value vectors.
- **Scaling argument.** If query and key coordinates are roughly independent with
  zero mean and unit variance, $q\cdot k=\sum_{m=1}^{d_k} q_m k_m$ has variance
  about $d_k$; dividing by $\sqrt{d_k}$ brings it back to about 1, keeping softmax
  out of its saturated regime (this is the standard justification for the book's
  "avoid small gradients" remark, p. 69).
- **Cost.** All-pairs attention is $O(T^2)$ in the sequence length $T$.
  [[flashattention]] (App. B) computes the *same* exact result faster by
  optimizing memory access. App. B also cites a blog post noting that for
  contexts under ~32k tokens most compute is in the feed-forward layers, not
  attention.
- **Order information.** Attention by itself treats the input as a set; position
  is injected through [[positional-embedding|positional embeddings]] (the
  `context_length` in the config is the number of positions the model supports).
- **Layer norm placement.** The original transformer uses Post-LN (normalize
  after attention/FFN); GPT-2 and newer LLMs use Pre-LN, which can train more
  stably (App. B; see [[pre-layernorm-vs-post-layernorm]]).
- **Transformers vs. LLMs.** Transformers are also used in computer vision (e.g.
  the Vision Transformer), and some LLMs are not transformers (RWKV, Hyena,
  Mamba) (sidebar p. 10; App. B). He & Hofmann (2023) show transformer blocks
  can be simplified, e.g. without the value matrix and output projection (App. B).

## Connections
- Builds on: [[inner-product]], [[matrix]], [[matrix-multiplication]], [[softmax]], [[vector]], [[linear-mapping]], [[neural-network]], [[recurrent-neural-network]], [[encoder-decoder-architecture]]
- Core parts: [[attention-mechanism]], [[self-attention]], [[scaled-dot-product-attention]], [[query-key-value]], [[causal-attention]], [[multi-head-attention]], [[transformer-block]], [[layer-normalization]], [[feed-forward-network]], [[shortcut-connections]], [[positional-embedding]]
- Leads to: [[gpt-architecture]], [[decoder-only-transformer]], [[bert]], [[large-language-model]], [[next-token-prediction]]
- Related: [[bahdanau-attention]], [[flashattention]], [[machine-translation]], [[long-range-dependencies]]
- People: [[ashish-vaswani]], [[dzmitry-bahdanau]], [[yoshua-bengio]], [[jacob-devlin]], [[alec-radford]], [[alexey-dosovitskiy]], [[tri-dao]], [[albert-gu]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1, book p. 3 — selective attention.
- Same file §1.4, book pp. 7–10 — original transformer, encoder/decoder, Fig. 1.4, BERT vs GPT (Fig. 1.5), sidebar "Transformers vs. LLMs".
- Same file §1.6, book pp. 12–14 — decoder-only GPT, 6 repeated blocks, GPT-3 96 layers.
- Same file §1.7 + summary, book pp. 14–15.
- Same file §3.1, book pp. 52–54 — RNN encoder–decoder bottleneck.
- Same file §3.2, book pp. 54–55 — Bahdanau attention, self-attention definition.
- Same file §3.3, book pp. 55–64 — simplified self-attention, dot products, softmax.
- Same file §3.4, book pp. 64–74 — trainable $W_q,W_k,W_v$, scaling by $\sqrt{d_k}$, worked numbers.
- Same file §3.5, book pp. 74–82 — causal masking.
- Same file §3.6, book pp. 82–91 — multi-head attention, GPT-2 head counts.
- Same file §4.1, book pp. 93–98 — GPT model overview, Figs. 4.2–4.4.
- Same file Appendix B, book pp. 289–293 — Vaswani et al. 2017, Bahdanau et al. 2014, FlashAttention, ViT, RWKV/Hyena/Mamba, Pre-LN vs Post-LN.
