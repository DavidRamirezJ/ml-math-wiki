---
title: Attention Mechanism
type: concept
tags: [llm, transformer, attention, deep-learning, nlp]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Attention Mechanism

> **One-liner:** Attention lets a model look back at *every* part of its input and decide, word by word, which parts matter most right now.

## 🟢 Intuition (Dummy)
Picture yourself translating a long sentence. You don't read it once, memorize it,
close the book and then write the translation from memory. You keep glancing back
at the original, and each time you write a word, your eyes jump to the few source
words that matter for *that* word. That glancing back is **attention**.

Older translation networks ([[recurrent-neural-network|RNNs]]) worked the "memorize
and close the book" way: they squeezed the whole sentence into one memory vector and
then had to translate from that single summary. For long sentences, things got lost
(book p. 53). Attention fixes this by giving the model a way to reach *all* input
words at every step, with a learned "spotlight" that is bright on the important
words and dim on the rest (Fig. 3.5, book p. 54).

The book calls attention the idea "at the heart of every LLM" (book p. 15): the
[[transformer-architecture|transformer]] is built around it, and it is what lets an
[[large-language-model|LLM]] "pay selective attention to different parts of the
input" when it makes a prediction (book p. 3).

## 🟡 Mechanics (Practitioner)

### The core recipe: score → normalize → mix
Every attention variant in the book follows the same three steps (Fig. 3.12, book p. 62):

1. **Score** how relevant each input is to the current position (the *query*).
2. **Normalize** the scores into weights that are positive and add up to 1.
3. **Mix**: take the weighted sum of the inputs. The result is a [[context-vector]].

In math, for one query position $i$ and $T$ input positions:

$$\omega_{ij} = \text{score}(\text{query}_i,\ \text{input}_j), \qquad
\alpha_{ij} = \frac{\exp(\omega_{ij})}{\sum_{t=1}^{T}\exp(\omega_{it})}, \qquad
\mathbf z^{(i)} = \sum_{j=1}^{T} \alpha_{ij}\,\mathbf v^{(j)}$$

Reading it symbol by symbol:
- $\omega_{ij}$ ("omega i j") = the **attention score**: a raw, unnormalized number saying how well position $i$ matches position $j$ (book pp. 57–58).
- $\exp(\cdot)$ = the exponential $e^{(\cdot)}$; it makes every score positive.
- $\sum_{t=1}^{T}$ = "add up over all $T$ positions $t = 1, 2, \dots, T$". Dividing by this sum makes the weights add up to 1. Together, this fraction is the [[softmax]].
- $\alpha_{ij}$ ("alpha i j") = the **attention weight**: how much of position $j$ goes into the output for position $i$. $\alpha_{ij} > 0$ and $\sum_j \alpha_{ij} = 1$ (book pp. 59–60).
- $\mathbf v^{(j)}$ = the vector we take from position $j$ (in the simplest version, just the input embedding $\mathbf x^{(j)}$; later a learned *value* vector).
- $\mathbf z^{(i)}$ = the output **context vector** for position $i$: a blend of all inputs, weighted by relevance.

In the book's attention for LLMs, the score is a **[[inner-product|dot product]]**,
$\omega_{ij} = \mathbf q^{(i)}\cdot\mathbf k^{(j)}$, scaled by $1/\sqrt{d_k}$ — see
[[self-attention]] and [[scaled-dot-product-attention]].

### Two flavors in the book
| Flavor | Query comes from | Inputs attended | Where in the book |
|---|---|---|---|
| **Bahdanau ("traditional") attention** (2014) | the RNN *decoder's* current state | the *encoder's* states of a different (source) sequence | §3.2, book p. 54; sidebar p. 56 |
| **[[self-attention]]** (2017 transformer) | a position in the sequence | all positions of the **same** sequence | §3.3–3.6, book pp. 55–91 |

The sidebar "The 'self' in self-attention" (book p. 56) draws exactly this line:
traditional attention relates elements of *two different* sequences (input and output
of a sequence-to-sequence model); self-attention relates positions *within one*
sequence.

### The book's four-step path to LLM attention (Fig. 3.2, book p. 51)
1. **Simplified self-attention** — no trainable weights; scores are dot products of raw embeddings (§3.3).
2. **Self-attention with trainable weights** — learned $W_q, W_k, W_v$ (§3.4).
3. **[[causal-attention|Causal attention]]** — a [[causal-attention-mask]] hides future tokens so the model can generate one word at a time (§3.5).
4. **[[multi-head-attention|Multi-head attention]]** — several attention "heads" run in parallel and their outputs are joined (§3.6).

The chapter summary condenses it: attention mechanisms "transform input elements into
enhanced context vector representations that incorporate information about all inputs"
(book p. 91).

### Tiny worked example (book §3.3.1, pp. 57–60)
Sentence "Your journey starts with one step." with 3-D embeddings, query = "journey"
$\mathbf x^{(2)} = [0.55, 0.87, 0.66]$:

- Scores $\omega_{2j}$: `[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]`
- Softmax weights $\alpha_{2j}$: `[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]` (sum = 1)
- Context vector $\mathbf z^{(2)}$: `[0.4419, 0.6515, 0.5683]`

"journey" pays the most attention to itself (0.2379) and to "starts" (0.2333), whose
embedding is almost the same — the dot product acts as a similarity meter.

### Attention also handles padding
When batching texts of different lengths, GPT pads with `<|endoftext|>`, but a
**mask** makes sure padded tokens are *not attended to*, so which pad token you pick
does not matter (book p. 32). The same "−∞ before softmax" trick is the
[[causal-attention-mask]].

## 🔴 Theory (Expert)
- **Attention as a soft, differentiable dictionary lookup.** The book borrows the
  query/key/value names from information retrieval and databases (sidebar, book p. 70):
  a hard lookup returns the value whose key *equals* the query; attention returns a
  softmax-weighted average of *all* values, weighted by query–key similarity. Because
  every step is differentiable, the "where to look" rule is learned by
  [[backpropagation]] (training happens in Ch. 5).
- **Convex combination.** Since $\alpha_{ij}\ge 0$ and $\sum_j\alpha_{ij}=1$, every
  context vector is a [[convex-combination]] of the value vectors: it lies inside their
  convex hull. Each row of the attention matrix is a categorical
  [[probability-distribution]] over positions.
- **Why it beats the RNN bottleneck.** An encoder–decoder RNN must carry the whole
  source in one fixed-size hidden state, and the decoder "cannot directly access
  earlier hidden states" (book p. 53). Attention gives every output step a direct,
  one-hop path to every input position, so information (and gradient) need not pass
  through many recurrent steps — the root of the [[long-range-dependencies]] gain.
- **Cost.** All-pairs scoring builds a $T\times T$ matrix, so time and memory grow as
  $O(T^2 d)$ and $O(T^2)$ in the sequence length $T$ (standard result; not stated in
  the book's Ch. 3). Appendix B points to [[flashattention]] (Dao et al., 2022; Dao,
  2023), an exact, IO-aware implementation that is mathematically the same as standard
  self-attention (book p. 291), and to PyTorch's `scaled_dot_product_attention`
  (book p. 292).
- **Order blindness.** Dot products plus softmax do not depend on *where* a token sits;
  attention alone is position-agnostic, which is why LLMs add
  [[positional-embedding|positional embeddings]] (book pp. 43–45).
- **History.** Bahdanau, Cho & Bengio (2014) added attention to RNN translation;
  Vaswani et al. (2017) showed RNNs are not needed at all and built the transformer on
  self-attention (book pp. 54–55, 291).

## Connections
- Builds on: [[vector]], [[inner-product]], [[softmax]], [[matrix]], [[recurrent-neural-network]], [[encoder-decoder-architecture]]
- Leads to: [[self-attention]], [[scaled-dot-product-attention]], [[causal-attention]], [[causal-attention-mask]], [[multi-head-attention]], [[context-vector]], [[transformer-architecture]]
- Related: [[bahdanau-attention]], [[attention-scores]], [[attention-weights]], [[query-key-value]], [[convex-combination]], [[flashattention]], [[long-range-dependencies]]
- Used by: [[large-language-model]], [[gpt-architecture]], [[bert]], [[machine-translation]]
- People: [[dzmitry-bahdanau]], [[yoshua-bengio]], [[ashish-vaswani]], [[tri-dao]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1, book p. 3 (selective attention in transformers)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7, book p. 15 (attention "at the heart of every LLM"; selective access to the whole input)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book p. 32 (masking padded tokens)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1, book pp. 50–54 (four variants, Fig. 3.2; RNN bottleneck, Fig. 3.4)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.2, book pp. 54–55 (Bahdanau attention, Fig. 3.5; self-attention, Fig. 3.6)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3, book pp. 55–64 (sidebar p. 56; worked example pp. 57–60; three steps, Fig. 3.12)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book pp. 291–292 (references: Bahdanau 2014, Vaswani 2017, FlashAttention)
