---
title: Context Vector (Contextualized Embedding)
type: concept
tags: [llm, attention, embeddings, linear-algebra]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Context Vector (Contextualized Embedding)

> **One-liner:** A context vector is a word's embedding after it has "listened" to the rest of the sentence — the same word gets a different vector in a different sentence.

## 🟢 Intuition (Dummy)
A plain [[token-embedding]] is like a dictionary entry: "bank" always gets the same
vector, whether the sentence is about rivers or money. That's a problem, because
meaning depends on the neighbors.

A **context vector** is that dictionary entry *rewritten after reading the
sentence*. Attention lets each word gather a little information from every other
word, weighted by how relevant that word is, and the result is a new, "enriched"
vector for that word (book p. 57). The book also calls these **contextualized output
embeddings** (book p. 20). Same word, different sentence → different context vector.

## 🟡 Mechanics (Practitioner)

### Definition
For each input $\mathbf x^{(i)}$ the attention mechanism computes a context vector
$\mathbf z^{(i)}$ that contains information about $\mathbf x^{(i)}$ **and all other
inputs** $\mathbf x^{(1)},\dots,\mathbf x^{(T)}$ (book p. 57, Fig. 3.7). It is a
**weighted sum**:

$$\mathbf z^{(i)} = \sum_{j=1}^{T}\alpha_{ij}\,\mathbf v^{(j)}$$

Symbol by symbol:
- $\mathbf z^{(i)}$ = the context vector for position $i$ (the letter $z$ is the book's choice);
- $\sum_{j=1}^{T}$ = add up over every position $j$ from 1 to $T$ ($T$ = number of tokens);
- $\alpha_{ij}$ = the [[attention-weights|attention weight]]: how much position $i$ "listens to" position $j$; all $\alpha_{ij}\ge0$ and $\sum_j\alpha_{ij}=1$;
- $\mathbf v^{(j)}$ = what position $j$ contributes. In simplified attention (§3.3) it is just the input $\mathbf x^{(j)}$; with trainable weights (§3.4) it is the **value vector** $\mathbf v^{(j)}=\mathbf x^{(j)}W_v$.

All positions at once, in [[matrix]] form: $Z = AV$, where row $i$ of $Z$ is $\mathbf z^{(i)}$ and row $i$ of $A$ holds the weights $\alpha_{i1},\dots,\alpha_{iT}$.

### Worked example 1 — no trainable weights (book pp. 60–63)
Inputs "Your journey starts with one step." ($d=3$). Query "journey", weights
`[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]`. First coordinate of $\mathbf z^{(2)}$:
$0.1385\cdot0.43+0.2379\cdot0.55+0.2333\cdot0.57+0.1240\cdot0.22+0.1082\cdot0.77+0.1581\cdot0.05\approx0.442$.
The book's full result: $\mathbf z^{(2)}$ = `[0.4419, 0.6515, 0.5683]`. In code:
```python
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
```
All six at once, `attn_weights @ inputs` (book p. 63):
```
[[0.4421, 0.5931, 0.5790],
 [0.4419, 0.6515, 0.5683],
 [0.4431, 0.6496, 0.5671],
 [0.4304, 0.6298, 0.5510],
 [0.4671, 0.5910, 0.5266],
 [0.4177, 0.6503, 0.5645]]
```

### Worked example 2 — trainable weights (book pp. 69–71)
With $W_q,W_k,W_v$ mapping 3-D → 2-D (`torch.manual_seed(123)`), the context vector
for "journey" is a weighted sum of **value** vectors: $\mathbf z^{(2)}$ = `[0.3061, 0.8210]`
(Fig. 3.17). `SelfAttention_v1` returns all six rows, the second matching:
`[[0.2996, 0.8053], [0.3061, 0.8210], [0.3058, 0.8203], …]`.

### Shapes through the book
| Module | Input shape | Context-vector shape | Book |
|---|---|---|---|
| simplified self-attention | (6, 3) | (6, 3) | p. 63 |
| `SelfAttention_v1` ($d_{out}=2$) | (6, 3) | (6, 2) | p. 71 |
| `CausalAttention` on a batch | (2, 6, 3) | (2, 6, 2) | p. 82 |
| `MultiHeadAttentionWrapper`, 2 heads | (2, 6, 3) | (2, 6, 4) | p. 85 |
| `MultiHeadAttention`, $d_{out}=2$ | (2, 6, 3) | (2, 6, 2) | p. 90 |
| GPT-2 transformer block | (2, 4, 768) | (2, 4, 768) | p. 116 |

### In a GPT model
- With a [[causal-attention-mask]], the sum only runs over $j\le i$, so $\mathbf z^{(i)}$
  mixes the current and *earlier* tokens only (book pp. 74–78).
- In the GPT model, the 768-dimensional output vectors for each input token are the
  context vectors from Ch. 3 (Fig. 4.4, book p. 98).
- A [[transformer-block]] keeps the shape the same, but each output vector is
  "re-encoded to integrate contextual information from across the entire input
  sequence" (book p. 116). Stacking 12 blocks (GPT-2 124M) refines the context
  vectors again and again.
- Because of the causal mask, the **last** token's context vector is the only one that
  has seen the whole input — which is why classification fine-tuning reads the last
  token (Fig. 6.12, book pp. 188–190; see [[last-token-pooling]]).

## 🔴 Theory (Expert)
- **Convex hull.** Since the weights are non-negative and sum to 1, $\mathbf z^{(i)}$ is
  a [[convex-combination]] of $\{\mathbf v^{(1)},\dots,\mathbf v^{(T)}\}$ and lies in
  their convex hull. One consequence: in simplified attention every coordinate of
  $\mathbf z^{(i)}$ lies between the smallest and largest value of that coordinate over
  the inputs (e.g. the first coordinates 0.4177–0.4671 above all lie in
  [0.05, 0.77]). Attention *averages*; the [[feed-forward-network]] and
  [[shortcut-connections]] in the transformer block let the model move beyond the hull.
- **Expectation view.** Treat row $i$ of $A$ as a [[probability-distribution]] $p_i(j)$
  over positions. Then $\mathbf z^{(i)}=\mathbb E_{j\sim p_i}[\mathbf v^{(j)}]$ — an
  [[expected-value-and-mean|expected value]] of the value vectors.
- **Static vs. contextual.** A token embedding is a function of the token ID only,
  $\mathbf x_t=E[\mathrm{id}_t]$ (plus a positional term). A context vector is a function
  of the *whole* sequence, $\mathbf z^{(i)}=f(\mathbf x^{(1)},\dots,\mathbf x^{(T)})_i$. That
  is the formal meaning of "contextualized" (book pp. 20, 57). Contrast with
  [[word2vec]], whose vectors are static (book p. 20).
- **Linear in values, nonlinear in queries/keys.** For fixed weights $A$, $Z=AV$ is
  linear in $V$; but $A$ depends on $X$ through softmax, so $Z$ as a function of $X$ is
  nonlinear.
- **Not the RNN "context vector".** In older encoder–decoder RNNs, the single final
  [[hidden-state]] (also sometimes called a context vector in the literature) had to
  summarize the whole sentence (book p. 53). Attention instead gives *one context
  vector per position*, built fresh for each query — removing that bottleneck.

## Connections
- Builds on: [[token-embedding]], [[vector]], [[attention-weights]], [[softmax]], [[convex-combination]], [[matrix-multiplication]]
- Leads to: [[transformer-block]], [[gpt-architecture]], [[logits]], [[last-token-pooling]]
- Related: [[self-attention]], [[attention-mechanism]], [[query-key-value]], [[causal-attention-mask]], [[multi-head-attention]], [[hidden-state]], [[word2vec]], [[expected-value-and-mean]]
- Used by: [[large-language-model]], [[classification-fine-tuning]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book p. 20 (contextualized output embeddings; LLM-learned vs. Word2Vec embeddings)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3, book pp. 56–64 (definition p. 57; Fig. 3.7, Fig. 3.10; tensors pp. 60–64)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4, book pp. 69–71 (weighted sum of value vectors, Fig. 3.17; `SelfAttention_v1` output)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5–3.6, book pp. 74–91 (causal context vectors; shapes pp. 82, 85, 90; chapter summary p. 91)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book p. 98 (GPT outputs are context vectors, Fig. 4.4)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book p. 116 (shape preserved, content re-encoded)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 188–190 (last token sees all tokens, Fig. 6.12)
