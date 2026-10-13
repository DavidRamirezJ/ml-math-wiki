---
title: Embedding Dimension
type: concept
tags: [llm, embeddings, hyperparameters, linear-algebra]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Embedding Dimension

> **One-liner:** The embedding dimension is how many numbers each token's vector has — the length of the "coordinates" a model gives every word.

## 🟢 Intuition (Dummy)
Think of describing a person. With 2 numbers (height, age) you can tell people
apart a little. With 100 numbers (height, age, favourite food, accent, …) you
can capture much finer differences. But every extra number costs memory and
time.

A word embedding is the same: its **dimension** is how many "traits" each word
gets. More dimensions = room for subtler relationships, but a bigger, slower
model. Toy examples in the book use 3; GPT-2 small uses 768; the biggest GPT-3
uses 12,288 (book pp. 20, 42).

We can only *draw* 2 or 3 dimensions, which is why pictures of embeddings are
always squashed down to a flat scatterplot (book p. 20).

## 🟡 Mechanics (Practitioner)
If the embedding of token $w$ is

$$\mathbf{e}_w \in \mathbb{R}^{d},$$

then $d$ is the embedding dimension. Read it as: "$\mathbf{e}_w$ (the
[[vector]] for word $w$) is an element of ($\in$) the space of lists of $d$
real numbers ($\mathbb{R}^{d}$)."

In code it is `output_dim` / `emb_dim`, the second argument of the
[[embedding-layer]]:

```python
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

which stores a weight [[matrix]] $W \in \mathbb{R}^{V\times d}$:
- $V$ = vocabulary size (number of rows, one per token),
- $d$ = embedding dimension (number of columns, one per "trait").

So the embedding layer alone has $V \cdot d$ parameters.

**What the book uses** (all from the text):

| Setting | $V$ | $d$ | Where |
|---|---|---|---|
| Fig. 2.2 / Fig. 2.3 illustrations | — | 3 / 2 | book pp. 19–20 |
| Toy embedding layer | 6 | 3 | §2.7, book p. 42 |
| Realistic experiment | 50,257 | 256 | §2.8, book p. 46 |
| GPT-2 small (124M) | 50,257 | 768 | §4.1, book p. 95 |
| GPT-2 medium (355M) | 50,257 | 1,024 | §5.5, book p. 163 |
| GPT-2 large (774M) | 50,257 | 1,280 | §5.5, book p. 163 |
| GPT-2 XL (1,558M) | 50,257 | 1,600 | §5.5, book p. 163 |
| Largest GPT-3 (175B) | — | 12,288 | book pp. 20, 42 |

**Worked numbers** (our arithmetic): the §2.8 layer has
$50{,}257 \times 256 = 12{,}865{,}792$ weights; GPT-2 small's token-embedding
matrix has $50{,}257 \times 768 = 38{,}597{,}376$.

**Shapes in practice** (§2.8, book pp. 46–47): a batch of 8 sequences of 4 token
IDs, shape `[8, 4]`, becomes `[8, 4, 256]` after the embedding layer — the
last axis is $d$. The [[positional-embedding]] must have the *same* $d$,
because the two are added element by element.

**Another name:** the embedding size is often called the **dimensionality of the
model's hidden states** (book p. 20). In a GPT, every [[transformer-block]]
takes $d$-dimensional vectors in and gives $d$-dimensional vectors out, so
$d$ (`emb_dim`) is the model's "width". The book notes the LLM returns one
768-dimensional output vector per 768-dimensional input embedding (§4.1, book
p. 98).

## 🔴 Theory (Expert)
- **The trade-off** the book names (book pp. 19–20): higher $d$ can capture more
  nuanced relationships but costs computational efficiency; model makers pick
  $d$ per variant as a performance/efficiency balance. Embeddings can range
  "from one to thousands" of dimensions.
- **How parameters scale.** Embedding parameters grow like $V d$. The
  [[transformer-block]] parameters grow roughly like $12 d^2$ per block
  (attention $\approx 4d^2$, feed-forward with 4× expansion $\approx 8d^2$;
  our estimate). So doubling $d$ roughly quadruples the per-block size — see
  [[parameter-count-and-memory]].
- **Heads split $d$.** In [[multi-head-attention]], $d$ is divided among $h$
  heads, $d_{\text{head}} = d/h$. For every GPT-2 size this gives 64
  (768/12, 1024/16, 1280/20, 1600/25; derived from the §5.5 table).
- **Rank limit.** The lookup $\mathbf{x}^{\top}W$ is a [[linear-mapping]] from
  $\mathbb{R}^{V}$ to $\mathbb{R}^{d}$; its rank is at most $\min(V,d)=d$
  ([[basis-and-rank]]). With weight tying, the output logits $\mathbf{h}W^{\top}$
  are therefore constrained to a $d$-dimensional subspace of $\mathbb{R}^{V}$ —
  one reason $d$ cannot be made tiny.
- **Visualisation.** For $d>3$ you need a projection (e.g.
  [[principal-component-analysis]], [[dimensionality-reduction]]) to look at
  embeddings; distances in the 2-D plot are only approximations of the real
  ones.

## Connections
- Builds on: [[vector]], [[vector-space]], [[matrix]], [[word-embeddings]]
- Leads to: [[gpt-model-configuration]], [[parameter-count-and-memory]], [[multi-head-attention]], [[model-capacity]]
- Related: [[hyperparameter]], [[embedding-layer]], [[token-embedding]], [[positional-embedding]], [[hidden-state]], [[dimensionality-reduction]], [[basis-and-rank]]
- Used by: [[gpt-2]], [[gpt-3]], [[transformer-architecture]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book pp. 19–20 (range, trade-off, hidden-state naming, GPT-2 768 / GPT-3 12,288).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7, book p. 42 (toy `output_dim = 3`, GPT-3 12,288).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book pp. 46–47 (256-dim layer, `[8, 4, 256]`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 95–98 (`emb_dim = 768`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book p. 163 (GPT-2 size table).
