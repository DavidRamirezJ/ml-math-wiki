---
title: Embedding Layer (Lookup Table)
type: concept
tags: [deep-learning, llm, pytorch, linear-algebra, embeddings]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Embedding Layer (Lookup Table)

> **One-liner:** An embedding layer is a big table of numbers with one row per token; you hand it a token ID and it hands you back that row as the token's vector.

## 🟢 Intuition (Dummy)
Imagine a huge address book. Every word the model knows (every [[token]]) has a page number,
its [[token-id]]. On each page there's a short list of numbers that describes the word. When
the model reads the ID "3", it just flips to page 3 and copies that list. That's it. That's the
whole embedding layer: **look up a row, return it**.

So why is this a "layer" and not just a dictionary? Because the numbers on every page start out
**random**, and during training the model keeps nudging them so that useful patterns appear
(similar words end up with similar lists). The table is *learned*. A neural network can't do
anything with the integer "3" itself; it needs a list of continuous numbers it can gently adjust.
The embedding layer is the bridge from "discrete ID" to "adjustable vector".

## 🟡 Mechanics (Practitioner)

### The weight matrix
An embedding layer stores one trainable [[matrix]]:
$$W \in \mathbb{R}^{V \times d}$$
Read it symbol by symbol:
- $W$ = the embedding weight matrix (PyTorch: `embedding_layer.weight`).
- $\mathbb{R}^{V\times d}$ = "a grid of real numbers with $V$ rows and $d$ columns".
- $V$ = vocabulary size (`vocab_size`, `num_embeddings`): **one row per possible token**.
- $d$ = embedding dimension (`output_dim`, `embedding_dim`): **one column per feature** of the vector. See [[embedding-dimension]].

### The lookup
$$\text{Embedding}(i) = W_{i,:} \in \mathbb{R}^{d}$$
- $i$ = the token ID (a row index, **counting from 0**).
- $W_{i,:}$ = "row $i$ of $W$, all columns" (the colon means "everything along this axis").
- $\in \mathbb{R}^d$ = the result is a [[vector]] with $d$ numbers.

### The book's toy example (§2.7)
Four tokens with IDs `[2, 3, 5, 1]` ("fox jumps over dog" in the small §2.3 vocabulary), a
vocabulary of $V=6$ (instead of the 50,257 BPE tokens) and $d=3$ (GPT-3 uses 12,288 dimensions;
book p. 42):

```python
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
```

$$
W=\begin{bmatrix}
0.3374 & -0.1778 & -0.1690\\
0.9178 & 1.5810 & 1.3010\\
1.2753 & -0.2010 & -0.1606\\
-0.4015 & 0.9666 & -1.1481\\
-1.1589 & 0.3255 & -0.6315\\
-2.8400 & -0.7849 & -1.4096
\end{bmatrix}
$$

6 rows (one per token) × 3 columns (one per embedding dimension), all small random values,
marked `requires_grad=True` so training can change them.

- `embedding_layer(torch.tensor([3]))` → `tensor([[-0.4015, 0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)`.
  That is the **fourth** row: row index 3, because Python counts from 0.
- `embedding_layer(input_ids)` → a $4\times3$ matrix whose rows are rows 2, 3, 5, 1 of $W$:

$$
\begin{bmatrix}
1.2753 & -0.2010 & -0.1606\\
-0.4015 & 0.9666 & -1.1481\\
-2.8400 & -0.7849 & -1.4096\\
0.9178 & 1.5810 & 1.3010
\end{bmatrix}
$$

Token ID 5 comes from the **sixth** row (Fig. 2.16, book p. 44).

**Position-blind.** The same token ID always returns the same row, wherever it sits in the
sequence: in Fig. 2.17 (book p. 44) the IDs `[2, 3, 5, 2]` give identical vectors in positions 1
and 4. Self-attention has no notion of order either, so §2.8 adds a second, *positional*
embedding layer (book pp. 43–44).

### Shapes in, shapes out
| Input IDs | Output |
|---|---|
| `(1,)` | `(1, d)` |
| `(4,)` | `(4, d)` |
| `(batch, seq_len)` | `(batch, seq_len, d)` |

At realistic size (§2.8): `torch.nn.Embedding(50257, 256)`, a batch of IDs of shape `[8, 4]`
→ token embeddings of shape `[8, 4, 256]` (book pp. 46–47).

### Two embedding layers in GPT
GPT uses the *same* layer type twice (§2.8, §4.1):
$$\mathbf{x}_t = E_{\text{tok}}[\text{id}_t] + E_{\text{pos}}[t]$$
- $\mathbf{x}_t$ = the input vector for position $t$ that goes into the transformer.
- $E_{\text{tok}} \in \mathbb{R}^{V\times d}$ = the **token** embedding table; $E_{\text{tok}}[\text{id}_t]$ = the row for the token at position $t$.
- $E_{\text{pos}} \in \mathbb{R}^{L\times d}$ = the **positional** embedding table with one row per position up to the [[context-length]] $L$; its input is just `torch.arange(context_length)` = $0,1,\dots,L-1$.
- $+$ = element-wise vector addition (PyTorch [[broadcasting]] repeats the positional rows over the batch).

```python
self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])      # 50257 x 768
self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 1024 x 768
```
For GPT-2 124M the token table alone holds $50{,}257 \times 768 = 38{,}597{,}376$ parameters
(book p. 121). See [[absolute-positional-embedding]] and [[positional-embedding]].

### Why not just feed the IDs in?
GPT-like LLMs are deep neural networks trained with [[backpropagation]], so they need a
**continuous vector representation** (book p. 42). The table rows are exactly that, and they
are optimized during [[pretraining]] in chapter 5.

## 🔴 Theory (Expert)

### Embedding = one-hot × matrix
The book's NOTE (p. 43): the embedding layer is "essentially just a more efficient way of implementing
[[one-hot-encoding]] followed by matrix multiplication in a fully connected layer".
Formally, let $\mathbf{x}_i\in\{0,1\}^V$ be the one-hot vector with a single 1 at index $i$
(the $i$-th standard basis vector of $\mathbb{R}^V$). Then
$$\mathbf{x}_i^{\top} W = W_{i,:}, \qquad X W = E \;\;\text{for } X\in\{0,1\}^{n\times V}.$$
- $\mathbf{x}_i^\top W$ = row vector times matrix = a weighted sum of the rows of $W$ with weights $\mathbf{x}_i$; since only weight $i$ is 1, you get row $i$.
- $X$ = stack of $n$ one-hot rows; $E\in\mathbb{R}^{n\times d}$ = the stacked embeddings.

So an embedding layer *is* the [[linear-mapping]] $\mathbb{R}^V\to\mathbb{R}^d$, $\mathbf{x}\mapsto W^\top\mathbf{x}$,
evaluated only on basis vectors. The lookup skips the $V-1$ multiplications by zero, which is
why it is cheaper (for $V=50{,}257$ that matters a lot). In `nn.Linear(V, d, bias=False)` the stored
weight would have shape $(d, V)$, i.e. $W^\top$ (detail from the supplementary code idea,
https://mng.bz/ZEB5, not the printed text).

### Gradient is sparse (implied)
Because $E = XW$, the [[chain-rule]] gives
$$\frac{\partial L}{\partial W} = X^{\top}\frac{\partial L}{\partial E},$$
- $L$ = the scalar loss; $\partial L/\partial E$ = gradient w.r.t. the looked-up rows.
- $X^\top$ scatters each row-gradient back into the row it came from.

Only the rows that were actually looked up in a batch get a non-zero gradient. The
`grad_fn=<EmbeddingBackward0>` tag shows autograd recorded the lookup.

### Initialization
The book only says "small, random values" (p. 42). PyTorch's default `nn.Embedding` init is
$\mathcal{N}(0,1)$ (external fact), which explains a value like $-2.84$ in the toy table.

### Weight tying
The output head `out_head` ($768\to50{,}257$) has the same weight shape as `tok_emb`. The
original GPT-2 **ties** them ($W_{\text{out}} = E_{\text{tok}}$), so each logit becomes the dot
product of the final hidden vector with a token's embedding row. The book keeps them
separate because, in the author's experience, that trains better (book p. 121). See
[[weight-tying]]. When loading OpenAI weights, `tok_emb.weight ← params["wte"]` and
`pos_emb.weight ← params["wpe"]` (shapes $50257\times768$ and $1024\times768$), and
`out_head.weight ← params["wte"]` (book pp. 165–166).

## Connections
- Builds on: [[token-id]], [[vocabulary]], [[matrix]], [[vector]], [[one-hot-encoding]], [[linear-mapping]]
- Leads to: [[token-embedding]], [[positional-embedding]], [[absolute-positional-embedding]], [[input-embeddings]], [[transformer-architecture]]
- Used by: [[gpt-architecture]], [[word-embeddings]], [[weight-tying]], [[llm-training-loop]]
- Related: [[embedding-dimension]], [[backpropagation]], [[pytorch-nn-module]], [[broadcasting]], [[context-length]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7, book pp. 41–44 (`torch.nn.Embedding`, toy 6×3 table, lookups, one-hot NOTE p. 43, Fig. 2.16 p. 44).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book pp. 43–49 (positional embedding layer, 50,257×256 table, `[8, 4, 256]` output).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 95–97 (`tok_emb`, `pos_emb` in `DummyGPTModel`); §4.6, book p. 121 (weight tying, 38,597,376 params); §5.5, book pp. 165–166 (`wte`/`wpe` loading).
