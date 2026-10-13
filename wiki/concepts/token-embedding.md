---
title: Token Embedding
type: concept
tags: [llm, embeddings, linear-algebra, pytorch]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Token Embedding

> **One-liner:** A token embedding is the list of numbers an LLM looks up for each token ID — one learned row of a big table — so words become vectors the network can compute with.

## 🟢 Intuition (Dummy)
After tokenization, a sentence is just a list of seat numbers ([[token-id|token IDs]]) like
`2, 3, 5, 1`. Those numbers can't be added or compared in any useful way — "fox" isn't "two-thirds
of jumps". So the LLM keeps a giant **table**: one row per word in its [[vocabulary]], and each row
holds a short list of numbers, the word's "coordinates". To embed a token you just go to its row
and copy it out.

At the start the table is filled with random numbers — every word sits at a random spot. During
training, the model keeps nudging the rows so that words used in similar ways drift toward each
other. After training, the table is a learned map of word meaning.

One limitation: the table has no idea *where* the word appears in the sentence. "fox" at the start
and "fox" at the end get the exact same row. That's why GPT adds a second vector for the position
(see [[positional-embedding]]).

## 🟡 Mechanics (Practitioner)

### It's a row lookup
The embedding layer holds a weight [[matrix]]
$$W \in \mathbb{R}^{V\times d}$$
- $W$ = the embedding weight matrix (the "table").
- $V$ = vocabulary size (number of rows); $d$ = embedding dimension (number of columns).
- $\mathbb{R}^{V\times d}$ = "a grid of real numbers with $V$ rows and $d$ columns".

The embedding of token ID $i$ is simply row $i$:
$$\text{Embedding}(i) = W_{i,:} \in \mathbb{R}^{d}$$
- $W_{i,:}$ = "row $i$, all columns" (the colon means "everything along this axis").
- $i$ starts at 0, as in Python.

"The embedding layer is essentially a lookup operation that retrieves rows from the embedding
layer's weight matrix via a token ID" (book p. 43).

### The book's toy example (§2.7, book pp. 42–44)
Vocabulary of 6 tokens, $d = 3$ (Fig. 2.16 labels the IDs with the §2.3 toy vocabulary of Fig. 2.6 —
which actually has 8 entries, so the 6 is just a simplification), IDs `[2, 3, 5, 1]` = "fox jumps over dog":
```python
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size, output_dim = 6, 3
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
- `embedding_layer(torch.tensor([3]))` → `[[-0.4015, 0.9666, -1.1481]]` = row index 3 (the *fourth* row).
- `embedding_layer(input_ids)` → a $4\times3$ matrix whose rows are rows 2, 3, 5, 1 of $W$:
$$
\begin{bmatrix}
1.2753 & -0.2010 & -0.1606\\
-0.4015 & 0.9666 & -1.1481\\
-2.8400 & -0.7849 & -1.4096\\
0.9178 & 1.5810 & 1.3010
\end{bmatrix}
$$
Fig. 2.16: ID 5 picks "the sixth instead of the fifth row because Python starts counting at 0"
(book p. 44). The weights are "small, random values" that get optimized during training in ch. 5
(book p. 42); the printed tensor has `requires_grad=True` and the output carries
`grad_fn=<EmbeddingBackward0>`, so gradients flow back into the looked-up rows.

### Same thing as one-hot × matrix
"The embedding layer approach described here is essentially just a more efficient way of
implementing one-hot encoding followed by matrix multiplication in a fully connected layer"
(book p. 43). In symbols:
$$\mathbf{e}_i = \mathbf{x}_i^{\top} W,\qquad \mathbf{x}_i = (0,\dots,0,\underbrace{1}_{\text{position } i},0,\dots,0)^{\top}$$
- $\mathbf{x}_i$ = the [[one-hot-encoding|one-hot]] vector for ID $i$ (length $V$, a single 1).
- $\mathbf{x}_i^\top$ = the same vector laid on its side (a row); multiplying it by $W$ picks out row $i$.
- $\mathbf{e}_i$ = the embedding. The lookup skips all the multiplications by 0 — that's the efficiency.

### Realistic sizes and batched shapes
| Setting | $V$ | $d$ | Parameters in $W$ ($V\cdot d$) | Page |
|---|---|---|---|---|
| Toy (§2.7) | 6 | 3 | 18 | p. 42 |
| §2.8 experiment | 50,257 | 256 | 12,865,792 (our arithmetic) | p. 46 |
| GPT-2 124M (`GPT_CONFIG_124M`) | 50,257 | 768 | 38,597,376 (our arithmetic) | pp. 95, 121, 163 |
| Largest GPT-3 | — | 12,288 | — | pp. 20, 42 |

In §2.8 a batch of IDs of shape `[8, 4]` (8 samples × 4 tokens, from the
[[sliding-window-sampling]] loader) becomes `token_embeddings.shape = [8, 4, 256]` (book pp. 46–47).
Then the [[positional-embedding]] of shape `[4, 256]` is added by [[broadcasting]]:
$$X_{b,t,:} = T_{b,t,:} + P_{t,:}$$
($T$ = token embeddings $[8,4,256]$, $P$ = positional embeddings $[4,256]$, $b$ = sample index,
$t$ = position; "+" adds $P$'s row $t$ to every sample.) The result, the
[[input-embeddings]], is what the transformer consumes (book p. 47, Fig. 2.19 p. 48).

### Inside the GPT model (ch. 4)
```python
self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])        # 50257 x 768
self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])    # 1024 x 768
...
tok_embeds = self.tok_emb(in_idx)                                     # [B, T, 768]
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # [T, 768]
x = tok_embeds + pos_embeds
```
(Listing 4.1, book pp. 96–97; Listing 4.7, book p. 119.) Fig. 4.4 (book p. 98): "Every effort moves you" → IDs
`6109, 3626, 6100, 345` → 768-dimensional token embeddings → GPT model; "the embedded input
dimension typically matches the output dimension". Inside each [[transformer-block]] every token is
still a 768-dimensional row (Fig. 4.13, book p. 114). In ch. 3, the attention examples start from
already-embedded tokens: "Your journey starts with one step" as six 3-D vectors, e.g.
"Your" = $[0.43, 0.15, 0.89]$, "journey" = $[0.55, 0.87, 0.66]$ (book p. 57).

### Weight tying and pretrained weights
- GPT-2 **reuses the token-embedding matrix as the output layer**: `tok_emb.weight` and
  `out_head.weight` both have shape `[50257, 768]` (book p. 121). The untied model has 163,009,536
  parameters (book p. 120); counting the shared matrix once gives 163,009,536 − 38,597,376 =
  **124,412,160**, the "124M" of GPT-2 (book p. 121). See
  [[weight-tying]].
- OpenAI's checkpoint stores it as `params["wte"]` (the token-embedding weights; reading "wte" as
  "word/weight token embedding" is our gloss), shape `(50257, 768)` (book pp. 162–163);
  loading does `gpt.tok_emb.weight ← wte` and `gpt.out_head.weight ← wte` (Listing 5.5, book pp. 165–166). See
  [[pretrained-weights]].
- In ch. 6 classification fine-tuning, the embedding layers are part of the frozen pretrained body;
  only the last transformer block, final LayerNorm and the new 2-class head are trained (§6.5,
  book p. 186); the input is still "text converted into token IDs" (book p. 202). See [[layer-freezing]].

## 🔴 Theory (Expert)
- **Linear map on basis vectors.** The embedding is the [[linear-mapping]]
  $\mathbb{R}^V\to\mathbb{R}^d$, $\mathbf{x}\mapsto W^\top\mathbf{x}$, evaluated only on the standard basis
  $\{\mathbf{x}_i\}$. Its image is the row space of $W$, so at most $\min(V,d)$ independent directions
  exist — with $V=50{,}257 \gg d=768$, tokens necessarily share directions.
- **Sparse gradients.** With $E = XW$ for a batch of one-hot rows $X\in\{0,1\}^{n\times V}$,
  $\partial L/\partial W = X^\top\,\partial L/\partial E$: only rows that were looked up get a nonzero
  gradient in a step (our derivation; the book only says the layer is trainable by backpropagation,
  book p. 43). Rare tokens are therefore updated rarely.
- **Why "continuous" matters.** The book's reason for embeddings: GPT-like LLMs are "deep neural
  networks trained with the backpropagation algorithm" (book p. 42), which needs real-valued
  parameters. An integer ID has no derivative; a row of $W$ does. See [[backpropagation]].
- **Static vs. contextual.** A token embedding is *context-free*: the same ID always yields the same
  vector, which "is good for reproducibility purposes" but carries no position or context
  (book pp. 43–45; Fig. 2.17). Attention turns these into *contextualized* vectors — the
  [[context-vector]] $z^{(i)}$ is an "enriched embedding vector" (book p. 57). The book also notes
  LLMs "can also create contextualized output embeddings" (book p. 20).
- **Learned in-model vs. pretrained Word2Vec.** "LLMs commonly produce their own embeddings that are
  part of the input layer and are updated during training", so they are "optimized to the specific
  task and data at hand" (book p. 20). Contrast [[word2vec]] / [[word-embeddings]].
- **Initialization.** PyTorch's default `nn.Embedding` init is $\mathcal{N}(0,1)$ (external fact; the
  book says only "small, random values"), which explains entries like $-2.84$.
- **Tying as an inner product.** With tied weights, $\text{logit}_j = \langle \mathbf{h}, W_{j,:}\rangle$ —
  the score for token $j$ is the dot product ([[inner-product]]) of the final hidden state with token $j$'s own
  embedding, so the same geometry serves input and output.

## Connections
- Builds on: [[token-id]], [[vocabulary]], [[vector]], [[vector-space]], [[matrix]], [[one-hot-encoding]]
- Leads to: [[positional-embedding]], [[input-embeddings]], [[self-attention]], [[context-vector]], [[logits]]
- Used by: [[embedding-layer]], [[gpt-architecture]], [[weight-tying]], [[pretrained-weights]], [[classification-fine-tuning]]
- Related: [[word-embeddings]], [[word2vec]], [[embedding-dimension]], [[broadcasting]], [[backpropagation]], [[linear-mapping]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1 (learned vs. pretrained embeddings, dimensions), book pp. 18–20; §2.2 (Fig. 2.4 pipeline), book p. 21; §2.3 (IDs as the bridge to embeddings), book p. 24.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7 (Fig. 2.15, `nn.Embedding`, $6\times3$ weight matrix, lookup, one-hot note, Fig. 2.16), book pp. 41–44; §2.8 (Fig. 2.17, $8\times4\times256$ embeddings, broadcast addition, Fig. 2.19, summary), book pp. 43–49.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1 (embeddings as LLM input; hidden state as embedding), book pp. 50, 53; §3.3.1 (embedded input tensor, context vectors), book p. 57.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1 (`GPT_CONFIG_124M`, Listing 4.1, Fig. 4.4), book pp. 95–98; §4.5 (768-D rows in the transformer block, Fig. 4.13), book pp. 113–114; §4.6 (Listing 4.7 GPTModel, 163,009,536 params, weight tying, 124,412,160), book pp. 117–121; §6.5 (freezing), book p. 186.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5 (`wte` shape, Listing 5.5 weight loading), book pp. 162–166.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.4 (loading GPT-2 for fine-tuning), book pp. 181–183.
