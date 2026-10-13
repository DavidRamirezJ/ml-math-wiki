---
title: Word Embeddings
type: concept
tags: [nlp, llm, embeddings, linear-algebra, representation-learning]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Word Embeddings

> **One-liner:** A word embedding turns each word (or token) into a list of numbers, so that a neural network can do math with it, and so that words with similar meanings end up as nearby points.

## 🟢 Intuition (Dummy)
A neural network only understands numbers. It can multiply, add and take
gradients, but it has no idea what to do with the word "duck". Words are
*categorical*: they are labels, like jersey numbers, not quantities.

So we give every word a set of coordinates, like a pin on a map. "Duck",
"eagle" and "goose" get pins close to each other. "Germany" and "Berlin" are
near each other too, but far away from the birds. "Squirrel" sits close to the
birds (it is an animal), but just outside their group (it is not a bird). That
"map" is the **embedding space**, and the coordinates of each word are its
**word embedding**.

The fun part: in an LLM, nobody draws this map by hand. The model starts with
random pins and slowly moves them around during training, until the
positions help it predict the next word well.

Embeddings are not only for words. You can also embed video, audio, images,
sentences or whole documents. Each data type needs its own embedding model:
a text embedding model cannot embed audio (book p. 18).

## 🟡 Mechanics (Practitioner)

### The definition
The book's definition: *an embedding is a mapping from discrete objects (words,
images, or whole documents) to points in a continuous vector space*; its job is
to turn non-numeric data into something a neural network can process (book
p. 19). Written as a function:

$$E : \mathcal{V} \to \mathbb{R}^{d}, \qquad w \mapsto \mathbf{e}_w$$

Reading it symbol by symbol:
- $E$ = the embedding function (the "map maker").
- $\mathcal{V}$ = the vocabulary, the finite set of all words/tokens the model knows.
- $\to$ = "maps into".
- $\mathbb{R}^{d}$ = the set of all lists of $d$ real numbers, i.e. a [[vector-space]] of dimension $d$. The number $d$ is the [[embedding-dimension]].
- $w \mapsto \mathbf{e}_w$ = "the word $w$ is sent to the [[vector]] $\mathbf{e}_w$".

(This notation is our formalisation of the book's sentence; the book does not print it.)

**Figure 2.2** (book p. 19) shows three toy embedding models, each turning raw
input into a 3-dimensional vector ($d = 3$):

| Input | Embedding model | Example vector |
|---|---|---|
| Video sample | video embedding model | $[1.23,\ -0.31,\ 0.89]$ |
| Audio sample | audio embedding model | $[-0.15,\ 0.45,\ 2.11]$ |
| Text sample | text embedding model | $[1.78,\ 0.18,\ -2.10]$ |

### Kinds of text embeddings
- **Word embeddings** are the most common (book p. 19).
- There are also **sentence, paragraph and whole-document** embeddings.
  Sentence/paragraph embeddings are popular for
  [[retrieval-augmented-generation]] (book p. 19).
- GPT-like LLMs generate text one word (token) at a time, so the book focuses on
  word-level (really token-level) embeddings.

### Two ways to get them
1. **Pretrained, separate model**, e.g. [[word2vec]] (book pp. 19–20).
2. **Learned inside the LLM.** LLMs usually make their own embeddings: they are
   part of the input layer and are updated during training. The advantage is
   that they get optimised for *this* task and *this* data (book p. 20).

### "Close means similar"
Figure 2.3 (book p. 20) plots 2-D word embeddings: birds (duck, eagle, goose)
cluster together, "squirrel" sits just outside them, countries sit near their
capitals (Germany–Berlin, England–London), and "long / longer / longest" form
their own group. The book doesn't give a formula for "close", but the usual two
are (our addition):

$$\text{dist}(a,b)=\lVert \mathbf{e}_a-\mathbf{e}_b\rVert_2,\qquad
\cos\theta_{ab}=\frac{\mathbf{e}_a^{\top}\mathbf{e}_b}{\lVert\mathbf{e}_a\rVert\,\lVert\mathbf{e}_b\rVert}$$

- $\mathbf{e}_a,\mathbf{e}_b$ = the embedding vectors of words $a$ and $b$.
- $\lVert\cdot\rVert_2$ = the Euclidean length ([[norm]]); a small distance means "near on the map".
- $\mathbf{e}_a^{\top}\mathbf{e}_b$ = the dot product ([[inner-product]]): multiply matching coordinates and add.
- $\cos\theta_{ab}$ = cosine of the angle between the two arrows ([[angle-between-vectors]]); close to 1 means "pointing the same way" = similar meaning.

### How an LLM stores them: the embedding layer
Inside an LLM the embeddings are rows of a weight [[matrix]]
$W \in \mathbb{R}^{|\mathcal{V}|\times d}$ (§2.7). Looking up token ID $i$ just
grabs row $i$:

$$\text{Embedding}(i) = W_{i,:} = \mathbf{x}_i^{\top} W$$

- $i$ = the [[token-id]] (an integer, counting from 0).
- $W_{i,:}$ = "row $i$, all columns" of $W$.
- $\mathbf{x}_i$ = the [[one-hot-encoding|one-hot]] vector for $i$: length $|\mathcal{V}|$, all zeros except a 1 in slot $i$.
- $\mathbf{x}_i^{\top} W$ = multiplying that one-hot row by $W$ — which picks out exactly row $i$.

The book's NOTE: the embedding layer is just a more efficient way of doing
one-hot encoding followed by a matrix multiplication in a fully connected
layer, so it can be trained by [[backpropagation]] (book p. 43).

**Worked example (book pp. 42–44).** Vocabulary of 6 tokens, $d = 3$:

```python
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(6, 3)   # vocab_size=6, output_dim=3
print(embedding_layer(torch.tensor([2, 3, 5, 1])))
```

The random starting weight matrix is
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
and the IDs `[2, 3, 5, 1]` ("fox jumps over dog") return rows 2, 3, 5, 1 — e.g.
ID 3 → `[-0.4015, 0.9666, -1.1481]`, the *fourth* row because Python counts
from 0 (Fig. 2.16).

**At real scale:** GPT-2 small's token-embedding matrix `wte` has shape
$(50257, 768)$ — one 768-number row per [[vocabulary]] entry (§5.5, book p. 163).

### The pipeline that produces them (book p. 20)
The book lists the steps to prepare an LLM's embeddings: (1) split text into
words, (2) convert words into tokens ([[tokenization]], [[token-id]]s),
(3) turn tokens into embedding vectors. The embedding weights start as small
random values and are optimised during LLM training in Ch. 5 (book p. 42).

### Where they go next
Token embeddings carry no information about *where* a word sits in the
sentence, so a [[positional-embedding]] is added before the transformer
(§2.8). After the attention layers, the model produces **contextualised**
output embeddings ([[context-vector]]s, Ch. 3) — the same word gets a
different vector depending on its neighbours (book p. 20).

Ch. 1 already uses the idea: in the original encoder–decoder transformer
(Fig. 1.4, book p. 8) the encoder turns the input text into "an embedding
representation (a numerical representation that captures many different
factors in different dimensions)" that the decoder uses to generate the
translation one word at a time (book pp. 7–8).

## 🔴 Theory (Expert)
- **Why continuous?** Training uses [[gradient-descent]], which needs to nudge
  parameters by tiny amounts. Integers like token IDs can't be nudged
  smoothly; real-valued vectors can. The book: embeddings are needed because
  LLMs are deep networks trained with backpropagation (book p. 42).
- **Static vs contextual.** A lookup table gives one fixed vector per token
  (static, Word2Vec-style). The same ID always maps to the same vector,
  regardless of position (Fig. 2.17, book p. 44). Context-dependence only comes
  later, from [[self-attention]]: the output at position $t$ is a weighted mix of
  all earlier tokens' vectors.
- **The lookup is a linear map.** $\mathbf{x}\mapsto\mathbf{x}^{\top}W$ is a
  [[linear-mapping]] from the one-hot space $\mathbb{R}^{|\mathcal{V}|}$ to
  $\mathbb{R}^{d}$. Its image is at most $d$-dimensional, so $|\mathcal{V}|=50{,}257$
  one-hot directions are compressed into 768 dimensions — a learned
  low-dimensional representation (cf. [[dimensionality-reduction]]).
- **Sparse gradients** (our observation). Because only the looked-up rows take
  part in the forward pass, only those rows get non-zero gradients in a step.
  The book shows the output is differentiable (`grad_fn=<EmbeddingBackward0>`,
  book p. 43) but does not discuss sparsity.
- **Weight tying.** The original GPT-2 reuses the token-embedding weights in
  the output layer to reduce the total number of parameters (book p. 166; see
  [[weight-tying]]). Consequence (our reading, not stated in the book): the
  logit for token $j$ is the dot product $\langle\mathbf{h},\mathbf{w}_j\rangle$
  between the final hidden state and token $j$'s embedding. One matrix then plays both "words → points" and
  "points → word scores".
- **Visualisation needs projection.** Real embeddings have hundreds or
  thousands of dimensions; humans see at most 3, so plots like Fig. 2.3 rely on
  a projection to 2-D (book p. 20), e.g. [[principal-component-analysis]].
- Further reading the book recommends: Raschka, *Machine Learning Q and AI*
  (2023), ch. 1, on embedding spaces vs latent spaces (book p. 290).

## Connections
- Builds on: [[vector]], [[vector-space]], [[matrix]], [[linear-mapping]], [[one-hot-encoding]], [[tokenization]], [[token-id]]
- Leads to: [[token-embedding]], [[embedding-layer]], [[positional-embedding]], [[input-embeddings]], [[context-vector]], [[self-attention]]
- Related: [[word2vec]], [[embedding-dimension]], [[inner-product]], [[norm]], [[angle-between-vectors]], [[dimensionality-reduction]], [[weight-tying]], [[retrieval-augmented-generation]]
- Used by: [[large-language-model]], [[gpt-architecture]], [[transformer-architecture]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 7–8 (embedding representation in the encoder, Fig. 1.4).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book pp. 18–20 (definition, Figs. 2.2–2.3, Word2Vec, dimensionality, learned vs pretrained).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7, book pp. 41–44 (embedding layer, 6×3 example, one-hot NOTE, Fig. 2.16).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book pp. 43–45 (Fig. 2.17, position-agnostic lookups).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 163–166 (`wte` shape, weight tying).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book p. 290 (further reading).
