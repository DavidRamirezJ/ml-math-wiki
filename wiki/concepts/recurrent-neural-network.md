---
title: Recurrent Neural Network (RNN)
type: concept
tags: [deep-learning, sequence-models, neural-network, nlp]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Recurrent Neural Network (RNN)

> **One-liner:** An RNN reads a sequence one item at a time and carries a running "memory" vector forward, feeding each step's output back in as input to the next step.

## 🟢 Intuition (Dummy)
Imagine reading a sentence through a tiny keyhole that shows one word at a time.
To understand it, you keep a mental note that you update after each word: "so far:
someone is asking me for help…". That note is the RNN's **hidden state**. Each new
word plus the old note produces a new note.

The book defines an RNN as a neural network in which **outputs from previous steps are
fed as inputs to the current step**, which makes it a natural fit for sequential data
like text (book pp. 52–53). Before transformers, RNNs were the **most popular
encoder–decoder architecture for language translation** (book p. 52). The book's
**hidden state** is "the internal values at the hidden layers", updated at every step
(book p. 53). The book also stresses that you do not need RNN details to build an LLM —
only the shortcoming that motivated attention (book p. 54).

The weak spot: everything must fit in that one note. In translation, the encoder RNN
tries to pack the meaning of the *whole* sentence into its final note, and the
decoder has to translate from that note alone. For long, complex sentences, early
details fade — the decoder "cannot directly access earlier hidden states" (book
p. 53). This weakness is exactly what motivated the [[attention-mechanism]]
(book p. 54).

## 🟡 Mechanics (Practitioner)

### The recurrence
The book deliberately gives **no RNN formulas** ("you don't need to know the inner
workings", book p. 53). The standard (Elman) RNN step, added here for the math link, is:

$$\mathbf h_t = f\big(W_{hh}\,\mathbf h_{t-1} + W_{xh}\,\mathbf x_t + \mathbf b\big)$$

Symbol by symbol:
- $t$ = the time step (position in the sequence: word 1, word 2, …);
- $\mathbf x_t$ = the input embedding of the word at step $t$;
- $\mathbf h_{t-1}$ = the previous [[hidden-state]] (the "note" so far); $\mathbf h_0$ is usually all zeros;
- $W_{hh}$ = a weight [[matrix]] that transforms the old memory; $W_{xh}$ = a weight matrix that transforms the new input; $\mathbf b$ = a [[bias-vector]];
- $f$ = a nonlinearity, typically $\tanh$;
- $\mathbf h_t$ = the new hidden state.

The **same** $W_{hh}, W_{xh}, \mathbf b$ are reused at every step (weight sharing over time).

### Encoder–decoder RNN for translation (Fig. 3.4, book p. 53)
1. The **encoder** reads the German input "Kannst du mir" token by token, updating its hidden state at each step.
2. Its **final hidden state** is meant to be a compressed representation of the entire input — a "memory cell". You can think of it as an embedding vector (book p. 53).
3. The **decoder** takes that state and generates "Can you help" **one word at a time**, updating its own hidden state at each step to carry the context for next-word prediction.

Compactly: $\mathbf c=\mathbf h_T^{\text{enc}}$ (one vector for the whole source), then
the decoder predicts $y_t$ from its own state, which starts from $\mathbf c$.

### The shortcoming, in the book's words
- RNNs work for short sentences but not for longer texts, since they have no direct
  access to earlier words in the input (book p. 54).
- During decoding, the model relies only on the **current** hidden state, which must
  hold all relevant information → **loss of context**, especially when dependencies
  span long distances (book p. 53). See [[long-range-dependencies]].

### What replaced (and fixed) it
- **2014 — Bahdanau attention:** keep the RNN, but let the decoder look at *all*
  encoder states at each step, weighted by attention (Fig. 3.5, book p. 54). See
  [[bahdanau-attention]].
- **2017 — the transformer:** "only three years later" researchers found RNNs are not
  required at all for deep NLP models; the transformer uses [[self-attention]] instead
  (book pp. 54–55).
- Not all LLMs are transformers: some modern LLMs use **recurrent** (or convolutional)
  architectures for better computational efficiency (book p. 10); Appendix B cites
  RWKV, "Reinventing RNNs for the Transformer Era" (Peng et al., 2023, book p. 290).

## 🔴 Theory (Expert)
- **Unrolling = a deep composition.** Unrolled over $T$ steps, an RNN is a depth-$T$
  network with tied weights: $\mathbf h_T = F(F(\cdots F(\mathbf h_0,\mathbf x_1)\cdots),\mathbf x_T)$.
  Training uses **backpropagation through time**, i.e. [[backpropagation]] on the
  unrolled graph (background, not in the book).
- **Vanishing / exploding gradients.** By the [[chain-rule]],
  $$\frac{\partial\mathbf h_T}{\partial\mathbf h_k}=\prod_{t=k+1}^{T}\frac{\partial\mathbf h_t}{\partial\mathbf h_{t-1}}=\prod_{t=k+1}^{T}\operatorname{diag}\!\big(f'(\cdot)\big)\,W_{hh}.$$
  If the largest singular value of these factors stays below 1, the product shrinks
  exponentially in $T-k$ (vanishing); above 1 it can blow up (exploding). Either way,
  signals from far-back words are hard to learn — the gradient-side reason for the
  long-distance failure the book describes. See [[vanishing-and-exploding-gradients]],
  [[singular-value-decomposition]], [[eigenvalues-and-eigenvectors]].
- **Fixed-capacity bottleneck.** $\mathbf h_T\in\mathbb R^d$ has fixed size while the
  input length $T$ is unbounded, so a lossy compression is unavoidable for long inputs
  (cf. [[dimensionality-reduction]], [[low-rank-approximation]]). Attention removes the
  bottleneck by keeping all $T$ states and reading them with data-dependent weights.
- **Path length and parallelism.** Information between positions $i$ and $j$ travels
  through $|i-j|$ recurrent steps in an RNN, but only one attention step in a
  transformer. RNN steps are also inherently **sequential** (step $t$ needs
  $\mathbf h_{t-1}$), while attention over all positions is a single matrix product
  that GPUs parallelize well (background).
- **Gated variants.** LSTM and GRU add multiplicative gates to ease gradient flow;
  they reduce but do not remove the bottleneck (background; the book does not cover
  them).

## Connections
- Builds on: [[neural-network]], [[vector]], [[matrix]], [[linear-mapping]], [[activation-function]], [[word-embeddings]]
- Leads to: [[bahdanau-attention]], [[attention-mechanism]], [[self-attention]], [[transformer-architecture]]
- Related: [[hidden-state]], [[encoder-decoder-architecture]], [[machine-translation]], [[long-range-dependencies]], [[vanishing-and-exploding-gradients]], [[backpropagation]], [[chain-rule]]
- Used by: encoder–decoder translation systems before 2017; recurrent LLMs such as RWKV
- People: [[dzmitry-bahdanau]], [[yoshua-bengio]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book p. 10 (some LLMs use recurrent architectures, "Transformers vs. LLMs" sidebar)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1, book pp. 52–54 (RNN definition; encoder–decoder RNN, Fig. 3.4; hidden state; loss of context)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.2, book pp. 54–55 (single-hidden-state bottleneck; Bahdanau 2014; transformer 2017)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book pp. 290–291 (RWKV, Peng et al. 2023; Bahdanau, Cho & Bengio 2014)
