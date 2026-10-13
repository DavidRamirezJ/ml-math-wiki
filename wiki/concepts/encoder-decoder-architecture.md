---
title: Encoder-Decoder (Seq2Seq) Architecture
type: concept
tags: [deep-learning, nlp, transformer, rnn, sequence-modeling]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Encoder-Decoder (Seq2Seq) Architecture

> **One-liner:** One network reads the whole input sentence and turns it into numbers (the encoder), and a second network turns those numbers into the output sentence one word at a time (the decoder).

## 🟢 Intuition (Dummy)
Think of a human interpreter at a meeting. First they **listen** to the whole German sentence and
form an idea of what it means. Then they **speak** the English version, one word after another,
checking what they've already said as they go.

An **encoder-decoder** model splits the work the same way:
- the **encoder** is the listener. It reads the full input and writes its meaning down as a list
  of numbers ([[vector|vectors]]).
- the **decoder** is the speaker. It reads those numbers and produces the output word by word.

Why not just translate word by word? Because languages put words in different orders. The German
"Kannst du mir helfen diesen Satz zu uebersetzen", translated word by word, gives "Can you me help
this sentence to translate", which is wrong. The right answer is "Can you help me to translate this
sentence" (book p. 52). Some output words need information from words that sit *earlier or later*
in the input, so the model has to see the whole input first.

This setup is also called **sequence-to-sequence (seq2seq)**: one sequence goes in, another comes out.

## 🟡 Mechanics (Practitioner)

### The general shape
Given a source sequence $\mathbf x = (x_1,\dots,x_S)$ and a target sequence $\mathbf y = (y_1,\dots,y_T)$:

$$\mathbf H = \mathrm{Enc}(\mathbf x),\qquad
p(\mathbf y\mid \mathbf x)=\prod_{t=1}^{T} p\big(y_t \mid y_1,\dots,y_{t-1},\ \mathbf H\big)$$

(Standard notation; the book describes this in words and figures.) Symbol by symbol:
- $\mathbf x$ = the input (source) tokens; $S$ = their count. $\mathbf y$ = the output (target) tokens; $T$ = their count.
- $\mathrm{Enc}$ = the encoder; $\mathbf H$ = what it outputs, the "encoded" vectors that hold the input's context.
- $\prod_{t=1}^{T}$ = multiply over every output position $t$.
- $p(y_t\mid y_1,\dots,y_{t-1},\mathbf H)$ = the decoder's probability for the next output word, **given**
  the words it has already produced *and* the encoder's vectors. This is what "one word at a time"
  means ([[autoregressive-text-generation]]).

### Version 1: the RNN encoder-decoder (before transformers)
Before transformers, [[recurrent-neural-network|RNNs]] were the most popular encoder-decoder for translation
(book p. 52). An RNN feeds each step's output back in as input to the next step. A standard
recurrence (background, not printed in the book) is

$$\mathbf h_t = f\big(W_{hh}\,\mathbf h_{t-1} + W_{xh}\,\mathbf x_t + \mathbf b\big)$$

- $\mathbf x_t$ = the embedding of input token $t$; $\mathbf h_t$ = the [[hidden-state]] after reading it;
  $\mathbf h_{t-1}$ = the previous hidden state.
- $W_{hh}, W_{xh}$ = weight [[matrix|matrices]]; $\mathbf b$ = a bias vector; $f$ = a nonlinearity such as $\tanh$.

How it works (book p. 53, Figure 3.4):
1. The encoder reads the input **sequentially** and updates its hidden state at each step.
2. It tries to squeeze the **entire meaning** of the sentence into the **final hidden state**, a
   "memory cell" (Figure 3.4: "Kannst du mir" → memory cell → "Can you help").
3. The decoder starts from that final hidden state and generates the translation one word at a time,
   updating its own hidden state as it goes.

You can think of that final hidden state as one [[word-embeddings|embedding vector]] for the whole sentence (book p. 53).

**The big flaw.** While decoding, the RNN **can't directly look back at the encoder's earlier
hidden states**. Everything must fit into the current hidden state, so context gets lost, especially
when "dependencies span long distances" (book p. 53). This flaw is what motivated
[[attention-mechanism|attention]] (book p. 54). [[bahdanau-attention]] (2014) let the decoder look at
*all* the encoder states. The transformer then replaced the recurrence with [[self-attention]].

### Version 2: the original transformer (2017)
"Attention Is All You Need" (Vaswani et al., 2017, arXiv 1706.03762) introduced the
[[transformer-architecture]] for **machine translation**, English → German and English → French
(book p. 7). It keeps the two-part idea (Figure 1.4, book p. 8):
- **Encoder:** has access to the **complete input text**. It turns it into a sequence of
  context-rich vectors ("embeddings").
- **Decoder:** takes those vectors plus the partial translation and generates the next word.
  Example: input "This is an example", partial output "Das ist ein", the decoder fills in
  **"Beispiel"**.
- Both have **many layers** connected by [[self-attention]] (book p. 7). The encoder and decoder
  blocks are each repeated **6 times** (book p. 13).

### Seq2seq vs. self-attention
The book's sidebar "The 'self' in self-attention" (book p. 56) makes the distinction clear:
- **Traditional attention** relates elements of **two different sequences**, as in seq2seq
  (encoder-decoder) models where attention runs between the input and the output sequence.
- **Self-attention** relates positions **within one sequence**.

### What became of the two halves
- **Encoder only** → [[bert]] (masked word prediction, good at classification).
- **Decoder only** → GPT ([[gpt-architecture]], [[decoder-only-transformer]]): simpler, no encoder,
  trained on next-word prediction (book pp. 8–9, 12).

## 🔴 Theory (Expert)
- **Information bottleneck.** The RNN encoder maps a variable-length input into a single fixed-size
  vector $\mathbf h_S\in\mathbb R^{d}$. However long the sentence, the capacity stays the same, so
  this is a lossy compression (compare the intuition behind [[dimensionality-reduction]]). Attention
  removes the bottleneck: the decoder's context becomes a data-dependent [[convex-combination]]
  $\mathbf c_t=\sum_s \alpha_{ts}\mathbf h_s$ of *all* encoder states, with weights $\alpha_{ts}$ from a
  [[softmax]].
- **Gradient path length.** In an RNN, a signal from input position $s$ has to pass through
  $O(S)$ recurrent steps to reach the output, and gradients get multiplied by $O(S)$ Jacobians
  ([[vanishing-and-exploding-gradients]], [[chain-rule]]). Attention connects any two positions in
  one step. (Background reasoning; the book states only that context is lost.)
- **Parallelism.** The RNN encoder is inherently sequential ($\mathbf h_t$ needs $\mathbf h_{t-1}$).
  The transformer encoder processes all positions at once with matrix multiplications, which is a
  big reason it scaled.
- **Shape preservation.** A transformer block maps $\mathbb R^{T\times d}\to\mathbb R^{T\times d}$. The
  book calls this "not incidental but a crucial aspect of its design": it is what makes transformers
  usable "across a wide range of sequence-to-sequence tasks," with each output vector matched
  one-to-one to an input vector while its content is re-encoded with context from the whole sequence
  (book p. 116).
- **Encoder-decoder is a special case.** A decoder-only model can do seq2seq tasks by putting source
  and target in one stream ("Translate English to German: breakfast =>", book p. 10). The
  conditional $p(\mathbf y\mid\mathbf x)$ becomes the continuation probability $p(\mathbf y\mid\text{prompt})$
  under one autoregressive model. That's why GPT models can translate even though they weren't
  designed to (book pp. 13–14, [[emergent-behavior]]).

## Connections
- Builds on: [[recurrent-neural-network]], [[hidden-state]], [[word-embeddings]], [[vector]], [[matrix]], [[probability-distribution]]
- Leads to: [[bahdanau-attention]], [[attention-mechanism]], [[self-attention]], [[transformer-architecture]], [[bert]], [[gpt-architecture]]
- Used by: [[machine-translation]], [[natural-language-processing]]
- Related: [[long-range-dependencies]], [[decoder-only-transformer]], [[autoregressive-text-generation]]
- People: [[ashish-vaswani]] (transformer), [[dzmitry-bahdanau]], [[yoshua-bengio]] (attention for RNN translation)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 7–10 (original transformer: encoder + decoder, Figure 1.4, BERT vs. GPT, Figure 1.5).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book pp. 12–13 (GPT drops the encoder; 6× encoder/decoder blocks in the original).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1–3.2, book pp. 52–54 (word-by-word translation fails, Figure 3.3; RNN encoder-decoder and hidden state, Figure 3.4; the long-sequence limitation; Bahdanau attention, 2014).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3, book pp. 55–56 (sidebar: self-attention vs. attention between two sequences in seq2seq models).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book p. 116 (shape preservation enables sequence-to-sequence tasks).
