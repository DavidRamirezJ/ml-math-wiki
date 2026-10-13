---
title: Bahdanau Attention
type: concept
tags: [attention, deep-learning, rnn, machine-translation, nlp]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Bahdanau Attention

> **One-liner:** Bahdanau attention (2014) let a translating RNN look back at every input word at each output step, instead of relying on one squashed summary — the idea that inspired the transformer's self-attention.

## 🟢 Intuition (Dummy)
Picture a human translator. A bad one reads the German sentence once, closes the
book, and translates from memory. A good one keeps the original open and, for each
English word they write, glances back at the German words that matter most right
then. Bahdanau attention turned the bad translator into the good one.

Before it, an encoder–decoder RNN had to squeeze the whole input into one
[[hidden-state]] (see [[long-range-dependencies]]). Bahdanau and colleagues said:
keep *all* the encoder's states, and at every output step, let the decoder decide
how much to "look at" each input word. In the book's picture (Fig. 3.5), when
generating "you" from "Kannst du mir", the line to "du" is thick and the others thin.

## 🟡 Mechanics (Practitioner)
**What the book says.** In 2014 researchers developed the **Bahdanau attention
mechanism** for RNNs, named after the first author of the paper (book p. 54). It
modifies the encoder–decoder RNN so that the decoder **can selectively access
different parts of the input sequence at each decoding step** (book p. 54). Some
input tokens matter more than others for a given output token; the importance is set
by **[[attention-weights]]**, which the book computes later (Fig. 3.5 caption, book
p. 54). Figure 3.5 shows the general idea, not the exact Bahdanau implementation,
which is an RNN method outside the book's scope (book p. 54). Only three years later
the transformer showed RNNs aren't needed and introduced a **self-attention**
mechanism *inspired by* Bahdanau's (book pp. 54–55).

**Bahdanau vs. self-attention.** The book's sidebar "The 'self' in self-attention"
(book p. 56) draws the line: *traditional* attention relates elements of **two
different sequences** — input and output in a sequence-to-sequence model, as in
Fig. 3.5 — while [[self-attention]] relates positions **within one sequence**.

**The original equations** (background from Bahdanau, Cho & Bengio 2014 — the book
does not print them). For output step $i$ with previous decoder state
$\mathbf s_{i-1}$ and encoder states $\mathbf h_1,\dots,\mathbf h_T$:
$$e_{ij}=\mathbf v^\top\tanh\!\big(W\mathbf s_{i-1}+U\mathbf h_j\big),\qquad
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k=1}^{T}\exp(e_{ik})},\qquad
\mathbf c_i=\sum_{j=1}^{T}\alpha_{ij}\,\mathbf h_j$$

Symbol by symbol:
- $\mathbf s_{i-1}$ = the decoder's hidden state just before writing output word $i$ ("where am I in the translation?").
- $\mathbf h_j$ = the encoder's hidden state at input word $j$ ("what does input word $j$ mean in context?").
- $W$, $U$ = learned weight [[matrix|matrices]] that project both into a common space.
- $\tanh$ = squashing nonlinearity; $\mathbf v^\top$ = a learned vector that turns the result into one number.
- $e_{ij}$ = the **alignment score** ([[attention-scores|attention score]]): how well input $j$ fits output step $i$.
- $\alpha_{ij}$ = the [[softmax]] of the scores over $j$ → [[attention-weights]], all $\ge 0$ and summing to 1.
- $\mathbf c_i$ = the [[context-vector]]: a weighted average ([[convex-combination]]) of encoder states, fed to the decoder for step $i$.

Because the score is a tiny feed-forward net ($\tanh$ of a sum), this is called
**additive attention**. The transformer swaps it for a [[inner-product|dot-product]]-based score
$\mathbf q^\top\mathbf k/\sqrt{d_k}$ — [[scaled-dot-product-attention]].

**Toy numbers (illustrative, not from the book).** Generating "you" from
"Kannst du mir", suppose $e = (0.5,\ 2.0,\ 0.3)$. Then
$\alpha=\operatorname{softmax}(e)\approx(0.16,\ 0.71,\ 0.13)$, and
$\mathbf c\approx 0.16\,\mathbf h_{\text{Kannst}}+0.71\,\mathbf h_{\text{du}}+0.13\,\mathbf h_{\text{mir}}$ —
mostly "du", as the thick line in Fig. 3.5 suggests.

## 🔴 Theory (Expert)
- **Additive vs. multiplicative.** Additive scoring
  $\mathbf v^\top\tanh(W\mathbf s+U\mathbf h)$ is a one-hidden-layer MLP on the
  pair; multiplicative (Luong) scoring is a bilinear form $\mathbf s^\top M\mathbf h$;
  transformer scoring factors $M=W_qW_k^\top$ so that
  $\mathbf q^\top\mathbf k=(\mathbf x^\top W_q)(W_k^\top\mathbf x')$ — an
  [[inner-product]] in a learned low-rank space, computable for all pairs with one
  matrix product $QK^\top$.
- **Soft alignment.** $\alpha_{i\cdot}$ is a [[probability-distribution]] over input
  positions — a differentiable relaxation of a hard word-alignment variable, trained
  end-to-end by backprop with no alignment labels (hence the paper title "Jointly
  Learning to Align and Translate").
- **Still recurrent.** Encoder and decoder remain RNNs (the paper used a
  bidirectional encoder), so computation is sequential in $T$; attention removes the
  information bottleneck but not the sequential cost. The transformer removed both.
- **Complexity.** $O(T_{\text{out}}\cdot T_{\text{in}})$ scores, the same quadratic
  pattern that transformer self-attention inherits.

## Connections
- Builds on: [[hidden-state]], [[recurrent-neural-network]], [[encoder-decoder-architecture]], [[long-range-dependencies]], [[softmax]]
- Leads to: [[attention-mechanism]], [[self-attention]], [[scaled-dot-product-attention]], [[transformer-architecture]]
- Used by: [[machine-translation]]
- Related math: [[convex-combination]], [[inner-product]], [[probability-distribution]]
- People: [[dzmitry-bahdanau]], [[yoshua-bengio]] (co-author with Kyunghyun Cho), [[ashish-vaswani]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.2, book pp. 54–55 (Fig. 3.5); §3.3 sidebar "The 'self' in self-attention", book p. 56; appendix B, Ch. 3 ref. 1 (Bahdanau, Cho & Bengio 2014, arXiv 1409.0473), book p. 291.
- The scoring equations and toy numbers are background from the original paper / illustrative, not printed in the book.
