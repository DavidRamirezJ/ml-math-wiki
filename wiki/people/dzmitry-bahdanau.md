---
title: Dzmitry Bahdanau
type: person
tags: [attention, machine-translation, rnn, deep-learning]
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Dzmitry Bahdanau (active 2010s–present)

> **Why they matter to ML:** first author of the 2014 paper that invented the attention mechanism for neural translation — the idea that, three years later, inspired the transformer's self-attention and so every modern LLM.

## Life (brief)
Dzmitry Bahdanau is a Belarusian computer scientist. He did his PhD at the
Université de Montréal (Mila) under [[yoshua-bengio]], where the attention paper was
written (he was then an early-career researcher, coming from Jacobs University Bremen). He later worked in industry AI research (Element AI,
then ServiceNow Research) and as an adjunct professor at McGill University.
(Biographical details are general knowledge. The book names him only through the
mechanism's name and the appendix B citation "Bahdanau, Cho, and Bengio".)

## Key contributions
- **Bahdanau attention** (Bahdanau, Cho & Bengio 2014, "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 1409.0473). → [[bahdanau-attention]], [[attention-mechanism]]
  - **The problem it fixed (book p. 54).** An encoder–decoder [[recurrent-neural-network]] "must remember the entire encoded input in a single hidden state before passing it to the decoder". Long sentences get squashed. → [[hidden-state]], [[long-range-dependencies]], [[encoder-decoder-architecture]]
  - **The fix (book p. 54).** The mechanism "modifies the encoder–decoder RNN such that the decoder can selectively access different parts of the input sequence at each decoding step." It is "named after the first author of the respective paper." Figure 3.5 shows the general idea with the German "Kannst du mir" → "Can you help". The caption warns that the figure "does not depict the exact implementation of the Bahdanau mechanism, which is an RNN method outside this book's scope"; the line widths show how important each input token is for the output token.
  - **The math** (from the paper, not the book). When producing output word $i$, score every input position $j$:
    $$e_{ij} = \mathbf{v}_a^\top \tanh\!\big(W_a\,\mathbf{s}_{i-1} + U_a\,\mathbf{h}_j\big),\qquad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k}\exp(e_{ik})},\qquad \mathbf{c}_i = \sum_j \alpha_{ij}\,\mathbf{h}_j.$$
    Symbol by symbol: $\mathbf{s}_{i-1}$ = the decoder's previous hidden state; $\mathbf{h}_j$ = the encoder's hidden state for input word $j$; $W_a, U_a, \mathbf{v}_a$ = learned weights; $\tanh$ = a squashing [[activation-function]]; $e_{ij}$ = an "alignment" score ([[attention-scores]]); $\alpha_{ij}$ = the [[softmax]] of the scores ([[attention-weights]], they add up to 1 over $j$); $\mathbf{c}_i$ = the [[context-vector]], a weighted average ([[convex-combination]]) of the encoder states. Because the score uses a small feed-forward net instead of a dot product, this is called **additive attention**.
  - **Its legacy (book pp. 54–55).** "Only three years later, researchers found that RNN architectures are not required for building deep neural networks for natural language processing and proposed the original transformer architecture … with a self-attention mechanism inspired by the Bahdanau attention mechanism." The transformer swapped additive scores for [[scaled-dot-product-attention]] and pointed attention at the *same* sequence. → [[self-attention]], [[transformer-architecture]]
- **Later work** (general knowledge, not in the book): co-author with Kyunghyun Cho et al. of early work on RNN encoder–decoders and GRUs, and work on training sequence models, systematic generalization, and code LLMs (e.g. the BigCode / StarCoder project).

## Connections
- Concepts: [[bahdanau-attention]], [[attention-mechanism]], [[machine-translation]], [[encoder-decoder-architecture]], [[recurrent-neural-network]], [[hidden-state]], [[context-vector]], [[attention-weights]], [[self-attention]]
- People: [[yoshua-bengio]] (co-author and PhD advisor), [[ashish-vaswani]] (transformer, inspired by this work)
- Appears in sources: [[llm-ch03-coding-attention-mechanisms]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.2, book pp. 54–55 (Bahdanau attention, 2014; named after the first author; Figure 3.5; inspired the transformer's self-attention).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` end of §3.2 (continuing on p. 55, just before §3.3 begins), book p. 55 (self-attention "inspired by the Bahdanau attention mechanism").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1, book pp. 52–53 (the encoder–decoder RNN and its single-hidden-state bottleneck, Figures 3.3–3.4, that motivated the mechanism).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 3 references, book p. 291 (Bahdanau, Cho & Bengio 2014, arXiv 1409.0473).
- Biography, the additive-attention equations, and later work are general knowledge / from the original paper, not from the book.
