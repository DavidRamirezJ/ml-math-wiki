---
title: Yoshua Bengio
type: person
tags: [deep-learning, attention, language-models, word-embeddings]
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Yoshua Bengio (1964– )

> **Why they matter to ML:** a founding figure of deep learning — co-author of the attention paper that led to transformers, of the first neural language model with learned word embeddings, and of the classic analysis of why RNNs struggle with long-range memory.

## Life (brief)
Yoshua Bengio was born in Paris in 1964 and grew up in Montréal. He is a professor
at the Université de Montréal and founded Mila, the Quebec AI institute. He shared
the 2018 ACM Turing Award with [[geoffrey-hinton]] and Yann LeCun "for conceptual
and engineering breakthroughs that have made deep neural networks a critical
component of computing". (All biographical details are general knowledge. The book
names him only as a co-author in the appendix B citation "Bahdanau, Cho, and Bengio".)

## Key contributions
- **Attention for translation** (Bahdanau, Cho & Bengio 2014, arXiv 1409.0473). The paper behind the "Bahdanau attention mechanism" that the book calls the inspiration for the transformer's self-attention (book pp. 54–55, 291). It let the decoder of an encoder–decoder RNN look back at every input word at each step, through a weighted average $\mathbf{c}_i = \sum_j \alpha_{ij}\mathbf{h}_j$ ($\alpha_{ij}$ = [[attention-weights]], $\mathbf{h}_j$ = encoder [[hidden-state|hidden states]], $\mathbf{c}_i$ = the [[context-vector]]). → [[bahdanau-attention]], [[attention-mechanism]], [[machine-translation]]
- **Neural probabilistic language model** (Bengio et al. 2003; general knowledge). Predict the next word from the previous few with a neural net, and learn a vector for every word along the way. This is the ancestor of [[word-embeddings]], [[embedding-layer|embedding layers]], and [[next-token-prediction]]. The model maximizes the log-likelihood $\sum_t \log P(w_t \mid w_{t-n+1},\dots,w_{t-1})$ — read: "add up, over every position $t$ in the text, the log of the probability the model gave to the word that actually came next, given the $n-1$ words before it". → [[maximum-likelihood-estimation]], [[log-probability]], [[cross-entropy-loss]]
- **Vanishing gradients in RNNs** (Bengio, Simard & Frasconi 1994; general knowledge). Showed that learning long-term dependencies with gradient descent is hard because gradients shrink or blow up as they are multiplied back through time. This is the problem attention and transformers sidestep. → [[vanishing-and-exploding-gradients]], [[long-range-dependencies]], [[recurrent-neural-network]]
- **Other landmarks** (general knowledge): co-author of Generative Adversarial Networks (2014), of the GRU / RNN encoder–decoder work with Kyunghyun Cho (2014), and of the textbook *Deep Learning* (Goodfellow, Bengio & Courville, 2016). → [[deep-learning]]

## Connections
- Concepts: [[bahdanau-attention]], [[attention-mechanism]], [[word-embeddings]], [[next-token-prediction]], [[vanishing-and-exploding-gradients]], [[recurrent-neural-network]], [[deep-learning]]
- People: [[dzmitry-bahdanau]] (PhD student, first author of the attention paper), [[geoffrey-hinton]] (Turing Award co-recipient), [[ashish-vaswani]] (transformer built on the attention idea)
- Appears in sources: [[llm-ch03-coding-attention-mechanisms]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 3 references, book p. 291 (Bahdanau, Cho & Bengio 2014, "Neural Machine Translation by Jointly Learning to Align and Translate").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.2, book pp. 54–55 (the Bahdanau attention mechanism and its influence on the transformer).
- Biography, the 2003 language model, the 1994 vanishing-gradient result, and other works are general knowledge, not from the book.
