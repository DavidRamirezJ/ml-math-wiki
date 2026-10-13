---
title: BERT
type: concept
tags: [llm, transformer, encoder, nlp, text-classification]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# BERT

> **One-liner:** BERT is a transformer made only of the *encoder* half. It learns by filling in hidden words, and that makes it very good at understanding and classifying text.

## 🟢 Intuition (Dummy)
Remember fill-in-the-blank exercises at school? "This is an ___ of how concise I ___ be." To fill
those gaps you have to look at the words **before and after** each blank. That's exactly how BERT is
trained: random words are hidden, and it learns to guess them back (book p. 9, Figure 1.5).

Because BERT always looks in **both directions**, it gets a very good sense of what a whole sentence
means. That makes it good at jobs like "is this review positive or negative?" or "is this tweet
toxic?" X (formerly Twitter) uses BERT to detect toxic content (book p. 9).

What BERT is *not* built for is writing long new text. That's the job of its sibling, GPT, which
reads only left to right and predicts the next word.

## 🟡 Mechanics (Practitioner)

### Name and parts
- **BERT** = **B**idirectional **E**ncoder **R**epresentations from **T**ransformers (book p. 8).
- Built on the **encoder** submodule of the original [[transformer-architecture]]
  ([[encoder-decoder-architecture]]). GPT is built on the **decoder** (book pp. 8–9).
- Paper: Devlin et al. (2018), "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding", arXiv 1810.04805 (appendix B, p. 289).

### Training objective: masked word prediction
BERT's pretraining task is [[masked-language-modeling]]: "predict masked or hidden words in a given
sentence" (book pp. 8–9). In symbols (standard notation; the book describes it only in words):

$$\mathcal L_{\text{MLM}}(\theta) = -\sum_{i\in M}\log p_\theta\big(x_i \mid \tilde{\mathbf x}\big)$$

Symbol by symbol:
- $\mathbf x = (x_1,\dots,x_T)$ = the original sentence.
- $M$ = the set of positions that were masked; $\sum_{i\in M}$ = "add up over the masked positions only."
- $\tilde{\mathbf x}$ = the sentence with those positions hidden (e.g. "This is an __ of how concise I __ be").
- $p_\theta(x_i\mid\tilde{\mathbf x})$ = the model's probability for the true word at position $i$,
  given the *whole* corrupted sentence, both left and right context.
- $-\log$ and the sum make this a [[cross-entropy-loss]]: the same likelihood idea as GPT, but on
  different targets ([[maximum-likelihood-estimation]]).

Book example (Figure 1.5, book p. 9): input "This is an __ of how concise I __ be" → BERT fills in
"example" and "can" → "This is an example of how concise I can be". There are 2 masked words.

### BERT vs. GPT at a glance

| | BERT | GPT |
|---|---|---|
| Transformer half | Encoder | Decoder |
| Reads | Both directions | Left to right only |
| Training task | Masked word prediction | Next-word prediction |
| Strong at | [[text-classification]]: sentiment, document categorization | Generation: translation, summarization, fiction, code |

(book pp. 8–9)

### Using it for classification
For the spam-classification task in chapter 6, the book fine-tunes a **decoder** (GPT-2). It notes in
appendix B that **encoder-based models** such as BERT (and RoBERTa, Liu et al. 2019) are an effective
alternative for classification (appendix B, p. 296). It also points to "Additional Experiments Classifying
the Sentiment of 50k IMDb Movie Reviews" for such comparisons (appendix B, p. 296).

The standard BERT recipe (background, not in the book) puts a small linear head
$\mathbf z = W\mathbf h_{\texttt{[CLS]}}+\mathbf b$ on the encoder's output for a special first token and
trains it with cross-entropy, much like the book's GPT [[classification-head]].

## 🔴 Theory (Expert)
- **Why bidirectionality helps classification.** In a GPT-style model the causal mask lets token $i$
  see only positions $\le i$. Only the last token sees the whole input, and that's why chapter 6
  classifies from the last token (book p. 190; see [[last-token-pooling]]). In an encoder there's no
  mask, so *every* position's representation depends on the full sentence. Appendix B cites two
  results that point the same way: "classification performance can be further improved by removing
  the causal mask during classification fine-tuning alongside other modifications" (Li et al. 2023,
  "Label Supervised LLaMA Finetuning"; BehnamGhader et al. 2024, "LLM2Vec: Large Language Models Are
  Secretly Powerful Text Encoders") (appendix B, p. 296).
- **Why BERT can't simply generate.** MLM models $p(x_i\mid\tilde{\mathbf x})$ for a few hidden
  positions given everything else. It does *not* give the chain-rule factorization
  $\prod_t p(x_t\mid x_{<t})$ that left-to-right generation needs. So sampling a fresh sequence from
  BERT isn't natural, while GPT's objective is exactly that factorization
  ([[autoregressive-text-generation]]).
- **Sample efficiency trade-off.** MLM only gets a training signal from the masked subset $M$ of
  positions, while next-token prediction gets one at every position. (Background; the book doesn't
  discuss it.)
- **Tokenizer.** BERT uses the **WordPiece** subword scheme, while GPT-2 uses [[byte-pair-encoding]].
  (Background: the book does not say which tokenizer BERT uses; appendix B p. 291 only lists the
  WordPiece paper, Song et al. 2020, "Fast WordPiece Tokenization", as an alternative scheme "used by
  some other popular LLMs".)

## Connections
- Builds on: [[transformer-architecture]], [[encoder-decoder-architecture]], [[self-attention]], [[word-embeddings]], [[cross-entropy-loss]]
- Leads to: [[text-classification]], [[classification-fine-tuning]], [[feature-extraction]]
- Trained with: [[masked-language-modeling]], [[self-supervised-learning]], [[pretraining]]
- Contrast: [[gpt-architecture]], [[gpt-3]], [[decoder-only-transformer]], [[causal-attention-mask]]
- People: [[jacob-devlin]] (BERT paper), [[ashish-vaswani]] (transformer)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 8–9 (BERT = bidirectional encoder representations from transformers; encoder submodule; masked word prediction; text classification; X uses it for toxic-content detection; Figure 1.5).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book p. 190 (causal mask; only the last token sees all tokens — contrast with bidirectional encoders).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 289, 291, 296 (Devlin et al. 2018; WordPiece; encoder models as a classification alternative; RoBERTa; Label Supervised LLaMA; LLM2Vec).
