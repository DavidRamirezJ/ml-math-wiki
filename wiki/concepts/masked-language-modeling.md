---
title: Masked Language Modeling
type: concept
tags: [llm, nlp, self-supervised-learning, pretraining, bert]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models]
created: 2026-10-13
updated: 2026-10-13
---

# Masked Language Modeling

> **One-liner:** Hide some words in a sentence and train the model to guess them back from the words around them.

## 🟢 Intuition (Dummy)
It's a fill-in-the-blank game. Take a normal sentence, cover a few words with a blank, and ask the
model: "what was under each blank?" To answer it has to use the words on **both sides** of the gap.

Nobody has to label anything. The original sentence *is* the answer key. That's why this is a kind
of [[self-supervised-learning]]: the data makes its own labels. It's the training game that
[[bert]] plays (the book calls it "masked word prediction").

GPT plays a different game, "guess the **next** word" ([[next-token-prediction]]), where it can only
look at words to the left.

## 🟡 Mechanics (Practitioner)

### The book's example (Figure 1.5, book p. 9)
- Original: "This is an example of how concise I can be"
- Input with random masks: "This is an __ of how concise I __ be"
- BERT fills in the missing words → "This is an **example** of how concise I **can** be"

The figure caption: BERT "receives inputs where words are randomly masked during training" and
"fills in the missing words to generate the original sentence" (book p. 9).

### The objective (standard notation; the book states it only in words)

$$\mathcal L_{\text{MLM}}(\theta) = -\sum_{i\in M}\log p_\theta\big(x_i \mid \tilde{\mathbf x}\big)$$

Symbol by symbol:
- $\mathbf x = (x_1,\dots,x_T)$ = the original token sequence.
- $M\subseteq\{1,\dots,T\}$ = the positions chosen at random to be masked (in the example, the
  positions of "example" and "can", so $|M|=2$).
- $\tilde{\mathbf x}$ = the corrupted sequence, where every position in $M$ is replaced by a blank.
- $p_\theta(x_i\mid\tilde{\mathbf x})$ = the model's [[softmax]] probability for the *true* word at
  position $i$, given the whole corrupted sentence.
- $\sum_{i\in M}$ = add up only over the masked positions. Unmasked positions don't contribute.
- The minus log makes it a [[cross-entropy-loss]]. Minimizing it is [[maximum-likelihood-estimation]]
  for the hidden words.

### Compare with next-token prediction

$$\mathcal L_{\text{next}}(\theta) = -\sum_{t=1}^{T}\log p_\theta\big(x_t \mid x_1,\dots,x_{t-1}\big)$$

| | Masked LM (BERT) | Next-token LM (GPT) |
|---|---|---|
| Context used | Left **and** right ($\tilde{\mathbf x}$) | Left only ($x_{<t}$) |
| Positions with a loss | Only the masked set $M$ | Every position |
| Natural use | Understanding: [[text-classification]] (sentiment, document categorization) | Generation: [[text-completion]], translation, code |

(Uses from book pp. 8–9.)

## 🔴 Theory (Expert)
- **Not a factorization of $p(\mathbf x)$.** Next-token training fits the chain-rule factors of the
  joint distribution, $p(\mathbf x)=\prod_t p(x_t\mid x_{<t})$, so you can sample text left to right.
  MLM fits conditionals of the form $p(x_i\mid \text{everything else, partly hidden})$. Those
  conditionals don't directly define a single joint distribution you can sample from step by step,
  which is why MLM models aren't natural generators.
- **Bidirectional context ⇒ no causal mask.** MLM works because the encoder attends over all
  positions. A GPT-style model applies a [[causal-attention-mask]] (scores for $j>i$ set to $-\infty$
  before the softmax). If you trained such a model with MLM, it couldn't use the right context it needs.
- **Signal density.** Only the $|M|$ masked positions give a gradient per sequence, while next-token
  prediction gives $T$ (or $T-1$). That's a known efficiency cost of MLM. (Background; not discussed in
  the book.)
- **Relation to denoising.** MLM is a denoising objective: corrupt the input, then reconstruct it.
  The model learns representations that are robust to missing pieces, which is part of why its
  features transfer well to classification ([[feature-extraction]], [[transfer-learning]]).

## Connections
- Builds on: [[self-supervised-learning]], [[cross-entropy-loss]], [[softmax]], [[maximum-likelihood-estimation]], [[token]]
- Used by: [[bert]], [[pretraining]]
- Leads to: [[text-classification]], [[classification-fine-tuning]]
- Contrast: [[next-token-prediction]], [[causal-attention-mask]], [[gpt-architecture]]
- People: [[jacob-devlin]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 8–9 (BERT's "masked word prediction"; Figure 1.5 example with 2 masked words; strengths in text classification).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book p. 289 (Devlin et al. 2018, arXiv 1810.04805).
