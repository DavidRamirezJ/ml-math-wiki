---
title: Jacob Devlin
type: person
tags: [nlp, transformers, bert, google]
sources: [llm-ch01-understanding-large-language-models, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Jacob Devlin (active 2010s–present)

> **Why they matter to ML:** first author of **BERT**, the famous *encoder-only* transformer that learns by filling in hidden words, and the go-to model family for text classification.

## Life (brief)
Jacob Devlin is a natural-language-processing researcher. He was at Google AI Language when he and his co-authors (Ming-Wei Chang, Kenton Lee, Kristina Toutanova) published BERT in 2018; before that he worked at Microsoft Research on machine translation. The BERT paper won the best long paper award at NAACL 2019. (These biographical details are general knowledge. The Raschka book names him only in the citation "Devlin et al." in appendix B.)

## Key contributions
- **BERT — Bidirectional Encoder Representations from Transformers** (Devlin et al. 2018, arXiv 1810.04805). The book calls it "the original encoder-style transformer." It keeps only the **encoder** half of the original transformer. → [[bert]], [[encoder-decoder-architecture]], [[transformer-architecture]]
  - **How it learns: masked word prediction.** During training some words are hidden and the model must guess them. The book's figure 1.5 example: the input "This is an __ of how concise I __ be" has **2** masked words, and BERT must rebuild "This is an example of how concise I can be." → [[masked-language-modeling]], [[self-supervised-learning]]
  - In symbols, the training goal is roughly: maximize $\sum_{i \in M} \log p_\theta(x_i \mid \mathbf{x}_{\setminus M})$. Reading it piece by piece: $M$ = the set of masked positions; $x_i$ = the true word at hidden position $i$; $\mathbf{x}_{\setminus M}$ = the sentence with those words blanked out (the "$\setminus$" means "without"); $p_\theta$ = the model's probability with parameters $\theta$; $\log$ = the logarithm, so we add up [[log-probability|log-probabilities]] instead of multiplying tiny numbers. Because the model sees words on *both* sides of each blank, it is "bidirectional," unlike GPT, which only looks left. → [[maximum-likelihood-estimation]], [[next-token-prediction]]
  - **What it is good at:** text classification, such as sentiment prediction and document categorization. The book notes that, at the time of writing, X (formerly Twitter) used BERT to detect toxic content. → [[text-classification]]
  - BERT uses the **WordPiece** subword tokenizer (external context). The book does not link WordPiece to BERT: its appendix B (Ch. 2 references, p. 291) only lists the "Fast WordPiece Tokenization" paper (Song et al. 2020) as an alternative scheme "used by some other popular LLMs," in contrast to the BPE tokenizer of GPT-2. → [[subword-tokenization]], [[tokenization]], [[byte-pair-encoding]]
- **Encoder models as a classification alternative.** In chapter 6 the book fine-tunes the *decoder* GPT-2 for spam classification, and then points to BERT (and its successor RoBERTa) as encoder-based models that "can be effective for classification tasks." → [[classification-fine-tuning]], [[gpt-2]]

## Connections
- Concepts: [[bert]], [[masked-language-modeling]], [[encoder-decoder-architecture]], [[transformer-architecture]], [[self-supervised-learning]], [[text-classification]], [[classification-fine-tuning]], [[subword-tokenization]], [[pretraining]], [[fine-tuning]]
- Contrast with: [[gpt-architecture]], [[decoder-only-transformer]]
- People: [[ashish-vaswani]] (original transformer), [[alec-radford]] (GPT)
- Appears in sources: [[llm-ch01-understanding-large-language-models]], [[llm-ch06-fine-tuning-for-classification]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 8–9 (BERT defined, masked-word figure 1.5, classification use, X/Twitter example).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 1 references, book p. 289 ("the original encoder-style transformer, called BERT"); Ch. 2 references, book p. 291 (WordPiece listed as an alternative tokenizer, Song et al., not tied to BERT in the book); Ch. 6 references, book p. 296 (BERT and RoBERTa as classification alternatives).
- Biography, co-author names and BERT's use of WordPiece are general knowledge, not from the book.
