---
title: Angela Fan
type: person
tags: [decoding, text-generation, sampling, nlp]
sources: [llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Angela Fan (active 2010s–present)

> **Why they matter to ML:** first author of the 2018 paper that introduced **top-k sampling**, the trick of letting an LLM pick its next word only from its $k$ best guesses.

## Life (brief)
The book names her only as "Fan et al." in appendix B: "The paper that originally introduced top-k sampling is 'Hierarchical Neural Story Generation' (2018) by Fan et al., https://arxiv.org/abs/1805.04833" (book p. 295).

General knowledge, not from the book: Angela Fan is a research scientist at Meta AI (FAIR). She did her PhD in France (INRIA Nancy / Université de Lorraine) while working at Facebook AI Research. The 2018 paper was written with Mike Lewis and Yann Dauphin. Later she worked on large multilingual translation (No Language Left Behind) and on the Llama 2 models.

## Key contributions
- **Top-k sampling** (Fan, Lewis & Dauphin 2018, "Hierarchical Neural Story Generation", arXiv 1805.04833). → [[top-k-sampling]], [[decoding-strategies]]
  - **The problem it fixes.** [[greedy-decoding]] always picks the single most likely token, so the text is boring and often copied word for word from training data. Plain [[multinomial-sampling]] (with [[temperature-scaling]]) adds variety, but it can also pick silly tokens — the book's example is "every effort moves you **pizza**" (book p. 155).
  - **The idea.** Keep only the $k$ most likely tokens, throw away the rest, and sample among the survivors. In the book's words, it restricts sampling "to the k most likely tokens" and excludes all others "by masking their probability scores" (book p. 156).
  - **The formula** (the book's method, written as math). Let $\tau_k$ be the $k$-th largest logit:
    $$\tilde z_i = \begin{cases} z_i & \text{if } z_i \ge \tau_k \\ -\infty & \text{if } z_i < \tau_k \end{cases} \qquad p_i = \frac{e^{\tilde z_i / T}}{\sum_{j=1}^{V} e^{\tilde z_j / T}}$$
    Symbol by symbol: $z_i$ = the raw score ([[logits|logit]]) of token $i$; $\tau_k$ ("tau k") = the cut-off, the $k$-th biggest score; $\tilde z_i$ ("z-tilde") = the masked score, equal to $z_i$ if token $i$ made the top $k$, otherwise $-\infty$; $T$ = the temperature; $V$ = the [[vocabulary]] size; $\sum_{j=1}^{V}$ = "add up over all $V$ tokens"; $p_i$ = the final probability of token $i$. Because $e^{-\infty} = 0$, every masked token gets probability exactly 0 — the same trick as the [[causal-attention-mask]] (book p. 156).
  - **The book's worked numbers** (book pp. 156–157). Toy vocabulary of 9 words, logits `[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]`, $k=3$. The top 3 are 6.75 ("forward"), 6.28 ("toward"), 4.51 ("closer"). After masking and [[softmax]]: `[0.0615, 0, 0, 0.5775, 0, 0, 0, 0.3610, 0]`. Only three tokens can ever be chosen now.
    ```python
    top_logits, top_pos = torch.topk(next_token_logits, 3)   # top positions: [3, 7, 0]
    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float('-inf')),
        other=next_token_logits
    )
    topk_probas = torch.softmax(new_logits, dim=0)
    ```
  - **Order of operations in the book's `generate`** (Listing 5.4, book pp. 157–158): first the top-k mask, then divide by the temperature, then softmax + `torch.multinomial`; if `temperature` is 0 it falls back to greedy `argmax`.
  - In the book's `generate` function, `top_k=25, temperature=1.4` turns the memorized output into the fresh "Every effort moves you stand to work on surprise, a one of us had gone with random-" (book p. 158). → [[autoregressive-text-generation]]
- **Hierarchical story generation** (same paper, external context): first write a short prompt or premise, then generate a whole story conditioned on it, using a convolutional sequence-to-sequence model with gated multi-scale self-attention, plus a "fusion" of two seq2seq models so that the story stays tied to the prompt. The book only cites the paper for top-k. → [[encoder-decoder-architecture]], [[self-attention]]

## Connections
- Concepts: [[top-k-sampling]], [[decoding-strategies]], [[temperature-scaling]], [[multinomial-sampling]], [[greedy-decoding]], [[softmax]], [[logits]], [[top-p-sampling]] (the alternative the book mentions), [[beam-search]]
- People: [[sebastian-raschka]] (book author)
- Appears in sources: [[llm-ch05-pretraining-on-unlabeled-data]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 5 references, book p. 295 (Fan et al. 2018 introduced top-k sampling; top-p as the alternative).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.2, book pp. 155–157 (top-k sampling, −inf masking, Figure 5.15, worked numbers).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.3, book pp. 157–158 (Listing 5.4 `generate`, example output).
- Biography, co-authors, and the story-generation model description are general knowledge, not from the book.
