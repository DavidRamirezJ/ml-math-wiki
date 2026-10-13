---
title: Biao Zhang
type: person
tags: [deep-learning, normalization, machine-translation, nlp]
sources: [llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Biao Zhang (active 2010s–present)

> **Why they matter to ML:** he is the first author of **RMSNorm**, the stripped-down LayerNorm that most modern open LLMs use in place of GPT-2's LayerNorm.

## Life (brief)
Biao Zhang is a machine-translation and NLP researcher. He wrote the RMSNorm paper during his PhD at the University of Edinburgh, where [[rico-sennrich]] was one of his advisors, and later moved to industry research at Google. (These biographical details are general knowledge and stated cautiously. The book names him only as "Zhang" in one appendix B citation.)

## Key contributions
- **RMSNorm** (Zhang & Sennrich, "Root Mean Square Layer Normalization", 2019, arXiv 1910.07467). The book calls it "a popular variant of LayerNorm used in modern LLMs ... due to its improved computing efficiency," which normalizes "using only the root mean square of the inputs, without subtracting the mean before squaring," so "it does not center the data before computing the scale" (book pp. 292–293). → [[rmsnorm]]
  - **The formula** (from the paper; the book gives it only in words):
    $$\mathrm{RMSNorm}(\mathbf{x}) = \boldsymbol\gamma \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}$$
    Symbol by symbol: $\mathbf{x}$ = one token's activation [[vector]]; $d$ = its length (the [[embedding-dimension]], 768 in GPT-2 124M); $x_i^2$ = the $i$-th entry squared; $\frac1d\sum_{i=1}^{d}$ = average over all entries; $\sqrt{\cdot}$ = square root, so the denominator is the **root mean square**, the "typical size" of an entry; $\epsilon$ = tiny constant against division by zero; $\boldsymbol\gamma$ = learned per-entry scale; $\odot$ = entry-by-entry multiplication.
  - **The idea in one line:** [[layer-normalization]] both *recentres* (subtracts the mean) and *rescales*. Zhang & Sennrich argued the rescaling is what matters, so RMSNorm keeps only that. It drops the mean $\mu$ and the shift $\boldsymbol\beta$, which halves the norm's parameters ($d$ instead of $2d$) and saves compute.
  - **Worked example** (our calculation on the book's §4.2 row `[0.2260, 0.3470, 0, 0.2216, 0, 0]`): RMS ≈ 0.1917, so RMSNorm gives ≈ `[1.179, 1.810, 0, 1.156, 0, 0]`. The book's LayerNorm gives `[0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719]` for the same row (book p. 102). The zeros stay zero under RMSNorm because nothing is subtracted. See [[rmsnorm]] for the full walk-through.
  - **Key property:** it is re-scaling invariant ($\mathrm{RMSNorm}(\alpha\mathbf{x}) = \mathrm{RMSNorm}(\mathbf{x})$ for $\alpha>0$, ignoring $\epsilon$) but not re-centering invariant. The paper reports quality similar to LayerNorm with noticeably less running time.
- **Machine-translation research** (general knowledge): work on efficient and deep [[encoder-decoder-architecture|encoder–decoder]] transformers for [[machine-translation]], often with Rico Sennrich.

## Connections
- Concepts: [[rmsnorm]], [[layer-normalization]], [[norm]], [[pre-layernorm-vs-post-layernorm]], [[transformer-block]], [[machine-translation]], [[large-language-model]]
- People: [[rico-sennrich]] (co-author and advisor), [[jimmy-lei-ba]] and [[geoffrey-hinton]] (LayerNorm, which RMSNorm simplifies)
- Appears in sources: [[llm-ch04-implementing-a-gpt-model]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 4 references, book pp. 292–293 (RMSNorm description; "Root Mean Square Layer Normalization" (2019) by Zhang and Sennrich, arXiv 1910.07467).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 100–102 (the example row and its LayerNorm output, used for the comparison).
- Biography, the full name, the formula and the paper's claims are general knowledge from the original paper, not printed in the book.
