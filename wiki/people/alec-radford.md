---
title: Alec Radford
type: person
tags: [researcher, openai, llm, gpt]
sources: [llm-ch01-understanding-large-language-models, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Alec Radford (contemporary)

> **Why they matter to ML:** first author of the original GPT paper and the GPT-2 paper — the decoder-only, next-word-prediction recipe that the whole book rebuilds from scratch.

## Life (brief)
American machine-learning researcher, long at OpenAI (general knowledge; the book names him only
as "Radford et al. from OpenAI"). Beyond GPT, he is known for co-authoring DCGAN, CLIP and
Whisper (general knowledge, not in the book). Birth year is not public in the sources used here.

## Key contributions
- **GPT (GPT-1)** — "Improving Language Understanding by Generative Pre-Training" by Radford et
  al. from OpenAI (https://mng.bz/x2qg), the paper where "GPT was originally introduced" (book
  §1.6, p. 12). It established generative [[pretraining]] of a
  [[decoder-only-transformer]] (the "followed by task [[fine-tuning]]" recipe and the year 2018
  are external context; the book only names the paper).
  → [[gpt-architecture]], [[next-token-prediction]], [[self-supervised-learning]]
- **GPT-2** — "Language Models Are Unsupervised Multitask Learners" (2019) by Radford et al.
  (linked as https://mng.bz/yoBq in §4.1 and https://mng.bz/DMv0 in Appendix B).
  Introduced "a series of transformer-based LLMs with varying sizes—124 million, 355 million,
  774 million, and 1.5 billion parameters" (Appendix B, book p. 293). The book scales its model to
  the smallest GPT-2, **124M** parameters, "as described in" this report, and notes that "the
  original report mentions 117 million parameters, this was later corrected" (§4.1, book p. 93).
  OpenAI's public GPT-2 weights are why the book uses GPT-2 at all (book p. 94).
  → [[gpt-2]], [[gpt-model-configuration]], [[pretrained-weights]], [[zero-shot-learning]]

> ⚠️ Contradiction (inside the book): §4.1 lists the larger GPT-2 sizes as **345, 762 and 1,542
> million** parameters (book p. 93), while Appendix B lists **355M, 774M and 1.5B** (book p. 293).
> The later chapters load `gpt2-medium (355M)`, `gpt2-large (774M)`, `gpt2-xl (1558M)` (§5.5,
> book p. 163). The 345/762/1,542 numbers are the original report's figures; 355/774/1,558 are the
> corrected counts (same correction as 117M → 124M).

## Connections
- Concepts: [[gpt-2]], [[gpt-architecture]], [[decoder-only-transformer]], [[pretraining]], [[next-token-prediction]], [[gpt-model-configuration]], [[pretrained-weights]], [[emergent-behavior]], [[llm-building-roadmap]]
- Related people: [[tom-brown]] (GPT-3, scaled-up successor), [[ashish-vaswani]] (transformer the GPT decoder comes from)
- Appears in sources: [[llm-ch01-understanding-large-language-models]], [[llm-ch04-implementing-a-gpt-model]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book p. 12 (original GPT paper).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 93–94 (GPT-2 124M / 117M correction, public weights).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B (Ch. 4 references), book p. 293 (GPT-2 paper, 2019, sizes).
- Biographical detail is general knowledge; the book gives only the surname and affiliation.
