---
title: Tom B. Brown
type: person
tags: [researcher, openai, llm, gpt-3, few-shot]
sources: [llm-ch01-understanding-large-language-models, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Tom B. Brown (contemporary)

> **Why they matter to ML:** first author of "Language Models are Few-Shot Learners" (2020), the GPT-3 paper — the 175-billion-parameter decoder-only model that showed few-shot learning and became the base of the original ChatGPT.

## Life (brief)
Engineer and researcher; at OpenAI when GPT-3 was built, later a co-founder of Anthropic
(general knowledge, not in the book). The book cites him only as "Brown et al.".

## Key contributions
- **GPT-3** — "Language Models are Few-Shot Learners" (2020) by Brown et al., arXiv 2005.14165.
  Appendix B calls it "the paper describing the decoder-style GPT-3 model, which inspired modern
  LLMs and will be used as a template for implementing an LLM from scratch in this book"
  (Appendix B, book pp. 289–290).
  - Size: **96 transformer layers**, **175 billion parameters** (§1.6, book p. 13); embedding size
    **12,288** (§2.1, book p. 20).
  - Same architecture as GPT-2, "except that the largest version (175 billion) is 100x larger
    than the largest GPT-2 model and has been trained on much more data" (Appendix B, book p. 293).
  - Data: Table 1.1 — 300 billion training tokens sampled from 499 billion; estimated
    pretraining cost **$4.6 million** (§1.5, book p. 11).
  - Training on one GPU would take 355 years (V100) or 665 years (RTX 8000), per Lambda Labs
    (§4.1, book p. 94).
  → [[gpt-3]], [[decoder-only-transformer]], [[pretraining-datasets]], [[parameter-count-and-memory]]
- **Few-shot / zero-shot prompting** — the paper's title idea. The book's Fig. 1.6 shows zero-shot
  ("Translate English to German: breakfast =>" → "Frühstück") and few-shot
  ("gaot => goat, sheo => shoe, pohne =>" → "phone") use with no retraining (§1.4, book p. 10);
  the GPT-3 base model has "limited few-shot capabilities" (§1.3, book p. 7).
  → [[few-shot-learning]], [[zero-shot-learning]], [[emergent-behavior]]
- **Base of ChatGPT** — the original ChatGPT came from fine-tuning GPT-3 on an instruction
  dataset with the InstructGPT method (§1.6, book p. 12). → [[instruction-fine-tuning]], [[long-ouyang]]

> ⚠️ Contradiction (book erratum): the Ch. 4 reference list prints the GPT-3 paper as **(2023)**
> (Appendix B, book p. 293), but the Ch. 1 list gives **(2020)** (book p. 290). 2020 is correct
> (arXiv 2005.14165, May 2020), and §1.6 also says "GPT-3 was introduced in 2020" (book p. 13).

## Connections
- Concepts: [[gpt-3]], [[few-shot-learning]], [[zero-shot-learning]], [[emergent-behavior]], [[decoder-only-transformer]], [[foundation-model]], [[pretraining-datasets]]
- Related people: [[alec-radford]] (GPT, GPT-2), [[ashish-vaswani]] (transformer), [[long-ouyang]] (InstructGPT on top of GPT-3)
- Appears in sources: [[llm-ch01-understanding-large-language-models]], [[llm-ch04-implementing-a-gpt-model]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book p. 7; §1.4, book p. 10; §1.5, book p. 11; §1.6, book pp. 12–13.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book p. 94 (GPT-2 vs GPT-3 sidebar).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book pp. 289–290 (Ch. 1 refs) and p. 293 (Ch. 4 refs).
- Biographical detail is general knowledge; the book names only "Brown et al.".
