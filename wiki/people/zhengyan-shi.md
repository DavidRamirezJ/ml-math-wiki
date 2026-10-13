---
title: Zhengyan Shi
type: person
tags: [llm, instruction-tuning, fine-tuning]
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Zhengyan Shi (active 2020s)

> **Why they matter to ML:** first author of "Instruction Tuning With Loss Over Instructions" (2024), the paper that showed you often should **not** mask the instruction tokens when you instruction-fine-tune an LLM.

## Life (brief)
Zhengyan Shi is a machine-learning researcher who worked on efficient fine-tuning and instruction tuning of language models at University College London. The 2024 paper was written with co-authors including Adam X. Yang, Bin Wu, Laurence Aitchison, Emine Yilmaz and Aldo Lipani. (These details are general knowledge. The book names the paper's author only as "Shi" in appendix B and as "Shi et al." in §7.3.)

## Key contributions
- **"Instruction Tuning With Loss Over Instructions" (2024, arXiv 2405.14394).**
  - *What it asks:* during [[instruction-fine-tuning]], should the [[cross-entropy-loss]] count only the response tokens, or the instruction and input tokens too?
  - *Common practice it questions:* masking the instruction, by replacing its target token IDs with `-100` so PyTorch's `cross_entropy` ignores them. The book shows this in Figure 7.13 (book p. 222). → [[instruction-masking]], [[ignore-index-loss-masking]]
  - *Finding, as the book reports it:* "not masking the instructions and inputs in instruction fine-tuning effectively improves performance on various NLP tasks and open-ended generation benchmarks, particularly when trained on datasets with lengthy instructions and brief outputs or when using a small number of training examples" (appendix B, book p. 297).
  - *In the language of the math:* with masking, the loss averages $-\log p(y_t\mid y_{<t})$ only over response positions. Without masking (the paper calls this "instruction modelling"), it averages over instruction positions too. More positions in the loss means more training signal per example and acts like a [[regularization|regularizer]], which helps most when data is small, a link to [[overfitting]]. (This interpretation is general knowledge, not in the book.)
- **How the book uses the paper.**
  - §7.3 (book p. 223): researchers are "divided on whether masking the instructions is universally beneficial"; Shi et al. "demonstrated that not masking the instructions benefits the LLM performance." The book therefore does **not** mask, and leaves masking as Exercise 7.2. → [[collate-function]], [[instruction-dataset]]
  - Appendix C, Exercise 7.2 solution (book pp. 308–309): a `custom_collate_fn` that adds `targets[:instruction_length-1] = -100`. The masked model scored "slightly worse (approximately 4 points using the Ollama Llama 3 method from chapter 7)," which the book calls "consistent with observations" in the paper. → [[llm-as-a-judge]], [[local-llm-inference-ollama]]

## Connections
- Concepts: [[instruction-fine-tuning]], [[instruction-masking]], [[ignore-index-loss-masking]], [[cross-entropy-loss]], [[collate-function]], [[overfitting]], [[regularization]], [[llm-as-a-judge]]
- People: [[sebastian-raschka]] (cites the work)
- Appears in sources: [[llm-ch07-fine-tuning-to-follow-instructions]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 222–223 (Figure 7.13 instruction masking; Shi et al. 2024; masking left as Exercise 7.2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 7 references, book p. 297 ("Instruction Tuning with Loss Over Instructions" (2024) by Shi, https://arxiv.org/abs/2405.14394).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, Exercise 7.2 solution, book pp. 308–309 (masked collate function; about 4 points worse).
- > ⚠️ Note: the book credits the paper to "Shi" alone in appendix B but to "Shi et al." in §7.3; the paper has several authors. Full first name, affiliation and co-authors are general knowledge, not from the book.
