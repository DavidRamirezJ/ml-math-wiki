---
title: Long Ouyang
type: person
tags: [llm, alignment, rlhf, openai]
sources: [llm-ch01-understanding-large-language-models, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Long Ouyang (active 2010s–present)

> **Why they matter to ML:** first author of the **InstructGPT** paper, the recipe that turned GPT-3 into a helpful instruction follower and became the method behind the original ChatGPT.

## Life (brief)
Long Ouyang is a researcher at OpenAI. He led the 2022 paper "Training Language Models to Follow Instructions with Human Feedback," written with a large OpenAI team (including Jeff Wu, Paul Christiano, Jan Leike and Ryan Lowe). (Biographical details are general knowledge. The book names him only as "Ouyang et al." in appendix B; in section 1.6 it just says "OpenAI's InstructGPT paper.")

## Key contributions
- **InstructGPT** (Ouyang et al. 2022, arXiv 2203.02155). The book says the original ChatGPT model "was created by fine-tuning GPT-3 on a large instruction dataset using a method from OpenAI's InstructGPT paper." → [[gpt-3]], [[instruction-fine-tuning]], [[instruction-dataset]]
- **RLHF — reinforcement learning from human feedback.** The paper's recipe has three steps (external context, not detailed in the book):
  1. **Supervised fine-tuning (SFT):** train GPT-3 on example prompts with human-written answers. This is the same idea as the book's chapter 7, where a GPT-2 model learns from 1,100 instruction–response pairs. → [[instruction-fine-tuning]], [[prompt-template]]
  2. **Reward model:** people rank several model answers; a model learns to score answers the way people would.
  3. **Reinforcement learning:** tune the LLM so it gets high reward scores, but do not let it drift too far from the SFT model.
  - The ranking step uses a pairwise loss: $\mathcal{L} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$. Symbol by symbol: $x$ = the prompt; $y_w$ = the answer people preferred (the "winner"); $y_l$ = the answer they liked less (the "loser"); $r_\phi$ = the reward model with parameters $\phi$, which gives each answer a score; $\sigma$ = the [[sigmoid-function]], which turns a score gap into a probability between 0 and 1; $-\log$ = turn "high probability" into "low loss." So the loss is small when the reward model gives the preferred answer a clearly higher score. → [[binary-cross-entropy]], [[logistic-regression]]
  - A headline result (external context): people preferred answers from a 1.3B-parameter InstructGPT over the 175B-parameter GPT-3.
- **Link to the book's chapter 7.** Appendix B says InstructGPT "will be discussed in more detail in chapter 7." Chapter 7 does the SFT step only. It then names **preference fine-tuning** as an optional next step and points to **DPO**, a simpler alternative to the reward-model-plus-RL part of RLHF. → [[preference-fine-tuning]], [[direct-preference-optimization]]

## Connections
- Concepts: [[instruction-fine-tuning]], [[preference-fine-tuning]], [[direct-preference-optimization]], [[gpt-3]], [[fine-tuning]], [[pretraining]], [[llm-evaluation]]
- People: [[tom-brown]] (GPT-3), [[alec-radford]] (GPT)
- Appears in sources: [[llm-ch01-understanding-large-language-models]], [[llm-ch07-fine-tuning-to-follow-instructions]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book p. 12 (ChatGPT made by fine-tuning GPT-3 with the InstructGPT method).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.9, book p. 247 (preference fine-tuning as an optional step after instruction fine-tuning; DPO).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 1 references, book p. 290 (Ouyang et al. 2022 citation).
- > ⚠️ Note: appendix B says InstructGPT is "discussed in more detail in chapter 7," but the extracted chapter 7 text does not name InstructGPT or RLHF. Chapter 7 covers only the instruction-fine-tuning (SFT) part.
- The three RLHF steps, the reward loss and the 1.3B-vs-175B result come from the InstructGPT paper (general knowledge), not from the book.
