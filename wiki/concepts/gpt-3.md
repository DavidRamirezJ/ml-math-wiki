---
title: GPT-3 and ChatGPT
type: concept
tags: [llm, transformer, gpt, foundation-model, instruction-tuning]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# GPT-3 and ChatGPT

> **One-liner:** GPT-3 is OpenAI's giant 175-billion-parameter "predict the next word" model from 2020, and the original ChatGPT is that same model taught to follow instructions.

## 🟢 Intuition (Dummy)
Picture a student who has read a huge pile of the internet, plus books and Wikipedia, and who
practised only one game: "guess the next word." That student is **GPT-3**. It is great at
finishing your sentences. Show it a couple of examples of a new task in the prompt and it can
often pick up the pattern too. But it isn't really a helpful assistant yet. Ask it a question and
it might just ramble on in the same style.

**ChatGPT** (the first version) is what you get when you take that well-read student and give
them a second, shorter course: thousands of "here's an instruction, here's a good answer"
examples. Now the student *answers* you instead of just continuing your text.

So there are two steps: first **[[pretraining]]** (GPT-3, the "base" or "foundation" model), then
**[[fine-tuning]]** on instructions (ChatGPT). That two-step recipe is the backbone of the whole
book.

## 🟡 Mechanics (Practitioner)

### What GPT-3 is
- **Architecture:** a decoder-only transformer, the same design as [[gpt-2]], just much bigger
  (book p. 94). See [[gpt-architecture]].
- **Size:** **96 transformer layers** and **175 billion parameters** (book p. 13). GPT-2's largest
  model has 1.5 billion parameters, so GPT-3 is about 100× bigger (book p. 94; appendix B p. 293).
- **Year:** introduced in **2020** (book p. 13), in Brown et al., "Language Models are Few-Shot
  Learners" (arXiv 2005.14165; appendix B pp. 289–290).
- **Weights:** not public when the book was written. That's why the book builds and loads
  GPT-2 instead, which runs on a single laptop (book p. 94).
- **Training cost:** about **$4.6 million** in cloud credits (book p. 11). According to
  Lambda Labs, training it on one V100 GPU would take **355 years**, and on one consumer
  RTX 8000 GPU **665 years** (book p. 94).

### The pretraining data (Table 1.1, book p. 11)

| Dataset | Description | Tokens | Share of training data |
|---|---|---|---|
| CommonCrawl (filtered) | Web crawl | 410 B | 60% |
| WebText2 | Web crawl | 19 B | 22% |
| Books1 | Internet book corpus | 12 B | 8% |
| Books2 | Internet book corpus | 55 B | 8% |
| Wikipedia | High-quality text | 3 B | 3% |

The subsets add up to $410+19+12+55+3 = 499$ billion tokens, but GPT-3 trained on only **300 billion**
tokens sampled with the proportions above (book p. 11). Notice that small, high-quality sources get
a big share: Wikipedia is $3/499 \approx 0.6\%$ of the tokens but $3\%$ of the training mix. (That
reweighting is our own reading of the table; the book doesn't say it in words.)

### What GPT-3 is trained to do
GPT-3 is trained only on **next-word prediction** ([[next-token-prediction]]). In probability
language (standard notation, not printed in the book):

$$P_\theta(x_1,\dots,x_T) = \prod_{t=1}^{T} P_\theta(x_t \mid x_1,\dots,x_{t-1})$$

Reading it symbol by symbol:
- $x_1,\dots,x_T$ = the tokens of a text, in order; $T$ = how many there are.
- $P_\theta(\cdot)$ = the probability the model assigns; $\theta$ = all 175 billion parameters.
- $\prod_{t=1}^{T}$ = "multiply together, for every position $t$ from 1 to $T$."
- $P_\theta(x_t \mid x_1,\dots,x_{t-1})$ = "the probability of token $x_t$, **given** (the bar $\mid$)
  all the tokens before it."

Training pushes $\theta$ so that the real next token gets high probability. That's
[[maximum-likelihood-estimation]], carried out by minimizing [[cross-entropy-loss]].

### What a base model can do (book pp. 7, 9–10)
- **[[text-completion]]:** "Breakfast is the" → "most important meal of the day."
- **[[zero-shot-learning]]:** "Translate English to German: breakfast =>" → "Frühstück", with no examples.
- **[[few-shot-learning]]:** "gaot => goat, sheo => shoe, pohne =>" → "phone", from just two examples in the prompt.

None of these needs any retraining. The weights $\theta$ don't change; the task lives in the prompt.
Note the book's wording for the base model itself: GPT-3 is capable of text completion and has only
"**limited** few-shot capabilities" (book p. 7).

### From GPT-3 to ChatGPT
- The original ChatGPT was built by **fine-tuning GPT-3 on a large instruction dataset**, using the
  method from OpenAI's **InstructGPT** paper (Ouyang et al., 2022, arXiv 2203.02155) (book p. 12;
  appendix B p. 290). That paper also uses human feedback (RLHF), which the book only points to
  ([[preference-fine-tuning]]).
- So ChatGPT = **base model + [[instruction-fine-tuning]]**. Chapter 7 of the book does a small version
  of exactly this on GPT-2 with an Alpaca-style dataset.
- Both GPT-3 and ChatGPT are **decoder-only**: "LLMs for generating text and following instructions, such as
  GPT-3 and ChatGPT, only implement decoder modules" (chapter 1 summary, book p. 15).

## 🔴 Theory (Expert)
- **Same architecture, more scale.** The book stresses that GPT-3 has "fundamentally the same
  architecture" as GPT-2 (book p. 94). The jump from 1.5 B to 175 B parameters comes from making the
  embedding wider and the stack deeper (96 layers), not from new building blocks. So the
  `GPTModel` class you code in chapter 4 is, structurally, a small GPT-3.
- **Emergence.** GPT-style models can translate, classify, and summarize even though they were
  only trained on next-word prediction. The book calls this [[emergent-behavior]]: the abilities
  arise from exposure to huge amounts of multilingual text in many contexts (book p. 14). The
  title of the GPT-3 paper, "Language Models are Few-Shot Learners", is about exactly this kind of
  capability.
- **In-context learning ≠ learning.** In few-shot prompting no gradient step happens. The model
  computes $P_\theta(\text{answer} \mid \text{examples}, \text{query})$ with frozen $\theta$. The
  "learning" is conditioning. (The book doesn't use the term "in-context learning"; this is
  standard terminology.)
- **Base vs. aligned.** The base model maximizes likelihood on web text. That objective doesn't
  reward helpfulness. Instruction tuning changes the data distribution the model is fitted to
  (instruction → answer pairs), and preference tuning (RLHF, [[direct-preference-optimization]])
  changes the objective itself. Appendix B also cites Gekhman et al. (2024): models mostly pick up
  factual knowledge during pretraining, and fine-tuning mostly teaches them how to use it (book pp. 297–298).
- **Book erratum.** Appendix B gives the GPT-3 paper the year 2023 in the chapter 4 list (book p. 293)
  and 2020 in the chapter 1 list (book p. 290). **2020 is correct** (arXiv 2005.14165).
- **Custom models can beat it.** Custom-built, domain-specific models such as BloombergGPT (a GPT
  pretrained from scratch on finance data, Wu et al. 2023) outperformed ChatGPT on financial tasks while
  staying good on general benchmarks (book p. 5; appendix B p. 289). See [[custom-llm]].

## Connections
- Builds on: [[gpt-architecture]], [[transformer-architecture]], [[next-token-prediction]], [[pretraining]], [[probability-distribution]]
- Leads to: [[instruction-fine-tuning]], [[preference-fine-tuning]], [[few-shot-learning]], [[zero-shot-learning]], [[emergent-behavior]]
- Related: [[gpt-2]], [[foundation-model]], [[large-language-model]], [[pretraining-datasets]], [[bert]]
- People: [[tom-brown]] (GPT-3 paper), [[long-ouyang]] (InstructGPT), [[alec-radford]] (GPT lineage)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 5–7 (GPT-3 as base model; ChatGPT as a general-purpose LLM).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 9–10 (zero-/few-shot, Figure 1.6).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book pp. 10–11 (Table 1.1, 300 B tokens, $4.6 M).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book pp. 12–14 (96 layers, 175 B parameters, 2020; ChatGPT via InstructGPT; emergent behavior).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7 summary, book p. 15.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1 sidebar "GPT-2 vs. GPT-3", book p. 94.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 289–290, 293, 297–298 (Brown et al. 2020; Ouyang et al. 2022; Lambda Labs; Gekhman et al.).
