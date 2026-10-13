---
title: Generative AI
type: concept
tags: [ai, deep-learning, llm, generative-models]
difficulty: beginner
aliases: [GenAI, generative artificial intelligence]
sources: [llm-ch01-understanding-large-language-models]
created: 2026-10-13
updated: 2026-10-13
---

# Generative AI

> **One-liner:** Generative AI is AI that *makes new stuff* — text, images, or other media — instead of just sorting or labeling what already exists.

## 🟢 Intuition (Dummy)
Most classic AI is a **judge**: "Is this email spam? Yes or no." Generative AI
is an **author**: "Write me an email about the meeting." It doesn't pick from a
fixed list of answers — it creates something new.

An LLM like ChatGPT is generative AI for text: you give it the start of
something (or an instruction), and it keeps writing. Image generators do the
same for pictures.

In the book's family tree (Fig. 1.1), GenAI lives **inside deep learning** and
**overlaps** with LLMs: LLMs are generative because they write text, but GenAI
also covers images and other media.

## 🟡 Mechanics (Practitioner)
**Book definition (§1.1, p. 3):** because LLMs can generate text, they are a form
of **generative artificial intelligence**, also called **generative AI** or
**GenAI**. GenAI "involves the use of deep neural networks to create new
content, such as text, images, or various forms of media" (Fig. 1.1 label).

**How an LLM generates.** It builds text one [[token]] at a time. At each step it
computes a probability for every possible next token and picks one:

$$x_{t} \sim P_\theta(x_t \mid x_1,\dots,x_{t-1})$$

- $x_t$ = the new token at step $t$.
- $\sim$ = "is drawn from" (or, with [[greedy-decoding]], "is the most likely token of").
- $P_\theta(\cdot\mid\cdot)$ = the model's [[probability-distribution]] over the vocabulary, given everything written so far; $\theta$ = its parameters.

Then $x_t$ is appended and the loop repeats ([[autoregressive-text-generation]]).
Example from the book (Fig. 1.6): "Breakfast is the" → "most important meal of
the day."

**Judge vs. author, side by side:**

| | Discriminative (classic ML) | Generative (GenAI) |
|---|---|---|
| Question | "Which label fits this input?" | "What new content fits this prompt?" |
| Output | a class, e.g. spam / not spam | text, image, audio… |
| Book example | spam filter (§1.1) | ChatGPT writing a poem (Fig. 1.2) |

## 🔴 Theory (Expert)
- **Probabilistic framing (standard, not stated in §1.1).** A generative model
  represents a distribution over the data itself, $p_\theta(\mathbf x)$, from which
  new samples can be drawn; a discriminative model represents only
  $p_\theta(y\mid\mathbf x)$. Autoregressive LLMs make $p_\theta(\mathbf x)$
  tractable by factorizing it with the chain rule,
  $p_\theta(x_{1:T})=\prod_t p_\theta(x_t\mid x_{<t})$, and are trained by
  [[maximum-likelihood-estimation]] ([[next-token-prediction]]).
- **Generation needs a decoding rule.** Turning the learned distribution into text
  requires a choice: greedy argmax, [[temperature-scaling]], [[top-k-sampling]],
  etc. ([[decoding-strategies]]). Sampling makes outputs varied; greedy decoding
  makes them deterministic.
- **One generative model, many tasks.** The book stresses that a single large
  generative language model can handle many tasks (translation, summarization,
  classification) instead of one model per task (§1.6, p. 14) — an
  [[emergent-behavior]] of large-scale generative pretraining.
- **Not the same as LLM.** The Fig. 1.1 overlap matters: GenAI includes
  non-text media, while some LLMs (e.g. [[bert]]-style encoders) are used mainly
  for classification rather than generation.

## Connections
- Builds on: [[deep-learning]], [[neural-network]], [[probability-distribution]], [[artificial-intelligence]]
- Leads to: [[large-language-model]], [[autoregressive-text-generation]], [[decoding-strategies]]
- Related: [[machine-learning]], [[next-token-prediction]], [[text-completion]], [[gpt-architecture]], [[mixture-model]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1, book p. 3 (PDF p. 25) — definition of generative AI / GenAI, Fig. 1.1.
- Same file §1.2, book p. 5 — Fig. 1.2 (ChatGPT generating a poem).
- Same file §1.4, book p. 10 — Fig. 1.6 (text completion example).
- Same file §1.6, book p. 14 — one generative model for many tasks.
