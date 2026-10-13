---
title: Emergent Behavior
type: concept
tags: [llm, generalization, scaling, nlp]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models]
created: 2026-10-13
updated: 2026-10-13
---

# Emergent Behavior

> **One-liner:** Emergent behavior is when a model can do a job nobody trained it for — like a next-word guesser that suddenly can translate.

## 🟢 Intuition (Dummy)
Imagine a kid who only ever plays one game: "guess the next word in the sentence". They play it
with millions of books, in English, German, French, Spanish… One day you write
"Translate English to German: breakfast =>" and the kid answers "Frühstück". Nobody ever gave
them a translation lesson! But to guess next words well in books that mix languages, they had
to notice how languages line up. Translation "fell out" of the guessing game.

That surprise is what the book calls an **emergent behavior**: a skill the model was never
explicitly taught, which shows up anyway because it saw a huge amount of varied text. The big
win: one model can do many different jobs, instead of building one model per job.

## 🟡 Mechanics (Practitioner)

### The book's definition
"The ability to perform tasks that the model wasn't explicitly trained to perform is called an
*emergent behavior*. This capability isn't explicitly taught during training but emerges as a
natural consequence of the model's exposure to vast quantities of multilingual data in diverse
contexts" (§1.6, book p. 14).

The flagship example is **translation**. The original [[transformer-architecture|transformer]]
was "explicitly designed for language translation". GPT models have a "larger yet simpler
[[decoder-only-transformer|decoder-only architecture]] aimed at next-word prediction" and can
still translate. "This capability was initially unexpected to researchers" (book pp. 13–14).
The Chapter 1 summary widens the list: GPT-like models show "emergent properties" such as the
ability to **classify, translate, or summarize** text (book p. 15).

### Why it can happen: one objective, many tasks
The only training signal is [[next-token-prediction]] (standard notation; the book states it in
words, §1.6, book p. 12):

$$\theta^\star=\arg\max_{\theta}\sum_{t}\log P_\theta(x_t\mid x_1,\dots,x_{t-1})$$

Symbol by symbol:
- $\theta$ = all the [[model-parameters|model parameters]] (175 billion for GPT-3, book p. 13).
- $\theta^\star$ = the best parameters found by training.
- $\arg\max_\theta$ = "the value of $\theta$ that makes the following as large as possible".
- $\sum_t$ = add up over every position $t$ in the training text.
- $\log P_\theta(x_t\mid x_1,\dots,x_{t-1})$ = the [[log-probability]] the model gives to the true
  next token $x_t$, given the tokens before it.

There is no "translation term" in this formula. But a task can be written **as text**, so it
becomes a next-token question at use time:

$$\text{answer} \approx \arg\max_{y}\;P_{\theta^\star}\big(y\mid \text{"Translate English to German: breakfast =>"}\big)$$

- $y$ = a candidate continuation (e.g. "Frühstück").
- $P_{\theta^\star}(y\mid\text{prompt})$ = how likely the trained model thinks $y$ is after the prompt.

### Worked examples from the book (Fig. 1.6, book p. 10)
| Mode | Input (prompt) | Output | Skill not explicitly trained |
|---|---|---|---|
| Text completion | "Breakfast is the" | "most important meal of the day." | — (this *is* the training task) |
| [[zero-shot-learning|Zero-shot]] | "Translate English to German: breakfast =>" | "Frühstück" | translation |
| [[few-shot-learning|Few-shot]] | "gaot => goat, sheo => shoe, pohne =>" | "phone" | spelling correction |

None of these needs retraining, [[fine-tuning]] or a change to the architecture (book p. 10).
§1.3 already says the pretrained GPT-3 base model has "limited few-shot capabilities" (book p. 7).

### Ingredients the book names
- **Scale of data:** "vast quantities of multilingual data in diverse contexts" (book p. 14);
  GPT-3 was trained on 300 billion tokens sampled from 499 billion (§1.5, book p. 11).
- **Scale of model:** GPT-3 has 96 layers and 175 billion parameters (book p. 13).
- **Generality of the objective:** the payoff is that "we can perform diverse tasks without
  using diverse models for each" (book p. 14).

## 🔴 Theory (Expert)
- **Two meanings of "emergent".** The book uses the *task-generalization* sense: a capability
  outside the training objective. The research literature (external context, not in the book)
  also uses a *scaling* sense: a capability that is near-chance in small models and appears
  sharply above some size (Wei et al., 2022, "Emergent Abilities of Large Language Models").
  Schaeffer et al. (2023, "Are Emergent Abilities of LLMs a Mirage?") argue that many sharp jumps
  come from discontinuous metrics such as exact-match accuracy; with smooth metrics like
  per-token [[log-probability]] or [[cross-entropy-loss]], improvement is often gradual.
- **Why next-token prediction implies many skills.** Minimizing cross-entropy on a corpus
  drives $P_\theta$ toward the data distribution. If the corpus contains parallel or
  code-switched text, then the true conditional $P(\text{German word}\mid\text{English context})$
  is part of what must be modeled; a sufficiently expressive model that approximates the
  data distribution well therefore *must* encode translation-like mappings. The skill is
  implicit in the [[maximum-likelihood-estimation]] target, not "free".
- **In-context learning.** Zero/few-shot use (Fig. 1.6) changes only the conditioning prefix,
  never the weights $\theta$. Formally the model computes $P_\theta(y\mid \text{demos},x)$; the
  "learning" is amortized inference inside one forward pass. (The book does not use the term
  "in-context learning" here.)
- **Limits.** Emergent skills are typically weaker than task-specific ones: the book's summary
  notes that LLMs "fine-tuned on custom datasets can outperform general LLMs on specific tasks"
  (book p. 16), which is why stage 3 of the [[llm-building-roadmap]] exists.

## Connections
- Builds on: [[next-token-prediction]], [[pretraining]], [[decoder-only-transformer]], [[maximum-likelihood-estimation]], [[probability-distribution]], [[pretraining-datasets]]
- Leads to: [[zero-shot-learning]], [[few-shot-learning]], [[foundation-model]], [[llm-applications]]
- Related: [[machine-translation]], [[text-completion]], [[gpt-3]], [[fine-tuning]], [[overfitting-and-underfitting]] (generalization)
- People: [[tom-brown]] (GPT-3, "Language Models are Few-Shot Learners"), [[alec-radford]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book pp. 12–14 (definition, translation example, 96 layers / 175B).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7 + Ch. 1 Summary, book pp. 15–16 (emergent properties: classify, translate, summarize; fine-tuned beats general).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 9–10 (Fig. 1.6 zero-/few-shot); §1.3, book p. 7 (few-shot capabilities); §1.5, book p. 11 (GPT-3 data).
- External (flagged above): Wei et al. 2022; Schaeffer et al. 2023 — not cited by the book.
