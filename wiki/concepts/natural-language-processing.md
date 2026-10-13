---
title: Natural Language Processing (NLP)
type: concept
tags: [nlp, llm, deep-learning, foundations]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Natural Language Processing (NLP)

> **One-liner:** NLP is the field of getting computers to work with human language: reading it, sorting it, translating it, and writing it.

## 🟢 Intuition (Dummy)
Human language is messy. The same word means different things in different sentences, sarcasm flips
meaning, and grammar has endless exceptions. **NLP** is the whole effort of teaching computers to deal
with that mess: is this review happy or angry? What is this paragraph about? How do you say it in
Spanish? Can you write a polite email from these five keywords?

For a long time computers were good at the easy part (sorting emails into "spam" or "not spam") and
bad at the hard part (really writing or understanding text). [[large-language-model|Large language
models]] changed that, and the book says they "ushered in a new era" for NLP (book p. 1).

## 🟡 Mechanics (Practitioner)

**Definition.** The book never formally defines NLP. The working meaning (a gloss) is: the field of
computational processing and generation of human language.

**Typical NLP tasks named in the book** (book pp. 2, 4):
| Task | What it does | Kind of output |
|---|---|---|
| [[machine-translation]] | text in one language → another language | text |
| sentiment analysis | decide the attitude of a text | a class label |
| question answering | answer a question about text | text |
| text summarization | shorten a long text | text |
| [[text-classification]] (e.g. spam) | assign a category | a class label |
| content creation, chatbots | write new text | text |

**Three eras, as the book tells it.**
1. **Rule-based systems and simpler statistical methods.** Before LLMs, NLP "relied mostly on explicit
   rule-based systems and simpler statistical methods" (book p. 15). These traditional methods were
   good at categorization tasks (e.g. email spam classification) and straightforward pattern
   recognition "that could be captured with handcrafted rules or simpler models" (book p. 1), often
   built on [[feature-extraction|hand-made features]] (book p. 4). They underperformed on tasks
   needing complex understanding and generation: parsing detailed instructions, contextual analysis,
   and writing coherent original text, e.g. an email from a list of keywords (book p. 1).
2. **Task-specific models.** Earlier NLP models were each built for one task (text
   categorization, translation, …) and were good only at that task (book p. 2).
3. **LLMs.** One deep, [[transformer-architecture|transformer-based]] model, [[pretraining|pretrained]]
   on a huge amount of text, shows broad skill across many NLP tasks, and can be
   [[fine-tuning|fine-tuned]] for specific ones (book pp. 2, 15). The §1.7 summary puts it as: LLMs
   "introduced new deep learning-driven approaches that led to advancements in understanding,
   generating, and translating human language" (book p. 15). Stage 2 of the book also covers the
   fundamentals of evaluating LLMs, "essential for developing capable NLP systems" (book p. 15).

**Why LLMs won** (book p. 2): (1) the [[transformer-architecture]], which lets the model pay selective
[[attention-mechanism|attention]] to different parts of the input, and (2) vast amounts of training
data, which capture "linguistic nuances, contexts, and patterns that would be challenging to manually
encode".

**How text becomes math.** Every NLP system must first turn text into numbers. In this book the
pipeline is: text → [[tokenization|tokens]] → [[token-id|token IDs]] (e.g. "Every effort moves you" →
`[6109, 3626, 6100, 345]`) → [[token-embedding|embedding vectors]] → transformer → [[logits]] over a
vocabulary of 50,257 tokens (book p. 131). Generation and classification are both read off those
logits.

## 🔴 Theory (Expert)
- **Language modeling as the unifying task.** Modern NLP folds most tasks into one: estimate
  $$p_\theta(x_1,\dots,x_T)=\prod_{t=1}^{T}p_\theta(x_t\mid x_1,\dots,x_{t-1}),$$
  where $x_t$ = the $t$-th token, $\theta$ = the [[model-parameters]], and
  $p_\theta(x_t\mid\cdot)$ = the predicted probability of the next token given all previous ones
  (the standard formalisation of what book p. 2 describes in words). Translation, QA, summarization
  all become "continue this text" once phrased as a prompt. See [[next-token-prediction]] and
  [[autoregressive-text-generation]].
- **"Understanding" caveat.** The book stresses that LLM "understanding" means producing text that
  *appears* coherent and relevant, not human-like comprehension (book p. 2). Evaluation in NLP is
  therefore behavioural: [[perplexity]], task accuracy, or [[llm-as-a-judge|judged response quality]]
  (§7.8).
- **Why rule-based NLP hit a wall.** Natural language has long-range dependencies and context-sensitive
  meaning that do not compress into a manageable rule set or a fixed bag-of-features. Learned
  representations with attention scale with data instead of with human effort.
- **Classification vs. generation.** Traditional NLP was mostly classification (a [[softmax]] over a
  few classes). Generative NLP is a softmax over the whole vocabulary at every step. The book shows
  both reuse the same [[cross-entropy-loss]]: 50,257 classes in pretraining, 2 classes in spam
  fine-tuning (§6.6).

## Connections
- Builds on: [[artificial-intelligence]], [[machine-learning]], [[deep-learning]]
- Leads to: [[large-language-model]], [[transformer-architecture]], [[tokenization]], [[llm-applications]]
- Used by: [[machine-translation]], [[text-classification]], [[text-completion]]
- Related: [[feature-extraction]], [[next-token-prediction]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 1 intro and §1.1, book pp. 1–4 (PDF pp. 23–26).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.2 (task list), book p. 4 (PDF p. 26).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7 "Summary", book p. 15 (PDF p. 37).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.1 (token-ID example), book p. 131 (PDF p. 153).
