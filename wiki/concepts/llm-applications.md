---
title: Applications of LLMs
type: concept
tags: [llm, nlp, applications, deep-learning]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Applications of LLMs

> **One-liner:** Because an LLM can read and write free-form text, you can point it at almost any job that is "text in, text out" — translating, summarizing, chatting, classifying, searching, even writing code.

## 🟢 Intuition (Dummy)
Most data that older software handled was neat and tidy: tables, forms, numbers in
columns. But a huge amount of what people actually produce is messy, free-form
text — emails, contracts, medical notes, chat messages. That's what the book calls
**unstructured text data**, and the big trick of a [[large-language-model]] is that it
can *parse and understand* it (book p. 4).

Once a machine can read messy text and also write fluent text back, a long list of
jobs opens up at once. Think of it like hiring one very well-read assistant instead
of a separate specialist for every task: the same assistant can translate a letter,
summarize a report, tell you whether a review is happy or angry, or draft a poem.
The book's own summary is that LLMs can automate "almost any task that involves
parsing and generating text," and that the applications are "virtually endless"
(book p. 4). It adds that LLMs may redefine our relationship with technology, making
it more **conversational, intuitive, and accessible** (book p. 4). The book's own goal
follows from this: code an LLM that generates text, then learn the techniques that
let it answer questions, summarize, and translate — i.e. build a ChatGPT-like
assistant step by step (book p. 4).

## 🟡 Mechanics (Practitioner)
**The catalogue the book names (§1.2, book p. 4):**

| Application | What it means (plain gloss) | Book's examples |
|---|---|---|
| [[machine-translation]] | turn text in one language into another | — |
| Generation of novel text | write new text from instructions | Figure 1.2: ChatGPT writing a poem |
| Sentiment analysis | decide whether a text's attitude is positive/negative | — |
| Text summarization | shrink a long text, keep the main points | — |
| Content creation | fiction, articles, even **computer code** | — |
| Chatbots / virtual assistants | conversational systems that answer queries | OpenAI's **ChatGPT**, Google's **Gemini** (formerly Bard); they *augment* search engines like Google Search and Microsoft Bing |
| Knowledge retrieval | dig through huge piles of specialist text | **medicine**, **law**: sifting documents, summarizing long passages, answering technical questions |

(The short glosses in the middle column are the wiki's wording; the book only
*names* translation, sentiment analysis and summarization.)

**Figure 1.2 (book p. 5)** shows the interface pattern behind most of these: a box
labelled *User input (instructions)* and a box labelled *Model output*. Every
application above is the same machine — one that keeps predicting the next
[[token]] — wrapped around a different kind of input.

**How the book turns a raw LLM into an application.** A freshly pretrained model
mostly does [[text-completion]]. To get specific applications, the book fine-tunes it
in two ways (Figure 1.9, book p. 14):
- **[[classification-fine-tuning]]** → a classifier (spam / not spam is the running
  example, Ch. 6). This is how you'd build sentiment analysis or document tagging.
- **[[instruction-fine-tuning]]** → a personal assistant / chat model (Ch. 7). This is
  how you get "translate this", "summarize this", "convert to passive voice".

A tiny bit of math you'll see under the hood of every application: the model
outputs a probability for each candidate next token,
$$P(x_t \mid x_1,\dots,x_{t-1}),$$
- $x_t$ = the token at position $t$ (the one we're predicting),
- $x_1,\dots,x_{t-1}$ = everything that came before it (the "context"),
- the bar "$\mid$" = "given".

Translation, summarization and chat all just *generate* with this rule (see
[[autoregressive-text-generation]]); classification reads off a few class scores
instead (see [[text-classification]]). (This formula is the wiki's notation; §1.2 itself
has no math.)

## 🔴 Theory (Expert)
- **Why one model covers many tasks.** Every task in the table can be phrased as
  "given some text, produce some text" — a conditional distribution over output
  sequences $p(\mathbf y\mid\mathbf x)$. A model trained to approximate
  $p(x_t\mid x_{<t})$ on a broad corpus can represent $p(\mathbf y\mid\mathbf x)$ for any
  task whose input and output are both strings, simply by concatenating
  $\mathbf x$ (prompt) and $\mathbf y$ (answer) into one sequence. This is the deep
  reason the book says applications are "virtually endless" (book p. 4) and
  why GPT models show [[emergent-behavior]] such as translation without being trained
  for it (book p. 14). The book also notes GPT models can do spelling correction,
  classification and translation from prompts alone, in zero-shot or few-shot
  settings (Figure 1.6, book pp. 10, 12) — see [[zero-shot-learning]], [[few-shot-learning]].
- **Generalist vs. specialist trade-off.** Instruction-tuned generalists handle many
  tasks but need more data and compute; classification-tuned specialists are cheaper
  and more accurate on a fixed label set but cannot say anything outside it
  (book p. 171, sidebar "Choosing the right approach").
- **Retrieval caveat.** "Knowledge retrieval" in §1.2 means the model reads and
  condenses large document collections. Production systems usually pair an LLM with
  an explicit search step ([[retrieval-augmented-generation]]); the book does not cover
  RAG in §1.2 — this is a pointer, not a book claim.
- **Custom vs. general models.** §1.3 adds that domain-tailored LLMs (e.g.
  BloombergGPT for finance, medical QA models) can *outperform* general-purpose ones
  on their domain (book p. 5; refs in appendix B, p. 289) — see [[custom-llm]].

## Connections
- Builds on: [[large-language-model]], [[natural-language-processing]], [[next-token-prediction]]
- Leads to: [[fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[custom-llm]]
- Examples of: [[machine-translation]], [[text-classification]], [[autoregressive-text-generation]], [[retrieval-augmented-generation]]
- Related: [[foundation-model]], [[emergent-behavior]], [[generative-ai]]
- Math: [[probability-distribution]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.2, book pp. 4–5 (application list, Figure 1.2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book p. 5 (custom-built LLMs); §1.7, book p. 14 (Figure 1.9, classifier vs. personal assistant).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.1, book pp. 170–171 (generalist vs. specialist sidebar).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book p. 289 (Wu et al. 2023, BloombergGPT; Singhal et al. 2023, "Towards Expert-Level Medical Question Answering with Large Language Models").
