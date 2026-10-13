---
title: Large Language Model (LLM)
type: concept
tags: [llm, deep-learning, nlp, transformer, generative-ai]
difficulty: beginner
aliases: [LLM, LLMs, large language models]
sources: [llm-ch01-understanding-large-language-models, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Large Language Model (LLM)

> **One-liner:** An LLM is a very big neural network that has read a huge amount of text and learned to guess the next word so well that it can write, translate, summarize, and answer questions.

## 🟢 Intuition (Dummy)
Imagine a friend who has read a big part of the internet, plus piles of books
and Wikipedia. You start a sentence — "Breakfast is the…" — and they finish it:
"…most important meal of the day." That is basically what an LLM does. It is a
machine that is *extremely* good at one game: **guess the next word**.

The surprise is how far that one simple game takes you. To guess the next word
really well, the model has to pick up grammar, facts, style, and even how to
translate between languages. So after enough practice, the same "next-word
guesser" can write emails from a few keywords, translate, summarize, or chat with
you like ChatGPT.

Two things make it **large**:
1. **Lots of knobs.** The model has tens or even hundreds of *billions* of
   adjustable numbers ([[model-parameters|parameters]]).
2. **Lots of reading.** It trains on a giant pile of text — sometimes a big part
   of all the public text on the internet.

A quick honesty note from the book: when we say an LLM "understands", we mean it
produces text that *looks* coherent and relevant. It does **not** mean it has
human-like consciousness or comprehension.

Where it sits in the family tree (book Fig. 1.1):

```
Artificial intelligence  ─ systems with human-like intelligence
 └─ Machine learning     ─ algorithms that learn rules from data
     └─ Deep learning    ─ ML with neural networks of many layers
         ├─ LLMs         ─ deep nets that parse and generate human-like text
         └─ GenAI        ─ deep nets that create new content (overlaps LLMs)
```

## 🟡 Mechanics (Practitioner)

### What an LLM is, precisely
The book's definition: an LLM is a **neural network designed to understand,
generate, and respond to human-like text** — a [[deep-learning|deep neural
network]] trained on massive amounts of text (§1.1, p. 2). Almost all modern LLMs
use the [[transformer-architecture]], which lets them *pay selective attention*
to different parts of the input ([[attention-mechanism]]). In the book, "LLM"
means a transformer-based, GPT-like model (sidebar, p. 10).

### The one training objective: next-word prediction
The parameters are tuned so the model predicts the next word (really, the next
[[token]]) in a sequence. Written as math (the book says this in words; the
formula is standard notation, implemented in Ch. 5):

$$p_\theta(x_1,\dots,x_T)=\prod_{t=1}^{T} p_\theta(x_t \mid x_1,\dots,x_{t-1})$$

Reading it symbol by symbol:
- $x_1,\dots,x_T$ = the tokens of a text, in order; $T$ = how many tokens there are.
- $\theta$ (theta) = all the model's parameters (its weights).
- $p_\theta(\cdot)$ = "the probability the model with weights $\theta$ assigns to…".
- $x_t \mid x_1,\dots,x_{t-1}$ = "token number $t$, **given** all the tokens before it". The bar $\mid$ means "given".
- $\prod_{t=1}^{T}$ = "multiply together, for $t$ from 1 to $T$".

So: the probability of a whole text = (prob. of word 1) × (prob. of word 2 given
word 1) × (prob. of word 3 given words 1–2) × … Training pushes $\theta$ to make
real text more probable. Full details on [[next-token-prediction]].

### How it is built: two (really three) stages
The book's roadmap (§1.3 Fig. 1.3, §1.7 Fig. 1.9):

| Stage | What happens | Data | Result |
|---|---|---|---|
| 1. Build | data preparation & sampling, attention mechanism, LLM architecture | — | an untrained model |
| 2. [[pretraining]] | next-word prediction ([[self-supervised-learning]]) | raw, *unlabeled* text — "trillions of words" (Fig. 1.3) | a **base / [[foundation-model]]** (e.g. [[gpt-3]]) |
| 3. [[fine-tuning]] | train further on a smaller *labeled* dataset | instruction–answer pairs, or texts + class labels | a chat assistant ([[instruction-fine-tuning]]) or a classifier ([[classification-fine-tuning]]) |

A freshly pretrained base model can already do [[text-completion]] and has some
[[few-shot-learning|few-shot]] ability (§1.3, p. 7). Pretraining a GPT-like LLM
from scratch costs "thousands to millions of dollars" in compute (§1.7, p. 15),
so the book also shows how to load openly available [[pretrained-weights]].

### Size, in real numbers
- Typical LLMs: **tens to hundreds of billions** of parameters (p. 2).
- [[gpt-3]]: **96** transformer layers, **175 billion** parameters, introduced 2020 (§1.6, p. 13).
- The model the book codes, [[gpt-2]] small, uses this configuration (§4.1, p. 95):

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # BPE vocabulary size
    "context_length": 1024,  # max tokens the model can look at
    "emb_dim": 768,          # each token becomes a 768-dim vector
    "n_heads": 12,           # attention heads
    "n_layers": 12,          # transformer blocks
    "drop_rate": 0.1,        # dropout rate
    "qkv_bias": False        # bias in the query/key/value layers
}
```

What a "parameter" is: one trainable number. A single $2048\times2048$ weight
[[matrix]] already holds $2048\times2048=4{,}194{,}304$ parameters (§4.1, p. 94).

### What LLMs are used for (§1.2)
[[machine-translation]], generating new text, sentiment analysis, text
summarization, content creation (fiction, articles, even code), chatbots and
virtual assistants (ChatGPT, Google's Gemini — formerly Bard), and knowledge
retrieval in specialized areas like medicine or law. In short: "almost any task
that involves parsing and generating text" (p. 4). See [[llm-applications]].

### Why build your own? (§1.3)
[[custom-llm|Custom-built LLMs]] tuned for a domain can beat general ones —
e.g. BloombergGPT for finance, or LLMs for medical question answering. Other
benefits: data privacy (no sharing with a third-party provider), running on a
laptop or phone (lower latency and server cost), and full control over updates.

## 🔴 Theory (Expert)
- **Why next-word prediction is enough.** Pretraining is
  [[maximum-likelihood-estimation]] of an autoregressive model: maximizing
  $\sum_t \log p_\theta(x_t\mid x_{<t})$ is the same as minimizing the average
  [[cross-entropy-loss]]. The chain rule of probability makes the factorization
  above *exact* for any distribution over sequences, so nothing is lost by
  modeling one conditional at a time — the difficulty is entirely in learning
  the conditionals.
- **Emergent behavior.** GPT models were never trained to translate, yet they can
  (§1.6, p. 14). The book calls this [[emergent-behavior]]: abilities that appear
  from exposure to vast, multilingual, diverse data rather than from a dedicated
  objective. The summary lists classification, translation and summarization as
  emergent properties (p. 15).
- **Encoder vs. decoder families.** The original transformer has an encoder and a
  decoder. [[bert]] keeps the encoder and trains by
  [[masked-language-modeling|masked word prediction]] (good for classification);
  GPT keeps only the decoder and generates left-to-right
  ([[decoder-only-transformer]]). Text-generating LLMs like GPT-3 and ChatGPT are
  decoder-only (summary, p. 15).
- **Transformers ≠ LLMs.** Not every transformer is an LLM (transformers are also
  used in computer vision), and not every LLM is a transformer — some use
  recurrent or convolutional designs for better efficiency (sidebar, p. 10;
  App. B cites RWKV, Hyena, Mamba). It is still open whether these can match
  transformer LLMs.
- **Self-supervision is the scaling trick.** Because the label for each position
  is just the next token in the text, labels come "for free", so the training set
  can be essentially all available text (§1.3 note, p. 6; §1.6, p. 12). Labeled
  data is only needed for the much smaller fine-tuning stage.
- **Why it works, per the book.** Next-word prediction is sensible because it
  harnesses the inherent sequential nature of language (§1.1, p. 2). The book
  attributes LLM success to two things: the transformer architecture and the vast
  training data, which let LLMs capture linguistic nuances, contexts and patterns
  that would be hard to encode manually (p. 2).
- **Still relevant architecture.** GPT-3 (2020) is "a long time ago" by deep
  learning standards, but newer models such as Meta's Llama keep the same
  underlying concepts with only minor modifications (§1.6, p. 13).
- **Context.** Before LLMs, NLP leaned on explicit rule-based systems and simpler
  statistical methods, and models were task-specific; LLMs are broadly proficient
  across tasks (§1.1, p. 2; summary, p. 15).

## Connections
- Builds on: [[deep-learning]], [[neural-network]], [[transformer-architecture]], [[attention-mechanism]], [[probability-distribution]], [[maximum-likelihood-estimation]]
- Leads to: [[next-token-prediction]], [[pretraining]], [[fine-tuning]], [[foundation-model]], [[instruction-fine-tuning]], [[classification-fine-tuning]], [[llm-building-roadmap]], [[emergent-behavior]]
- Related: [[generative-ai]], [[natural-language-processing]], [[gpt-architecture]], [[gpt-2]], [[gpt-3]], [[bert]], [[token]], [[tokenization]], [[model-parameters]], [[llm-applications]], [[custom-llm]]
- People: [[alec-radford]], [[tom-brown]], [[ashish-vaswani]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 1 intro + §1.1, book pp. 1–4 (PDF pp. 23–26) — definition, "large", next-word objective, Fig. 1.1.
- Same file §1.2, book pp. 4–5 — applications, Fig. 1.2.
- Same file §1.3, book pp. 5–7 — custom LLMs, pretraining + fine-tuning, Fig. 1.3.
- Same file §1.4, book pp. 7–10 — transformer, BERT vs GPT, zero/few-shot, sidebar "Transformers vs. LLMs".
- Same file §1.6, book pp. 12–14 — GPT-3 (96 layers, 175B params), emergent behavior.
- Same file §1.7 + Ch. 1 summary, book pp. 14–16 — three-stage roadmap, Fig. 1.9.
- Same file §4.1, book pp. 93–95 — parameters, `GPT_CONFIG_124M`.
