---
title: Token
type: concept
tags: [llm, nlp, tokenization, foundations]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data]
created: 2026-10-13
updated: 2026-10-13
---

# Token

> **One-liner:** A token is one small piece of text — a word, a punctuation mark, or a word-chunk — and it is the unit an LLM reads and writes.

## 🟢 Intuition (Dummy)
An LLM can't read sentences the way you do. First, the text gets chopped into little pieces, like
cutting a sentence into LEGO bricks. Each brick is a **token**. "Hello, world." becomes four bricks:
`Hello`, `,`, `world`, `.`.

The model reads one brick at a time and guesses the next brick. Everything an LLM does — reading
your question, writing its answer, even how much it costs to run — is counted in tokens.

A handy rule of thumb from the book: the number of tokens is *roughly* the number of words plus the
number of punctuation marks.

## 🟡 Mechanics (Practitioner)
**Definition (book §1.5, p. 11):** "a token is a unit of text that a model reads." The number of
tokens in a dataset is roughly the number of words and punctuation characters. Turning text into
tokens is called [[tokenization]].

**Simple word-level tokens (book §2.2, pp. 21–24).** In Chapter 2's first tokenizer, a token is
either a word or a special character (punctuation included). The book splits text with a regular
expression:

```python
import re
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']   -> 10 tokens
```

Written as math, the filter step is:

$$\text{tokens} = \big[\, s.\text{strip}() \;:\; s \in \text{split}(T, P),\; s.\text{strip}() \neq \text{""} \,\big]$$

Symbol by symbol:
- $T$ = the raw text string.
- $P$ = the regex pattern (split on punctuation, `--`, or whitespace).
- $\text{split}(T,P)$ = the list of pieces, *including* the separators (the pattern's parentheses keep them).
- $s$ = one piece. $s.\text{strip}()$ = that piece with spaces trimmed.
- "$s.\text{strip}() \neq$ ''" = "throw away pieces that are only whitespace".

**Worked numbers from the book.**
- The training story, Edith Wharton's *The Verdict*, has **20,479 characters** (p. 22) and gives **4,690 tokens** with this tokenizer, without whitespace (p. 24). That is about $20479/4690 \approx 4.37$ characters per token (derived).
- The first 30 tokens start: `'I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', …, 'genius', '--', 'though', …` (p. 24).
- Case is **kept** (no lowercasing): capital letters help the model tell proper nouns from common nouns, understand sentence structure, and learn to write with correct capitals (p. 23).
- Whitespace is **dropped** here for simplicity. Keeping it matters for text where spacing is meaningful, like Python code (p. 23).

**Where tokens sit in the pipeline (Fig. 2.4, p. 21):**

```
"This is an example."  →  [This][is][an][example][.]  →  40134 2052 133 389 12  →  vectors  →  GPT
      input text              tokens                       token IDs           token embeddings
```

Each token is mapped to an integer [[token-id]] using a [[vocabulary]], and each ID is then turned
into a [[vector]] by an [[embedding-layer]] (§2.3, §2.7).

**Tokens as a measure of scale (book §1.5, Table 1.1, p. 11).** Dataset size is counted in tokens.
GPT-3's sources total **499 billion tokens**, and GPT-3 was trained on **300 billion** of them.
CommonCrawl alone is 410B tokens ≈ 570 GB, i.e. about $570\text{ GB}/410\text{B} \approx 1.4$ bytes
per token (derived).

## 🔴 Theory (Expert)
- **Tokens as a finite alphabet.** Tokenization is a map from character strings to sequences over a
  finite set $\mathcal{V}$ (the vocabulary). An LLM then models a distribution over sequences in
  $\mathcal{V}^*$, factored token by token:
  $p(x_1,\dots,x_T)=\prod_{t} p(x_t\mid x_{<t})$. Here $x_t \in \mathcal{V}$ is the token at position
  $t$ and $x_{<t}$ means "all tokens before $t$". The choice of tokens therefore *defines* the
  sample space of the [[probability-distribution]] the model learns ([[next-token-prediction]]).
- **Granularity trade-off.** Character-level tokens give a tiny vocabulary but very long sequences
  (more compute per sentence, harder [[long-range-dependencies]]). Word-level tokens give short
  sequences but a huge vocabulary and the [[out-of-vocabulary-words]] problem (a word not seen
  during vocabulary building has no ID). [[subword-tokenization]] such as [[byte-pair-encoding]]
  (used by GPT-2/GPT-3, vocabulary 50,257, §2.5) sits in between and can encode *any* string.
- **Special tokens.** Not every token comes from text. Markers like `<|unk|>` (unknown word) and
  `<|endoftext|>` (boundary between documents) are added to the vocabulary on purpose (§2.4; see
  [[special-tokens]], [[end-of-text-token]]).
- **Token count is the real unit of cost.** [[context-length]] (how much the model can see at once),
  training budget (e.g. GPT-3's 300B), and inference cost are all measured in tokens, not words. The
  "words + punctuation" estimate is only rough; with subword tokenizers, rare words split into
  several tokens.
- **Tokenization is not lossless by default.** The book's regex tokenizer throws away whitespace, so
  `decode(encode(text))` cannot rebuild the exact spacing. BPE keeps whitespace inside tokens and
  round-trips exactly.

## Connections
- Builds on: [[natural-language-processing]], [[vocabulary]]
- Leads to: [[tokenization]], [[token-id]], [[token-embedding]], [[byte-pair-encoding]], [[subword-tokenization]], [[special-tokens]]
- Used by: [[large-language-model]], [[next-token-prediction]], [[context-length]], [[pretraining-datasets]], [[sliding-window-sampling]]
- Related: [[vector]], [[embedding-layer]], [[out-of-vocabulary-words]]
- People: [[rico-sennrich]] (BPE for NLP), [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book p. 11 (definition of token; Table 1.1 token counts).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.2, book pp. 21–24 (word-level tokens, regex tokenizer, 20,479 chars p. 22 → 4,690 tokens p. 24, Figs. 2.4–2.5).
