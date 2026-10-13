---
title: Tokenization
type: concept
tags: [nlp, llm, preprocessing, tokenization]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Tokenization

> **One-liner:** Tokenization chops text into small pieces called tokens and gives each piece a number, so a language model can read it.

## 🟢 Intuition (Dummy)
A model can't read letters on a page. Before anything else, we cut the text
into bite-sized pieces — words, punctuation marks, or chunks of words — like
cutting a sandwich into squares before a toddler eats it. Each piece is a
**[[token]]**.

Then we look each piece up in a dictionary that says "this piece is number 7,
that piece is number 0…". Now the sentence is a list of numbers (**token
IDs**), and numbers are something a computer can work with. At the very end,
when the model answers, we run the dictionary backwards and turn numbers back
into text.

Tokenization is the **very first step** of the whole LLM pipeline (book
p. 21):

```
input text → tokenized text → token IDs → token embeddings → GPT model → postprocessing → output text
"This is an example."  → This | is | an | example | .  → 40134 2052 133 389 12 → …
```

(Fig. 2.4 / Fig. 2.15, book pp. 21, 41.)

## 🟡 Mechanics (Practitioner)

### What a token is
In Ch. 1 the book says the number of tokens is *roughly* the number of words
plus punctuation characters (book p. 11) — e.g. GPT-3 was trained on 300
billion tokens. In the simple scheme of §2.2 a token is "an individual word or
a special character, including punctuation" (book p. 21). Real GPT tokenizers
use subword tokens instead ([[byte-pair-encoding]]).

### Step 1 — split the text (regex tokenizer, §2.2)
Training text: Edith Wharton's public-domain short story *The Verdict*,
**20,479 characters** (book p. 22). *Scale note (book p. 22):* real LLM training
processes millions of articles and hundreds of thousands of books — many gigabytes of
text. One short story is enough to show every pipeline step and still runs on a laptop.
The book refines a regular-expression
split in three moves (book pp. 22–24):

```python
import re
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']   (10 tokens, Fig. 2.5)
```

- The parentheses `( … )` in the pattern keep the separators as tokens.
- `if item.strip()` throws away whitespace-only and empty strings.
- Text is **not lowercased**: capital letters help the model tell proper nouns
  from common nouns, understand sentence structure, and learn to write with
  proper capitalisation (book p. 23).
- Whitespace: dropping it saves memory and compute; keeping it matters when
  exact layout counts, e.g. Python code indentation (book p. 23).

On the whole story this gives **4,690 tokens** (book p. 24) — about 4.4
characters per token (our arithmetic: $20{,}479/4{,}690\approx 4.37$).

As a formula (our notation):

$$\text{tokens} = \big[\, s.\text{strip}() \;:\; s \in \text{split}(T, P),\ s.\text{strip}() \neq \text{""} \,\big]$$

- $T$ = the raw text string.
- $P$ = the regex pattern.
- $\text{split}(T,P)$ = the list of pieces, separators included.
- $s$ = one piece; "$:$" reads "such that"; the condition keeps only non-empty pieces.

### Step 2 — build a vocabulary and map tokens to IDs (§2.3)
Collect the unique tokens, sort them, and number them (Fig. 2.6, book p. 25):

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)                       # 1130
vocab = {token: integer for integer, token in enumerate(all_words)}
```

*The Verdict* has a [[vocabulary]] of **1,130** unique tokens; the first entries
are `('!', 0)`, `('"', 1)`, `("'", 2)`, … (book pp. 25–26). In math:

$$\text{vocab}: V \to \{0, 1, \dots, |V|-1\}, \qquad t \mapsto \operatorname{rank}(t)$$

- $V$ = the set of unique tokens; $|V|$ = how many there are (1,130).
- $t$ = one token; $\operatorname{rank}(t)$ = its position after sorting, counting from 0.
- This is a one-to-one map, so it has an inverse $\text{vocab}^{-1}$: ID → token, which is the **inverse vocabulary** used for decoding (book p. 26).

### Step 3 — encode and decode (§2.3)
Every tokenizer has the same two methods (Fig. 2.8, book p. 28) — see
[[tokenizer-encode-decode]]:

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)   # remove space before punctuation
```

Encoding `"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.`
gives `[1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]`
(book p. 27). But `"Hello, do you like tea?"` crashes with `KeyError: 'Hello'`
— "Hello" never appears in the story: the [[out-of-vocabulary-words]] problem
(book p. 28).

### Step 4 — special tokens (§2.4)
Add two [[special-tokens]] to the vocabulary, which grows to **1,132**
(book p. 30):
- `<|unk|>` (ID 1131) replaces any unknown word.
- `<|endoftext|>` (ID 1130) is put between unrelated documents (Fig. 2.10) —
  see [[end-of-text-token]].

`SimpleTokenizerV2` encodes
`"Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."` as
`[1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]`
and decodes it as `<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.`
— "Hello" and "palace" are lost (book pp. 31–32). Other LLMs use `[BOS]`,
`[EOS]`, `[PAD]`; GPT uses only `<|endoftext|>` (also as padding) and no
`<|unk|>` (book p. 32).

### Step 5 — the real thing: BPE with tiktoken (§2.5)
GPT-2/GPT-3/original ChatGPT use [[byte-pair-encoding]], via OpenAI's
`tiktoken` library (book p. 33):

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
ids = tokenizer.encode("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
                       "of someunknownPlace.", allowed_special={"<|endoftext|>"})
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250,
#  8812, 2114, 286, 617, 34680, 27271, 13]
```

Vocabulary 50,257; `<|endoftext|>` = 50256; unknown words get split into
subwords, so no `<|unk|>` is needed (book p. 34). On *The Verdict*, BPE gives
**5,145 tokens** (book p. 35) — more than the 4,690 regex tokens. Our
explanation (not in the book): less common words are split into several
subword tokens, and characters such as line breaks, which the regex tokenizer
dropped, become tokens.

### How tokenization is used in the rest of the book
- **Training pairs** (§2.6): the token-ID sequence is cut with a
  [[sliding-window-sampling|sliding window]] into inputs and targets shifted by
  one.
- **Generation** (§4.7, §5.1): `"Hello, I am"` → `[15496, 11, 314, 716]` (book
  p. 124); helper functions `text_to_token_ids` (adds a batch dimension with
  `unsqueeze(0)`) and `token_ids_to_text` (book pp. 131–132). Generation can
  stop early when the `eos_id` token appears (book pp. 157–158).
- **Classification fine-tuning** (§6.3): SMS messages are tokenized and
  [[padding-and-truncation|padded]] with 50256 to the longest training message,
  **120 tokens** (book pp. 176–178).
- **Instruction fine-tuning** (§7.3): prompts are pretokenized, padded with
  50256 to the longest sequence *in each batch*, and in the targets every
  padding token **except the first** (kept as the end-of-text signal) is
  replaced by −100 so the loss ignores it (book pp. 212–219; see
  [[ignore-index-loss-masking]]).
- **Evaluation** (§7.5): the fine-tuned model's responses are generated with
  the same `text_to_token_ids` / `token_ids_to_text` helpers from Ch. 5
  (book p. 228).

## 🔴 Theory (Expert)
- **Tokenization as a function.** Let $\Sigma^*$ be all strings over the
  character alphabet. A tokenizer is $\text{enc}:\Sigma^*\to\{0,\dots,|V|-1\}^*$
  (a string goes to a finite sequence of IDs), and $\text{dec}$ goes back.
  A good tokenizer is **lossless**: $\text{dec}(\text{enc}(s)) = s$.
  - `SimpleTokenizerV1` is only *partial*: $\text{enc}$ is undefined outside the
    training words (the `KeyError` = applying a function outside its domain).
  - `SimpleTokenizerV2` makes $\text{enc}$ total with `<|unk|>`, but it becomes
    many-to-one, so $\text{dec}\circ\text{enc}\neq\text{id}$ (lossy).
  - Even V1 isn't perfectly lossless on whitespace: the decoded string is
    `" It' s the last …` (book p. 28) because decode only removes spaces before
    punctuation (our observation).
  - BPE over bytes is total *and* lossless (book pp. 33–34).
- **Why it matters for the model.** The tokenizer fixes the vocabulary size $V$,
  which fixes the number of rows of the [[embedding-layer]] ($V\times d$) and the
  width of the output [[logits]] ($V$ = 50,257 in GPT-2). The tokenizer is part
  of the model: you must use the same one at training and inference — GPT-2's
  ships as `encoder.json` + `vocab.bpe` with the weights (book p. 162).
- **Sequence length vs vocabulary size.** Coarser tokens (bigger $V$) mean
  shorter sequences, so more text fits in the [[context-length]]; finer tokens
  (smaller $V$, down to characters) mean no OOV words but longer sequences and
  more attention cost.
- **Language modelling factorisation.** Once text is a token sequence
  $t_1,\dots,t_T$, an LLM models
  $P(t_1,\dots,t_T)=\prod_{i=1}^{T}P(t_i\mid t_{<i})$ — the chain rule of
  probability behind [[next-token-prediction]]. Every probability the model
  talks about is *per token*, so all losses ([[cross-entropy-loss]],
  [[perplexity]]) depend on the tokenizer.
- **Order of the vocabulary.** Python sorts by Unicode code point, so
  punctuation < uppercase < lowercase (hence `('!', 0)` first, `('Hermia', 50)`
  before lowercase words; book p. 26). The ordering is arbitrary for the model:
  any permutation of IDs gives an equivalent model after permuting embedding rows.
- **Alternatives** the book points to, "used by some other popular LLMs"
  (book p. 291): SentencePiece (Kudo & Richardson 2018) and WordPiece (Song et
  al. 2020, "Fast WordPiece Tokenization"; its use in [[bert]] is background,
  not stated in the book).

## Connections
- Builds on: [[token]], [[natural-language-processing]]
- Leads to: [[token-id]], [[vocabulary]], [[tokenizer-encode-decode]], [[token-embedding]], [[word-embeddings]], [[sliding-window-sampling]]
- Related: [[byte-pair-encoding]], [[subword-tokenization]], [[special-tokens]], [[end-of-text-token]], [[out-of-vocabulary-words]], [[padding-and-truncation]], [[dynamic-padding]], [[context-length]], [[ignore-index-loss-masking]]
- Used by: [[pretraining]], [[autoregressive-text-generation]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[large-language-model]]
- People: [[rico-sennrich]], [[andrej-karpathy]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book p. 11 (token, tokenization named; token counts).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book pp. 17–20 (chapter roadmap).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.2, book pp. 21–24 (Fig. 2.4 pipeline, regex tokenizer, 20,479 chars → 4,690 tokens).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.3, book pp. 24–28 (vocabulary 1,130, Listing 2.3, KeyError).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book pp. 29–32 (special tokens, Listing 2.4, vocab 1,132).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 33–35 (BPE, tiktoken).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6, book p. 35 (5,145 BPE tokens).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 122–126 (encode "Hello, I am").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book pp. 131–132 (Listing 5.1 helpers); §5.3, book p. 158 (`eos_id`); §5.5, book p. 162 (`encoder.json`, `vocab.bpe`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3, book pp. 175–180 (padding with 50256, max length 120).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 211–219 (pretokenizing, padding, −100 targets); §7.5, book p. 228.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 290–291 (further reading).
