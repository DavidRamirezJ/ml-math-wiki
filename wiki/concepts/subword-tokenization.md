---
title: Subword Tokenization
type: concept
tags: [nlp, llm, tokenization, byte-pair-encoding]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data]
created: 2026-10-13
updated: 2026-10-13
---

# Subword Tokenization

> **One-liner:** Subword tokenization splits text into pieces that are often whole words but can also be word fragments or single characters, so any word — even a made-up one — can be written with a fixed, finite vocabulary.

## 🟢 Intuition (Dummy)
Think of LEGO. If your box only had complete, pre-built houses (whole words),
you could never build a house you haven't seen before. If it only had 1×1
bricks (single letters), you could build anything, but it would take forever.

Subword tokenization gives you a box with **both**: big ready-made pieces for
common things ("the", " text", " message") and small bricks for the rare stuff.
A strange word like "Akwirw" just gets built from smaller pieces: "Ak" + "w" +
"ir" + "w". You never run out of pieces, and common words still cost just one
piece.

## 🟡 Mechanics (Practitioner)

### Where it sits
The book's first tokenizers (`SimpleTokenizerV1/V2`, §2.3–2.4) split on
spaces and punctuation, so every token is a whole word — and unknown words
either crash the tokenizer or become `<|unk|>` (see
[[out-of-vocabulary-words]]). GPT-2, GPT-3 and the original ChatGPT model
instead use a subword scheme called [[byte-pair-encoding]] (BPE) (book p. 33).

### Using it: tiktoken (book p. 33)
Writing BPE yourself is "relatively complicated", so the book uses OpenAI's
`tiktoken` library (version 0.7.0, core written in Rust):
```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252,
#  18250, 8812, 2114, 286, 617, 34680, 27271, 13]
tokenizer.decode(integers)
# 'Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.'
```
Two things to notice (book p. 34):
1. The vocabulary has **50,257** tokens; `<|endoftext|>` is the largest ID,
   50256.
2. "someunknownPlace" is not a real word, yet it encodes and decodes back
   **perfectly** — no `<|unk|>` needed. (It becomes several subword IDs; the
   exact split, e.g. 617, 34680, 27271, is read off the output, not spelled
   out in the book.)
3. Small typo in the book's listing: the two string pieces `"...terraces"`
   `"of someunknownPlace."` concatenate to "terracesof" (no space), but the
   printed IDs contain 286 (" of") and the printed decode has "terraces of",
   so the printed output corresponds to text *with* the space.

### A worked split: Figure 2.11 / Exercise 2.1 (book p. 34)
```python
ids = tokenizer.encode("Akwirw ier")      # [33901, 86, 343, 86, 220, 959]
[tokenizer.decode([i]) for i in ids]      # ['Ak', 'w', 'ir', 'w', ' ', 'ier']
tokenizer.decode(ids)                     # 'Akwirw ier'
```
| Piece | "Ak" | "w" | "ir" | "w" | " " | "ier" |
|---|---|---|---|---|---|---|
| ID | 33901 | 86 | 343 | 86 | 220 | 959 |

The repeated "w" gets the same ID (86) both times, and the space is a token of
its own (220).

### How the vocabulary is built (book pp. 34–35)
The full algorithm is out of scope in the book; the short version is:
1. Start with **all single characters** ("a", "b", …).
2. **Merge** characters that often appear together into subwords — e.g. "d" +
   "e" → "de", common in "define", "depend", "made", "hidden".
3. Keep merging frequent subwords into larger subwords and words, until the
   counts fall below a **frequency cutoff**.

The same idea as formulas (our LaTeX version of the book's description, not
printed in the book):
$$\mathrm{count}(a,b) = \#\{\, i : x_i = a,\ x_{i+1} = b \,\}$$
$$(a^*, b^*) = \arg\max_{(a,b)} \mathrm{count}(a,b), \qquad V_{t+1} = V_t \cup \{a^*b^*\}$$
Symbol by symbol:
- $x_i$ = the $i$-th token in the training text (at the current merge stage),
- $a, b$ = two tokens already in the vocabulary,
- $\#\{\dots\}$ = "how many positions $i$ satisfy the condition", i.e. how often
  $a$ is directly followed by $b$,
- $\arg\max$ = "the pair that gives the largest count" (the stars mark the winner),
- $V_t$ = the vocabulary after $t$ merges, $\cup$ = "add to the set",
- $a^*b^*$ = the two winning tokens glued into one new token.

Every occurrence of the winning pair in the text is then replaced by the new
token, and the counting repeats.

## 🔴 Theory (Expert)
- **Coverage guarantee.** Because the base vocabulary contains every single
  character (GPT-2: every **byte**, background), every string is a
  concatenation of vocabulary items. Encoding is therefore *total* and
  $\text{decode}(\text{encode}(s)) = s$ — the property that lets the LLM
  "process any text" (book p. 34). See [[tokenizer-encode-decode]].
- **A greedy compression scheme.** Each merge greedily picks the most frequent
  adjacent pair — a [[histogram]] of pair counts followed by an
  [[argmax]]. Frequent strings get short codes, so typical text needs fewer
  tokens than with characters. BPE comes from a 1994 data-compression
  algorithm by Philip Gage that replaced the most frequent byte pair with an
  unused byte (background, not in the book).
- **Spectrum of granularity.** Word-level: short sequences, huge and brittle
  vocabulary (OOV). Character-level: tiny vocabulary, very long sequences
  (costly for $O(T^2)$ attention and for the [[context-length]]). Subwords
  trade these off by choosing the vocabulary size (50,257 for GPT-2) as a
  [[hyperparameter]].
- **Greedy, not optimal.** The merge sequence is not guaranteed to give the
  shortest encoding of a corpus for a given vocabulary size; it is a fast
  heuristic. Other subword methods (WordPiece used by [[bert]], Unigram LM)
  choose pieces by likelihood instead of raw frequency (background).
- **History.** Subword units for neural machine translation were introduced by
  [[rico-sennrich]], Haddow & Birch (2016), adapting BPE; GPT-2 (Radford et
  al., 2019) moved it to the byte level (background, not cited in §2.5).
- **Side effect.** Token boundaries don't match human words: " the" (with a
  leading space) and "the" are different tokens, and numbers or rare names
  split unpredictably. This affects tasks like counting letters or arithmetic
  (background).

## Connections
- Builds on: [[tokenization]], [[token]], [[vocabulary]], [[out-of-vocabulary-words]], [[histogram]]
- Leads to: [[byte-pair-encoding]], [[token-id]], [[token-embedding]], [[sliding-window-sampling]]
- Used by: [[gpt-2]], [[gpt-3]], [[large-language-model]], [[bert]]
- People: [[rico-sennrich]], [[alec-radford]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 33–35 (tiktoken, encode/decode example, Figure 2.11, Exercise 2.1, merge description)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book p. 32 (GPT tokenizers use BPE instead of `<|unk|>`)
