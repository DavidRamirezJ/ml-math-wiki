---
title: Tokenizer Encode and Decode
type: concept
tags: [nlp, llm, tokenization, preprocessing]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch05-pretraining-on-unlabeled-data, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Tokenizer Encode and Decode

> **One-liner:** A tokenizer has two jobs: *encode* turns text into a list of numbers, and *decode* turns a list of numbers back into text.

## 🟢 Intuition (Dummy)
Think of a secret-code book. On the left page, every word has a number next to
it ("brown = 0, dog = 1, fox = 2, …"). To **encode** a sentence, you look up
each word and write down its number. To **decode**, you flip the book around
(the *inverse* book, sorted by number) and turn each number back into its word.

An LLM only ever sees the numbers. So we encode the text on the way *in* and
decode the model's numbers on the way *out*. Every tokenizer you will ever meet
— the toy one in the book, or the real GPT-2 one — has exactly these two
methods (book p. 28).

## 🟡 Mechanics (Practitioner)

### Step 1: build a vocabulary
The book builds the code book from its training text, Edith Wharton's short
story "The Verdict" (book pp. 24–26, Figure 2.6, Listing 2.2):
1. Tokenize the whole training text (4,690 tokens).
2. Remove duplicates.
3. Sort the unique tokens alphabetically.
4. Give each token its position in the sorted list as its [[token-id]].

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)            # 1130
vocab = {token: integer for integer, token in enumerate(all_words)}
# ('!', 0), ('"', 1), ("'", 2), ..., ('Her', 49), ('Hermia', 50)
```
Punctuation comes first and capitalised words come before lowercase ones,
because Python sorts strings by character code.

In math notation, the [[vocabulary]] is a function:
$$\text{vocab}: V \to \{0,1,\dots,|V|-1\}$$
Read it symbol by symbol:
- $V$ = the set of unique tokens in the training text,
- $|V|$ = how many there are (the vocabulary size, here $1{,}130$),
- $\to$ = "maps each element on the left to one element on the right",
- $\{0,1,\dots,|V|-1\}$ = the integers from 0 up to 1,129.

### Step 2: the inverse vocabulary
To decode, we need the map going the other way (book p. 26):
$$\text{vocab}^{-1}: \{0,\dots,|V|-1\} \to V,\qquad \text{vocab}^{-1}(\text{vocab}(t)) = t$$
- $\text{vocab}^{-1}$ = the inverse vocabulary ("vocab inverse"),
- $t$ = any one token,
- the equation says "look up $t$'s number, then look that number up in the
  inverse book, and you get $t$ back".

This inverse exists because every token got a *different* number (the map is
one-to-one). In code it is one line that swaps keys and values.

### Step 3: SimpleTokenizerV1 (Listing 2.3, book p. 27)
```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab                           # vocabulary
        self.int_to_str = {i: s for s, i in vocab.items()}  # inverse vocabulary

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)   # drop space before punctuation
        return text
```
- `encode`: string → split with a regex → strip → look up each piece → list of ints.
- `decode`: list of ints → look up each ID in the inverse map → join with spaces
  → remove the space before punctuation (`\1` keeps the punctuation mark).

**Worked example** (book pp. 27–28):
```python
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
tokenizer.decode(ids)
# '" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
```
You can read some IDs straight off the vocabulary: `1` = `"`, `2` = `'`, `5` =
`,` (twice), `7` = `.` (twice). Small catch the book does not point out: the
round trip adds stray spaces (`" It' s`), because `decode` only removes spaces
*before* punctuation.

Figure 2.8 (book p. 28) shows the same idea on the toy sentence "The brown dog
playfully chased the swift fox": `encode` → `7, 0, 1, …`, `decode` → back to text.

### Step 4: V1 breaks on new words
```python
tokenizer.encode("Hello, do you like tea?")   # KeyError: 'Hello'
```
"Hello" is not in "The Verdict", so it is not a key in the dictionary (book
p. 28). This is the [[out-of-vocabulary-words]] problem. **SimpleTokenizerV2**
(Listing 2.4, book p. 31) fixes it by adding [[special-tokens]]: one new
line in `encode` replaces any unknown piece with `"<|unk|>"` (V2 also adds `:`
and `;` to the split and punctuation-cleanup regexes), and the vocabulary
grows from 1,130 to 1,132 entries.

### Step 5: the real thing — tiktoken's GPT-2 tokenizer (book p. 33)
```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
strings  = tokenizer.decode(integers)
```
Same two methods, but now a [[byte-pair-encoding]] vocabulary of 50,257 tokens
and no unknown words at all (see [[subword-tokenization]]).

### In the training pipeline (Ch. 5)
Chapter 5 wraps the two methods in helpers: `text_to_token_ids(text,
tokenizer)` encodes and adds a batch dimension (shape $[1, T]$), and
`token_ids_to_text(ids, tokenizer)` removes it and decodes (§5.1, book
pp. 131–132; `.unsqueeze(0)` adds and `.squeeze(0)` removes the batch
dimension). These sit on either side of every
[[autoregressive-text-generation]] call.

## 🔴 Theory (Expert)
- **Encode as a function on strings.** Write $\Sigma^*$ for all finite strings.
  `encode` $: \Sigma^* \to \mathbb{N}^*$ (finite sequences of IDs), and `decode`
  $: \mathbb{N}^* \to \Sigma^*$. The ideal property is that decode is a **left
  inverse** of encode: $\text{decode}(\text{encode}(s)) = s$ for every string $s$.
- **V1 is a partial function.** It is defined only on strings whose pieces are
  all in $V$; outside that domain it raises `KeyError`. Its round trip is only
  "identity up to whitespace", because the regex split throws away spacing
  information that `decode` cannot rebuild.
- **V2 is total but lossy.** Mapping every unknown word to `<|unk|>` makes
  encode many-to-one, so no left inverse can exist:
  $\text{decode}(\text{encode}(\text{"Hello"})) = \text{"<|unk|>"}$ (book p. 32).
- **BPE is total and lossless.** GPT-2's byte-level BPE keeps spaces as part of
  tokens (e.g. ID 220 is a lone space) and can fall back to single bytes, so
  $\text{decode}\circ\text{encode} = \text{id}$ on all text (book pp. 33–34).
  Encode is still not surjective onto $\mathbb{N}^*$ in a canonical sense: many
  ID sequences decode to the same string, but only one is what encode produces.
- **Why IDs are only an intermediate step.** IDs are arbitrary labels (the sort
  order means nothing). They become meaningful only after the
  [[embedding-layer]] uses them as row indices into a learned [[matrix]] —
  equivalent to multiplying a [[one-hot-encoding|one-hot]] [[vector]] by that
  matrix (book p. 24 calls IDs "an intermediate step before converting the
  token IDs into embedding vectors"; the one-hot equivalence is the NOTE in
  §2.7, book p. 43).

> **From [[llm-ch04-implementing-a-gpt-model]]:** generation wraps the model in an encode/decode pair. "Hello, I am" → `[15496, 11, 314, 716]`; `.unsqueeze(0)` adds a batch axis (shape `[1, 4]`); after [[greedy-decoding]], `.squeeze(0)` removes it and `decode` turns the IDs back into text (book pp. 125–127).

## Connections
- Builds on: [[tokenization]], [[token]], [[vocabulary]], [[token-id]]
- Leads to: [[out-of-vocabulary-words]], [[special-tokens]], [[byte-pair-encoding]], [[subword-tokenization]], [[token-embedding]], [[embedding-layer]]
- Used by: [[sliding-window-sampling]], [[autoregressive-text-generation]], [[pytorch-dataset-and-dataloader]], [[large-language-model]]
- People: — (training text by Edith Wharton; no ML contributor named)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.3, book pp. 24–28 (Figures 2.6–2.8, Listings 2.2–2.3)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book pp. 29–32 (Listing 2.4, SimpleTokenizerV2)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 33–34 (tiktoken encode/decode)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7, book p. 43 (embedding = one-hot × matrix NOTE)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book pp. 131–132 (`text_to_token_ids`, `token_ids_to_text`)
- Same PDF, §4.1, book p. 97 (tiktoken `encode` of two texts, stacked into a batch); §4.7, book pp. 125–127 (`encode` → `unsqueeze(0)` adds a batch dimension, shape [1, 4]; `decode(out.squeeze(0).tolist())` → "Hello, I am Featureiman Byeswickattribute argue").
