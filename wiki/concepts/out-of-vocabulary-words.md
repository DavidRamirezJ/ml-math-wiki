---
title: Out-of-Vocabulary Words and the <|unk|> Token
type: concept
tags: [nlp, llm, tokenization, vocabulary]
difficulty: beginner
sources: [llm-ch02-working-with-text-data]
created: 2026-10-13
updated: 2026-10-13
---

# Out-of-Vocabulary Words and the <|unk|> Token

> **One-liner:** An out-of-vocabulary (OOV) word is a word the tokenizer has never seen; old tokenizers replace it with a catch-all `<|unk|>` token, while GPT's tokenizer breaks it into smaller known pieces instead.

## 🟢 Intuition (Dummy)
Imagine your code book only lists the 1,130 words from one short story. Someone
hands you the sentence "Hello, do you like tea?" — and "Hello" isn't in the
book. What now?

There are three options:
1. **Give up** (crash). That is what the book's first tokenizer does.
2. **Write "UNKNOWN"** in that spot. You can keep going, but you've forgotten
   what the word was. That's the `<|unk|>` token.
3. **Spell it out** from smaller pieces you *do* know ("Hel" + "lo", or even
   single letters). Nothing gets lost. That's what GPT's [[byte-pair-encoding]]
   tokenizer does.

## 🟡 Mechanics (Practitioner)

### Option 1: crash — SimpleTokenizerV1 (book p. 28)
```python
tokenizer.encode("Hello, do you like tea?")
# KeyError: 'Hello'
```
"Hello" never appears in "The Verdict", so it is not a key of the vocabulary
dictionary. The book's lesson: LLMs need **large and diverse training sets** to
grow the [[vocabulary]].

### Option 2: `<|unk|>` — SimpleTokenizerV2 (book pp. 29–32)
Add two [[special-tokens]] to the end of the vocabulary:
```python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
len(vocab)   # 1132  (was 1130)
# ..., ('yourself', 1129), ('<|endoftext|>', 1130), ('<|unk|>', 1131)
```
and one new line in `encode`:
```python
preprocessed = [item if item in self.str_to_int else "<|unk|>"
                for item in preprocessed]
```
As a formula, the encoder now does:
$$\text{enc}(w) = \begin{cases} \text{vocab}(w) & \text{if } w \in V \\ \text{vocab}(\texttt{<|unk|>}) = 1131 & \text{if } w \notin V \end{cases}$$
Symbol by symbol:
- $w$ = one word/piece from the text,
- $V$ = the set of known tokens, $w \in V$ = "$w$ is in the book", $w \notin V$ = "it is not",
- $\text{vocab}(w)$ = the ID the vocabulary gives $w$,
- the big brace = "pick the line whose condition is true".

**Worked example** (book pp. 31–32):
```python
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
tokenizer.encode(text)
# [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
tokenizer.decode(tokenizer.encode(text))
# '<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.'
```
Both "Hello" and "palace" became 1131: they are not in "The Verdict". Figure
2.9 (book p. 29) shows the same thing on a toy vocabulary, where "playfully"
gets the `<|unk|>` ID 783.

### Option 3: no unknowns at all — BPE (book pp. 33–34)
GPT models do **not** use `<|unk|>` (book p. 32). The GPT-2 BPE tokenizer
breaks any unknown word into [[subword-tokenization|subword units or single
characters]]:
```python
tokenizer = tiktoken.get_encoding("gpt2")
tokenizer.encode("Akwirw ier")
# [33901, 86, 343, 86, 220, 959]  ->  "Ak", "w", "ir", "w", " ", "ier"
```
(Figure 2.11, Exercise 2.1.) And "someunknownPlace" encodes and decodes back
perfectly. So the model can read **any** text, even words it never saw in
training (book p. 34).

## 🔴 Theory (Expert)
- **Domain view.** With a fixed finite vocabulary $V$ of whole words, the
  word-to-ID map is only defined on $V$. OOV = trying to evaluate a function
  outside its domain. V1 leaves it partial (error); V2 extends it to a total
  function by sending the whole complement $\Sigma^* \setminus V$ to one ID.
- **Information loss.** That extension is many-to-one, so it has no inverse:
  every unknown word collapses to the same symbol and its identity is gone. The
  model also learns one shared embedding for "anything rare", which is a poor
  representation.
- **Open vocabulary.** Byte-level BPE fixes this by construction: the base
  alphabet contains every single byte, so every string is a concatenation of
  vocabulary items (the vocabulary *generates* all of $\Sigma^*$ under
  concatenation). OOV becomes impossible; rare words just cost more tokens.
- **Trade-off.** Whole-word vocabularies are short sequences but brittle;
  character vocabularies never fail but make sequences very long (bad for
  [[context-length]] and attention cost). Subword vocabularies sit between the
  two. (Analysis beyond the book.)
- **Zipf's law** (background, not in the book): word frequencies have a long
  tail, so any finite word list will keep meeting new words in new text. This
  is why "just use a bigger training set" (book p. 28) only shrinks, and never
  removes, the OOV rate for word-level tokenizers.

## Connections
- Builds on: [[tokenization]], [[vocabulary]], [[tokenizer-encode-decode]], [[token-id]]
- Leads to: [[special-tokens]], [[subword-tokenization]], [[byte-pair-encoding]]
- Used by: [[large-language-model]], [[gpt-2]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.3, book p. 28 (`KeyError: 'Hello'`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book pp. 29–32 (Figure 2.9, Listing 2.4, `<|unk|>` = 1131)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 33–34 (Figure 2.11, Exercise 2.1)
