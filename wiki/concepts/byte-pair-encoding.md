---
title: Byte Pair Encoding (BPE)
type: concept
tags: [nlp, llm, tokenization, algorithms, compression]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Byte Pair Encoding (BPE)

> **One-liner:** BPE builds a vocabulary by repeatedly gluing together the pair of symbols that appears most often, so common words become one token and rare words are spelled out from smaller pieces.

## 🟢 Intuition (Dummy)
Imagine you text a lot and get lazy: you notice you type "t-h-e" all the time,
so you invent one shortcut key for "the". Then you notice "ing" everywhere and
make a key for that too. Keep going and you end up with a keyboard where
common words are one key, and weird rare words are typed from a few smaller
keys.

That's BPE. It starts with single characters, then keeps merging the pairs
that show up most often — "d" + "e" → "de" (common in *define*, *depend*,
*made*, *hidden*; book pp. 34–35) — then merges those into bigger pieces, and
so on.

The big win: **it never meets a word it can't handle.** A made-up word like
"Akwirw ier" just gets broken into known pieces: "Ak", "w", "ir", "w", " ",
"ier" (Fig. 2.11, book p. 34). So there's no need for an "unknown word" token.

## 🟡 Mechanics (Practitioner)

### Using it: the `tiktoken` library
BPE was used to train **GPT-2, GPT-3 and the original ChatGPT model** (book
p. 33). Writing BPE yourself is "relatively complicated", so the book uses
OpenAI's open-source **tiktoken** (core in Rust, very fast; the book uses
version 0.7.0; book p. 33):

```python
# pip install tiktoken
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250,
#  8812, 2114, 286, 617, 34680, 27271, 13]
print(tokenizer.decode(integers))
# Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.
```

- `allowed_special={"<|endoftext|>"}` tells tiktoken to treat that string as the
  single special token, not as ordinary text (our explanation; the book uses
  the argument without explaining it, and reuses it in §6.3, book p. 176, to
  confirm that the padding token ID is 50256).
- Same API as the hand-made tokenizers: `encode` (text → IDs) and `decode`
  (IDs → text) — see [[tokenizer-encode-decode]].

### Key facts (book p. 34)
- Vocabulary size **50,257**; IDs run $0,\dots,50{,}256$.
- `<|endoftext|>` has the **largest** ID, **50256**:

$$|V| = 50{,}257,\qquad \text{IDs}\in\{0,\dots,50{,}256\},\qquad \text{id}(\texttt{<|endoftext|>}) = |V| - 1$$

  Read: "$|V|$ (the number of tokens in the [[vocabulary]] $V$) is 50,257; every
  ID is a whole number from 0 to 50,256; the end-of-text token's ID is the
  last one, $|V|-1$."
- Unknown words like *someunknownPlace* are split into subword units or single
  characters, so BPE can process **any** text and needs no `<|unk|>`.
- 220 is the token for a lone space (Fig. 2.11).

### Exercise 2.1 (book p. 34; solution book p. 300)
Encode `"Akwirw ier"`, decode each ID on its own, then decode the whole list:

| Token | "Ak" | "w" | "ir" | "w" | " " | "ier" |
|---|---|---|---|---|---|---|
| ID | 33901 | 86 | 343 | 86 | 220 | 959 |

`tokenizer.decode([33901, 86, 343, 86, 220, 959])` → `'Akwirw ier'` — a
perfect round trip. Notice "w" maps to 86 both times.

### How the vocabulary is built (the algorithm)
The book only sketches it (detailed implementation is out of scope, book
p. 34): start with all single characters, then **iteratively merge frequent
characters into subwords and frequent subwords into words**, stopping at a
**frequency cutoff** (book pp. 34–35). In symbols (our rendering, not printed in
the book):

$$\text{count}(a,b) = \#\{\, i : x_i = a,\ x_{i+1} = b \,\}$$

$$(a^*, b^*) = \arg\max_{(a,b)} \text{count}(a,b), \qquad V_{t+1} = V_t \cup \{a^*b^*\}$$

Symbol by symbol:
- $x_1, x_2, \dots$ = the training text, written as a sequence of current tokens; $x_i$ = the token at position $i$.
- $a, b$ = two tokens already in the vocabulary; $(a,b)$ = "$a$ immediately followed by $b$".
- $\#\{\dots\}$ = "how many positions $i$ satisfy the condition" — a pair [[histogram]].
- $\arg\max$ = "the pair that gives the biggest count" (not the count itself) — see [[argmax]].
- $V_t$ = the vocabulary after $t$ merges; $\cup\{a^*b^*\}$ = "add the glued-together token".
- Then every occurrence of $a^* b^*$ in the text is replaced by the new token, and we repeat until the counts drop below the cutoff or the target size (50,257 for GPT-2) is reached.

**Toy run** (our example): text `aaabdaaabac`.
1. Pair counts: `aa`=4, `ab`=2, others 1 → merge `aa`→`Z`: `ZabdZabac`.
2. Now `Za`=2 and `ab`=2 tie; pick `ab`→`Y`: `ZYdZYac`.
3. `ZY`=2 → merge to `X`: `XdXac`.

Eleven symbols became five, and the vocabulary grew by `Z`, `Y`, `X`.
**Encoding** new text replays the learned merges in the same order.

### Where BPE shows up later in the book
- §2.6: *The Verdict* becomes **5,145** BPE tokens (vs 4,690 regex tokens
  in §2.2), the raw material for the
  [[sliding-window-sampling]] data loader (book p. 35).
- §2.8 / §4.1: `vocab_size = 50257` sets the rows of the token
  [[embedding-layer]] ($50{,}257\times 768$ in GPT-2 small) and the width of the
  output [[logits]] (book pp. 46, 95, 99).
- §5.5: OpenAI's GPT-2 download contains `encoder.json` (1.04M, token → ID map)
  and `vocab.bpe` (456k, the merge rules) — the tokenizer travels with the
  weights (book pp. 161–162).
- Appendix E.2: the LoRA experiment reuses `tiktoken.get_encoding("gpt2")` for
  the spam dataset (book p. 325).

## 🔴 Theory (Expert)
- **Why it can encode anything.** GPT-2's BPE works on **bytes**, and all 256
  byte values are in the base vocabulary (background, not from the book). Any
  string is a byte sequence, and the merges only ever combine existing
  symbols, so encoding is a **total** function and decoding is its exact
  inverse: $\text{dec}(\text{enc}(s)) = s$. This is what lets the book say BPE
  "can handle any unknown word" (book p. 34).
- **A greedy compressor.** The name comes from a 1994 data-compression
  algorithm (Gage) that replaced the most frequent byte pair with an unused
  byte (background). Each merge is a greedy step that shortens the corpus by
  $\text{count}(a^*,b^*)$ symbols (minus overlaps, e.g. `aaa`). It's not
  globally optimal; it's a simple, fast heuristic.
- **Vocabulary-size trade-off.** Every merge adds one token: longer tokens
  make sequences shorter (more text fits in the [[context-length]] and attention
  is cheaper), but the [[embedding-layer]] and output layer grow linearly in
  $|V|$. GPT-2 settles at 50,257 = 256 bytes + 50,000 merges + 1 special token
  (background arithmetic consistent with the book's 50,257).
- **Frequency ≈ probability.** Merging by count means frequent strings get
  short codes — the same instinct as entropy coding in information theory.
  Rare words are spelled out from pieces, so the model still sees their
  sub-parts (prefixes, suffixes) and can generalise.
- **Not the only subword scheme.** The book points to SentencePiece (Kudo &
  Richardson 2018) and WordPiece (used by [[bert]]; Song et al. 2020) as
  alternatives "used by some other popular LLMs" (book p. 291). (That
  WordPiece is BERT's tokenizer is background, not stated in the book.) See
  [[subword-tokenization]].
- **Further reading the book recommends** (book pp. 290–291):
  Sennrich et al. (2015), *Neural Machine Translation of Rare Words with
  Subword Units* (arXiv 1508.07909) — BPE as a tokenization method;
  OpenAI's GPT-2 `encoder.py`; the interactive OpenAI tokenizer page; and
  Andrej Karpathy's **minbpe**, a minimal readable BPE implementation for
  training your own.

## Connections
- Builds on: [[tokenization]], [[vocabulary]], [[histogram]], [[argmax]]
- Leads to: [[token-id]], [[token-embedding]], [[embedding-layer]], [[sliding-window-sampling]], [[gpt-model-configuration]]
- Related: [[subword-tokenization]], [[out-of-vocabulary-words]], [[special-tokens]], [[end-of-text-token]], [[tokenizer-encode-decode]], [[context-length]], [[word2vec]]
- Used by: [[gpt-2]], [[gpt-3]], [[large-language-model]], [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]]
- People: [[rico-sennrich]], [[andrej-karpathy]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book p. 18 (BPE announced as GPT's tokenizer).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book p. 32 (GPT uses BPE instead of `<|unk|>`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 33–35 (tiktoken, 50,257, Fig. 2.11, Exercise 2.1, merge sketch).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6, book p. 35 (5,145 tokens); §2.8, book p. 46 (`vocab_size = 50257`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 95–99 (`vocab_size` in `GPT_CONFIG_124M`, logits of width 50,257).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 161–162 (`encoder.json`, `vocab.bpe`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3, book p. 176 (`allowed_special`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.2, book p. 325 (tokenizer in the LoRA experiment).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 290–291 (further reading); appendix C, book p. 300 (Exercise 2.1 solution).
