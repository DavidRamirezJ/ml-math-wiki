---
title: Vocabulary (Tokenizer)
type: concept
tags: [llm, tokenization, data-preparation]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Vocabulary (Tokenizer)

> **One-liner:** A vocabulary is the tokenizer's fixed list of every token it knows, each paired with its own integer ID — the dictionary that turns text into numbers and back.

## 🟢 Intuition (Dummy)
Picture a phone book for words. Every word the tokenizer knows gets a line with its own number.
"brown" is line 0, "dog" is line 1, "fox" is line 2, and so on. To convert a sentence into
numbers, you look each word up; to convert numbers back into a sentence, you read the phone book
backwards.

Two problems show up right away:
1. **Words that aren't in the book.** If "Hello" never appeared in the text you built the book
   from, you're stuck. Simple tokenizers either crash or write "unknown"; GPT's tokenizer cleverly
   spells the word out from smaller pieces it *does* know.
2. **Size.** A bigger book covers more words but makes the model bigger, because the model keeps
   one learned vector — and one output score — per entry. GPT-2's book has 50,257 entries.

## 🟡 Mechanics (Practitioner)

### Building one (§2.3, Fig. 2.6, book p. 25)
1. Tokenize the whole training text.
2. Remove duplicates (keep each unique token once).
3. Sort alphabetically.
4. Give each token its position in the sorted list — its [[token-id]].

As a formula (our notation):
$$\text{vocab}: V \to \{0,1,\dots,|V|-1\},\qquad t\mapsto \operatorname{rank}\big(t \text{ in } \operatorname{sorted}(V)\big)$$
- $V$ = the set of unique tokens; $|V|$ = its size (the vertical bars mean "number of elements").
- $t$ = one token; $\operatorname{rank}$ = its 0-based position after sorting.

Fig. 2.6's toy example, "The quick brown fox jumps over the lazy dog":
`brown→0, dog→1, fox→2, jumps→3, lazy→4, over→5, quick→6, the→7` (book p. 25).

**Code (Listing 2.2, book p. 25):**
```python
all_words = sorted(set(preprocessed))      # unique tokens, sorted
vocab_size = len(all_words)                # 1130 for "The Verdict"
vocab = {token: integer for integer, token in enumerate(all_words)}
```
Fig. 2.7 (book p. 26): the vocabulary built from the training set is then applied to *new* text —
"The brown dog playfully chased the swift fox" → `7, 0, 1, …`.

The first entries printed are `('!', 0)`, `('"', 1)`, `("'", 2)`, …, `('Her', 49)`,
`('Hermia', 50)` (book p. 26) — punctuation and capitalized words come first because Python sorts by
character code.

### The inverse vocabulary
To turn model outputs back into text you need the reverse map (book p. 26):
$$\text{vocab}^{-1}: \{0,\dots,|V|-1\}\to V,\qquad \text{vocab}^{-1}(\text{vocab}(t)) = t$$
In the book's tokenizer classes: `self.str_to_int = vocab` and
`self.int_to_str = {i: s for s, i in vocab.items()}` (Listing 2.3, book p. 27). These power
`encode` and `decode` — see [[tokenizer-encode-decode]].

### Out-of-vocabulary words and special tokens
- `SimpleTokenizerV1` on "Hello, do you like tea?" → `KeyError: 'Hello'`: "Hello" isn't in "The
  Verdict". Lesson: "the need to consider large and diverse training sets to extend the vocabulary"
  (book p. 28). See [[out-of-vocabulary-words]].
- §2.4 extends the vocabulary with two [[special-tokens]] (book p. 30):
  ```python
  all_tokens = sorted(list(set(preprocessed)))
  all_tokens.extend(["<|endoftext|>", "<|unk|>"])
  vocab = {token: integer for integer, token in enumerate(all_tokens)}
  ```
  Size goes from **1,130 → 1,132**; the last entries are `('younger', 1127)`, `('your', 1128)`,
  `('yourself', 1129)`, `('<|endoftext|>', 1130)`, `('<|unk|>', 1131)`.
  (Fig. 2.9 shows them as 783/784 in its toy vocabulary, in the opposite order — book p. 29.)

### GPT-2's BPE vocabulary (§2.5)
- `tiktoken.get_encoding("gpt2")` loads a vocabulary of **50,257** tokens, IDs 0…50,256;
  `<|endoftext|>` has the largest ID, **50256** (book p. 34):
  $$|V| = 50{,}257,\qquad \mathrm{id}(\texttt{<|endoftext|>}) = |V|-1 = 50{,}256$$
- It has **no** `<|unk|>`: unknown words are broken into "smaller subword units or even individual
  characters" (book p. 34). Fig. 2.11: "Akwirw ier" → "Ak"(33901) "w"(86) "ir"(343) "w"(86) " "(220)
  "ier"(959).
- How BPE builds it: start with all single characters, then repeatedly merge frequent pairs
  (e.g. "d"+"e" → "de", common in "define", "depend", "made", "hidden"), with a frequency cutoff
  (book pp. 34–35). See [[byte-pair-encoding]], [[subword-tokenization]].

### Vocabulary size shapes the model
`vocab_size` appears twice in a GPT model (`GPT_CONFIG_124M`, book p. 95):
```python
self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])            # 50257 x 768
self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)   # 768 -> 50257
```
- **Input side:** one embedding row per vocabulary entry (§2.7: toy `vocab_size = 6`, real 50,257;
  book p. 42). See [[token-embedding]].
- **Output side:** the model returns one score per vocabulary entry — output shape
  `[2, 4, 50257]` for 2 texts × 4 tokens; "the embedding has 50,257 dimensions because each of these
  dimensions refers to a unique token in the vocabulary" (book p. 99). Softmax turns these [[logits]] into a [[probability-distribution]]
  over the vocabulary (§5.1, book p. 134).
- Those two layers are "very large due to the number of rows for the 50,257 in the tokenizer's
  vocabulary" — $50{,}257\times768 = 38{,}597{,}376$
  parameters each (book p. 121; arithmetic ours), which is why GPT-2 ties them ([[weight-tying]]).
- Fig. 5.4 (book p. 133) walks generation through a toy 7-word vocabulary
  `{"a":0, "effort":1, "every":2, "forward":3, "moves":4, "you":5, "zoo":6}` and its inverse.

## 🔴 Theory (Expert)
- **Bijection on a closed set; partial function on the open world.** On $V$, vocab is a bijection, so
  decode∘encode is the identity. On arbitrary strings it's only a *partial* function (the `KeyError`
  is "evaluated outside its domain"). Two fixes: make it total but lossy with `<|unk|>`
  (many-to-one; "Hello" and "palace" both become 1131, book p. 32), or choose a vocabulary that
  *generates* all strings under concatenation — BPE's byte-level base alphabet — so
  $\mathrm{decode}(\mathrm{encode}(s)) = s$ for every $s$ (book p. 33).
- **Case sensitivity is a modelling choice.** The book keeps capitalization because it helps
  distinguish proper nouns, sentence structure and correct capitalized output (§2.2, book p. 23);
  this enlarges $V$. (Fig. 2.6 quietly merges "The"/"the", which the case-sensitive code would not.)
- **Size trade-off.** Larger $|V|$ → shorter sequences (more text per [[context-length]] window) but
  more embedding/output parameters and a costlier softmax over $|V|$ classes; smaller $|V|$ → longer
  sequences. An untrained model's loss is about $\log|V| = \log 50{,}257 \approx 10.8$ nats (perplexity
  ≈ $|V|$), the "uniform guess" baseline — the book's untrained model has perplexity 48,725 (§5.1,
  book p. 140). See [[perplexity]], [[cross-entropy-loss]].
- **Order is arbitrary.** Any permutation of IDs gives an equivalent model; alphabetical order (§2.3)
  and BPE merge order are conventions.

## Connections
- Builds on: [[token]], [[tokenization]]
- Leads to: [[token-id]], [[tokenizer-encode-decode]], [[token-embedding]], [[logits]]
- Used by: [[byte-pair-encoding]], [[special-tokens]], [[end-of-text-token]], [[out-of-vocabulary-words]], [[gpt-model-configuration]], [[weight-tying]]
- Related: [[subword-tokenization]], [[softmax]], [[probability-distribution]], [[parameter-count-and-memory]]
- People: [[sebastian-raschka]], [[rico-sennrich]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.2 (case preservation), book p. 23; §2.3 (Fig. 2.6, Listing 2.2, inverse vocabulary, Listing 2.3, `KeyError`), book pp. 24–28; §2.4 (extended vocabulary 1,132, Fig. 2.9), book pp. 29–32; §2.5 (GPT-2 BPE vocabulary 50,257, Fig. 2.11, BPE merges), book pp. 33–35; §2.7 (`vocab_size` as rows of the embedding matrix), book p. 42.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1 (`GPT_CONFIG_124M`, output shape `[2, 4, 50257]`), book pp. 95–99; §4.6 (size of embedding/output layers), book p. 121.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1 (Fig. 5.4 toy vocabulary, softmax over vocabulary, perplexity 48,725), book pp. 133–140.
