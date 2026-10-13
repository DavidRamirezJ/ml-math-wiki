---
title: Token ID
type: concept
tags: [llm, tokenization, data-preparation]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Token ID

> **One-liner:** A token ID is the integer "seat number" a tokenizer gives each piece of text, so the computer can handle words as numbers.

## 🟢 Intuition (Dummy)
Think of a huge dictionary where every entry has a line number. Instead of writing the word "fox",
you write "2" — the line where "fox" sits. A token ID is exactly that line number. Turning a sentence
into token IDs is like turning it into a list of page-and-line references.

The number itself means nothing: ID 3 isn't "bigger" or "more" than ID 2. It's just a label you can
use to look something up. In an LLM, what you look up is a **vector** for that token (its
[[token-embedding]]). Token IDs are the middle step: **text → tokens → IDs → vectors**.

## 🟡 Mechanics (Practitioner)

### From tokens to integers
A [[vocabulary]] is a mapping
$$\text{vocab}: V \to \{0,1,\dots,|V|-1\}$$
- $V$ = the set of unique tokens; $|V|$ = how many there are (the vocabulary size).
- $\{0,1,\dots,|V|-1\}$ = the allowed IDs (Python counts from 0).
- The ID of token $t$ is $\text{vocab}(t)$. (Formalization ours; the procedure is from book p. 25.)

Encoding a sentence = applying vocab to each token:
$$\text{encode}(t_1,\dots,t_n) = \big(\text{vocab}(t_1),\dots,\text{vocab}(t_n)\big)$$

### The book's examples
| Tokenizer | Example | Token IDs | Page |
|---|---|---|---|
| Toy vocabulary (Fig. 2.6) | brown, dog, fox, jumps, lazy, over, quick, the | 0, 1, 2, 3, 4, 5, 6, 7 | p. 25 |
| `SimpleTokenizerV1`, vocab 1,130 | `"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.` | `[1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]` | p. 27 |
| `SimpleTokenizerV2`, vocab 1,132 | `Hello, do you like tea? <\|endoftext\|> In the sunlit terraces of the palace.` | `[1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]` | p. 32 |
| GPT-2 BPE (`tiktoken`), vocab 50,257 | `Hello, do you like tea? <\|endoftext\|> In the sunlit terraces of someunknownPlace.` | `[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]` | p. 33 |
| GPT-2 BPE (Fig. 2.11) | `Akwirw ier` → "Ak", "w", "ir", "w", " ", "ier" | `[33901, 86, 343, 86, 220, 959]` | p. 34 |
| GPT-2 BPE (ch. 4/5) | "Every effort moves you" | `[6109, 3626, 6100, 345]` | pp. 98, 120, 131 |

Notice: in V1, ID 5 shows up for both commas and 7 for both periods — the same token always gets the
same ID. In V2, 1131 = `<|unk|>` (unknown words "Hello" and "palace") and 1130 = `<|endoftext|>`
(book p. 32). In GPT-2, `<|endoftext|>` = **50256**, the largest ID, because the vocabulary has
50,257 entries (book p. 34). See [[special-tokens]], [[end-of-text-token]].

### Going back: decoding
The inverse map $\text{vocab}^{-1}$ turns IDs back into text (`int_to_str` in the book's tokenizer
classes, book p. 27). For the BPE tokenizer the round trip
$\text{decode}(\text{encode}(s)) = s$ is lossless (book p. 33); with `<|unk|>` it isn't
(book p. 32). See [[tokenizer-encode-decode]].

### Token IDs as tensors (PyTorch)
```python
def text_to_token_ids(text, tokenizer):            # Listing 5.1, book p. 132
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)       # add batch dimension -> shape [1, n]

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())
```
Batches of IDs are integer tensors of shape `(batch, seq_len)`, e.g. the `[8, 4]` batch on
book p. 46 from the [[sliding-window-sampling]] loader.

### Token IDs everywhere else in the book
- **Input to the embedding layer:** an ID is a *row index* into the embedding matrix
  (§2.7, book pp. 42–43).
- **Output of the model:** the logits have one entry per possible ID (shape `[batch, tokens, 50257]`);
  [[argmax]] over the last dimension picks the predicted ID, which is decoded to text
  (§5.1, book pp. 131–134; Fig. 5.4 uses a toy 7-word vocabulary with IDs 0–6).
- **Padding:** in ch. 6, short SMS messages are padded with ID 50256 to the longest training
  message (120 tokens). Fig. 6.6: "This is the first text message" → `1212, 318, 262, 717, 2420, 3275`
  plus five 50256s (book pp. 176–178). See [[padding-and-truncation]].
- **Class labels as "token IDs":** the book compares turning "spam"/"ham" into 1/0 to token IDs, but
  with "only two token IDs: 0 and 1" instead of the 50,000+ word GPT vocabulary (§6.2, book p. 174).
  See [[label-encoding]].
- **Inference on new text (§6.8, Listing 6.12):** `tokenizer.encode(text)` → truncate to the supported context length →
  pad with 50256 (book p. 201).

> Note on the book's numbers: the ch. 6 summary says the old output layer had "50,256" nodes
> (book p. 202). That is the *largest token ID*; the vocabulary size — and the number of output
> nodes — is **50,257** (book p. 34).

## 🔴 Theory (Expert)
- **Nominal data.** Token IDs are labels on a nominal scale: any permutation of the IDs gives an
  equivalent model (permute the embedding rows and output-head rows the same way). Ordering — the
  alphabetical sort in §2.3, or BPE merge order in GPT-2 — is a convention. That's why IDs are never
  fed to a network as numbers; they index into an embedding matrix.
- **Bijection, and where it breaks.** On a fixed vocabulary, vocab is a bijection
  $V\leftrightarrow\{0,\dots,|V|-1\}$, so the inverse exists. Mapping *arbitrary strings* to IDs is
  only total if you add `<|unk|>` (many-to-one, lossy) or use a subword scheme like
  [[byte-pair-encoding]] whose byte-level base alphabet makes every string representable.
- **One-hot view.** ID $i$ is equivalent to the standard basis vector $\mathbf{x}_i\in\{0,1\}^{|V|}$;
  the embedding lookup is $\mathbf{x}_i^\top W$ (book p. 43). See [[one-hot-encoding]].
- **Compression ratio.** "The Verdict" has 20,479 characters, 4,690 regex tokens (book p. 24) and
  5,145 BPE tokens (book p. 35). The counts aren't directly comparable (the regex tokenizer drops
  whitespace; BPE folds spaces into tokens), but splitting rare words into pieces is one reason BPE
  needs more IDs — the price of never having an unknown word (our interpretation).

## Connections
- Builds on: [[token]], [[tokenization]], [[vocabulary]]
- Leads to: [[token-embedding]], [[embedding-layer]], [[sliding-window-sampling]], [[logits]]
- Used by: [[tokenizer-encode-decode]], [[byte-pair-encoding]], [[special-tokens]], [[padding-and-truncation]], [[greedy-decoding]]
- Related: [[one-hot-encoding]], [[label-encoding]], [[out-of-vocabulary-words]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.2 (Fig. 2.4 pipeline, IDs 40134 2052 133 389 12), book p. 21; §2.3 (vocabulary → IDs, Listings 2.2–2.3, Figs. 2.6–2.8), book pp. 24–28; §2.4 (`SimpleTokenizerV2`, vocab 1,132, IDs 1130/1131), book pp. 29–32; §2.5 (BPE IDs, 50256, Fig. 2.11, Exercise 2.1), book pp. 33–35.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1 (Listing 5.1, Figs. 5.3–5.4, logits → IDs), book pp. 130–134.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.2 (labels as two "token IDs"), book p. 174; §6.3 (Fig. 6.6, padding ID 50256), book pp. 175–178; §6.8 (inference pipeline; "50,256" summary), book pp. 200–203.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` App. C (Exercise 2.1 solution), book p. 300.

## From llm-ch07-fine-tuning-to-follow-instructions
- Step 2.2 of §7.3: `InstructionDataset` formats each entry with the [[prompt-template]] and then turns the whole text into token IDs with `tokenizer.encode(full_text)` (book p. 213). The pad ID is 50256 (`<|endoftext|>`), the same as in Ch. 6 (book p. 214).
- Ch. 7 adds a new *target* value. The target ID **−100** is not a real token. It is a signal that tells `cross_entropy` to skip that position (book pp. 218–222). See [[ignore-index-loss-masking]], [[collate-function]], [[llm-ch07-fine-tuning-to-follow-instructions]].
- Source: `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 211–223.
