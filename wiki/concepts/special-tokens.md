---
title: Special Context Tokens
type: concept
tags: [nlp, llm, tokenization, vocabulary]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Special Context Tokens

> **One-liner:** Special tokens are made-up "words" like `<|endoftext|>` or `<|unk|>` that we add to the vocabulary to tell the model something about the text — where a document ends, where a word is unknown, or where filler was added.

## 🟢 Intuition (Dummy)
Real text doesn't come with stage directions. When you glue a sports story, a
fairy tale and a finance report into one long stream, the model can't tell
where one stops and the next begins. So we sprinkle in little **signposts** —
tokens that are not real words, but that the model learns to read as
"new document starts here", "I don't know this word", or "this is just
padding, ignore me".

They live in the [[vocabulary]] like any other token and get an ID like any
other token. They're just *our* invention, not part of the language.

## 🟡 Mechanics (Practitioner)

### The two tokens the book adds (book pp. 29–30)
| Token | Meaning | ID in the "Verdict" vocab |
|---|---|---|
| `<\|unk\|>` | an unknown word not in the vocabulary — see [[out-of-vocabulary-words]] | 1131 |
| `<\|endoftext\|>` | a boundary between two unrelated texts — see [[end-of-text-token]] | 1130 |

They are appended **after** the sorted word list, so they get the last IDs:
```python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
len(vocab)   # 1132 = 1130 words + 2 special tokens
```
The vocabulary size simply grows by the number of special tokens:
$$|V_{\text{new}}| = |V_{\text{words}}| + |S| = 1130 + 2 = 1132$$
- $|V_{\text{words}}|$ = number of ordinary tokens,
- $S$ = the set of special tokens, $|S|$ = how many (here 2),
- $|V_{\text{new}}|$ = the final vocabulary size.

Figure 2.9 (book p. 29) draws the same idea on a toy vocabulary, with
`<|unk|>` = 783 and `<|endoftext|>` = 784 (the figure lists them in the
opposite order from the code). Figure 2.10 (book p. 30) shows four unrelated
texts joined with `<|endoftext|>` before each new one.

### Other special tokens you'll meet (book p. 32)
- **[BOS]** (beginning of sequence) — marks where a text starts.
- **[EOS]** (end of sequence) — marks where a text ends; useful when joining
  unrelated texts (e.g. two Wikipedia articles). Similar to `<|endoftext|>`.
- **[PAD]** (padding) — filler added to shorter texts so every text in a batch
  has the same length — see [[padding-and-truncation]].

### What GPT actually uses (book pp. 32–34)
GPT-style tokenizers keep it minimal: **only `<|endoftext|>`**.
- It plays the role of [EOS].
- It is **also** used as the padding token (the choice doesn't matter much,
  because a mask stops the model from attending to padded positions).
- There is **no `<|unk|>`**: [[byte-pair-encoding]] splits unknown words into
  subwords instead.

In the GPT-2 vocabulary, `<|endoftext|>` has ID **50256**, the largest ID
(vocabulary size 50,257).

### Telling tiktoken a special token is allowed (book pp. 33, 176)
A special token is typed as an ordinary string, so the tokenizer must be told
to treat it as one token rather than as the characters `<`, `|`, `end`, …:
```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
# [50256]
```
`allowed_special` lists which special strings may be turned into their special
IDs (book p. 176). Background (not stated in the book): without it, tiktoken
raises an error when it meets a special-token string, as a safety guard
against untrusted text injecting control tokens.

## 🔴 Theory (Expert)
- **Out-of-band symbols.** Formally a special token extends the alphabet the
  model reads: the input space becomes sequences over $V \cup S$ with
  $V \cap S = \emptyset$. Because no ordinary text maps to an ID in $S$ (when
  `allowed_special` is not set), these IDs work as a control channel that the
  user's text cannot fake.
- **Learned, not hard-coded, meaning.** The model has no built-in rule for
  `<|endoftext|>`. Its meaning comes only from where it appears in training
  data: its [[token-embedding]] is trained like any other row of the
  embedding [[matrix]]. Pretraining teaches "after this, the topic resets";
  instruction fine-tuning teaches "emit this when the answer is finished"
  (§7.3; see [[end-of-text-token]]).
- **Pad tokens vs. masks.** A pad token is only a placeholder for tensor shape.
  Whether it affects learning is decided by masks — an attention mask and/or
  `ignore_index=-100` in the loss ([[ignore-index-loss-masking]]). That is why
  GPT can reuse `<|endoftext|>` for padding.
- **Chat markers.** Later prompt formats add role markers such as
  `<|user|>`/`<|assistant|>` (Phi-3 style, Exercise 7.1 solution, Appendix C,
  book pp. 307–308). In the book's solution these are written as plain text in the
  [[prompt-template]] and tokenized normally, not registered as new special IDs.

## Connections
- Builds on: [[tokenization]], [[vocabulary]], [[tokenizer-encode-decode]], [[token-id]]
- Leads to: [[end-of-text-token]], [[out-of-vocabulary-words]], [[padding-and-truncation]], [[byte-pair-encoding]]
- Used by: [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[prompt-template]], [[gpt-2]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.3, book p. 29 (forward pointer to special tokens)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book pp. 29–32 (Figures 2.9–2.10, Listing 2.4, [BOS]/[EOS]/[PAD])
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 33–34 (`allowed_special`, ID 50256, vocabulary 50,257)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3, book p. 176 (encoding `<|endoftext|>` → `[50256]`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise 7.1 solution, book pp. 307–308 (Phi-3 markers)
