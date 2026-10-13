---
title: End-of-Text Token (<|endoftext|>)
type: concept
tags: [nlp, llm, tokenization, text-generation, fine-tuning]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# End-of-Text Token (<|endoftext|>)

> **One-liner:** `<|endoftext|>` (ID 50256 in GPT-2) is a special token that means "this text is over" — it separates unrelated documents, tells the generator when to stop, and doubles as padding.

## 🟢 Intuition (Dummy)
Picture a long roll of paper with many different stories printed on it, one
after another. Without a clear mark between them, the end of a sports report
would run straight into the start of a fairy tale, and a reader would get
confused. `<|endoftext|>` is that mark: a big **"THE END"** stamp.

GPT models learn to read the stamp ("a new, unrelated text starts now") and
also to *write* it ("I'm done answering"). When the model writes it, we stop
asking for more words. Same token, also handy as blank filler at the end of
short texts.

General name: an **end-of-sequence (EOS)** token. `<|endoftext|>` is GPT's
version (book p. 32).

## 🟡 Mechanics (Practitioner)

### Job 1 — separating documents (book pp. 29–30)
When training on many independent documents, we glue them into one stream and
put `<|endoftext|>` **before each document that follows a previous one**
(Figure 2.10: a sports story, a fairy tale, a finance note, a story ending).
In sequence notation:
$$d_1 \,\Vert\, \langle\text{eot}\rangle \,\Vert\, d_2 \,\Vert\, \langle\text{eot}\rangle \,\Vert\, d_3 \cdots$$
- $d_1, d_2, d_3$ = the token sequences of three documents,
- $\Vert$ = "concatenate" (stick one sequence after another),
- $\langle\text{eot}\rangle$ = the single end-of-text token.

Book example (pp. 31–32), with the toy "Verdict" vocabulary:
```python
text = " <|endoftext|> ".join(("Hello, do you like tea?",
                               "In the sunlit terraces of the palace."))
SimpleTokenizerV2(vocab).encode(text)
# [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, ...]   1130 = <|endoftext|>
```

### Its ID in GPT-2: 50256 (book pp. 33–34)
```python
tokenizer = tiktoken.get_encoding("gpt2")
tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})   # [50256]
```
GPT-2's vocabulary has $|V| = 50{,}257$ tokens, IDs $0,\dots,50{,}256$, and
`<|endoftext|>` is the **last** one:
$$\mathrm{id}(\langle\text{eot}\rangle) = |V| - 1 = 50{,}257 - 1 = 50{,}256$$
- $\mathrm{id}(\cdot)$ = "the token ID of",
- $|V|$ = vocabulary size; IDs start at 0, so the largest ID is $|V|-1$.

### Job 2 — padding (book pp. 32, 176, 214)
GPT has no separate [PAD] token, so `<|endoftext|>` also fills short sequences
up to a common length. Both fine-tuning chapters use `pad_token_id=50256`.
Details: [[padding-and-truncation]].

### Job 3 — stopping generation (book pp. 157–158, 228)
The `generate` function (Listing 5.4) takes an `eos_id` argument and quits the
loop as soon as the model picks that token:
```python
if idx_next == eos_id:      # stop early at end-of-sequence
    break
```
In Chapter 7 the model is called with `eos_id=50256`: first the *pretrained*
baseline with `max_new_tokens=35` (§7.5, book p. 228), then the fine-tuned
model with `max_new_tokens=256` (§7.7, book p. 234). A response ends either
when the model picks `<|endoftext|>` or when the token budget runs out,
whichever comes first. As a rule, the number of appended tokens is
$$T_{\text{new}} = \min\bigl(\text{max\_new\_tokens},\; t^{*} - 1\bigr)$$
- $t^{*}$ = the loop step at which the model first picks ID 50256,
- the $-1$ is there because the `break` runs *before* `torch.cat`, so the EOS
  token itself is not appended (Listing 5.4, book p. 158).

### Job 4 — teaching the model *when* to stop (book pp. 217–219)
In instruction fine-tuning, the collate function appends one end-of-text token
to every example and builds targets shifted by one. It then replaces **all but
the first** padding token in each target with `-100`:
```
Target 2: [6, 50256, 50256, 50256, 50256]  ->  [6, 50256, -100, -100, -100]
Target 3: [8, 9, 50256, 50256, 50256]      ->  [8, 9, 50256, -100, -100]
```
That one kept 50256 is a real training target: the loss rewards the model for
predicting `<|endoftext|>` right after the response is finished. This is what
makes Job 3 work later (book p. 218). See [[ignore-index-loss-masking]].

## 🔴 Theory (Expert)
- **Modelling view.** A language model defines
  $P(x_{t+1}\mid x_{\le t})$ over $V$. Including an absorbing "end" symbol turns
  this into a distribution over *finite* sequences: the probability of a whole
  answer $y$ is $\prod_t P(y_t \mid \cdot)\cdot P(\langle\text{eot}\rangle\mid y)$.
  Without an end symbol, the model has no way to say "the answer is complete",
  and generation can only be cut off by a length limit. (Standard view; the book
  states the practical version on p. 218.)
- **Two roles, one embedding.** As a *separator* it is an input token that the
  model must learn to "reset" on; as a *target* it is a class the output layer
  must predict. Both share the same [[token-embedding]] row (and, with
  [[weight-tying]], the same output row).
- **Why padding with it is safe.** Positions filled with padding are excluded
  either by an attention mask (book p. 32) or by `ignore_index` in the
  [[cross-entropy-loss]]. With causal attention and padding on the *right*,
  real tokens never see the pads anyway — they only come later in the
  sequence (see [[causal-attention-mask]]).
- **Batch caveat** (note-taker's observation, not in the book): the check
  `idx_next == eos_id` in Listing 5.4 assumes batch size 1; for larger batches
  each row must track its own "finished" flag.

## Connections
- Builds on: [[special-tokens]], [[token-id]], [[vocabulary]], [[byte-pair-encoding]]
- Leads to: [[padding-and-truncation]], [[ignore-index-loss-masking]], [[autoregressive-text-generation]], [[decoding-strategies]]
- Used by: [[pretraining]], [[pretraining-datasets]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[collate-function]], [[response-extraction]], [[gpt-2]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book pp. 29–32 (Figures 2.9–2.10, [EOS])
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 33–34 (ID 50256, vocabulary 50,257)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.3, book pp. 157–158 (Listing 5.4, `eos_id`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3, book p. 176 (pad token 50256)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 214–222 (padding, keep-one-EOS rule, Figures 7.10–7.12)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5, book p. 228 (`eos_id=50256`, `max_new_tokens=35`, pretrained baseline)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.7, book p. 234 (`eos_id=50256`, `max_new_tokens=256`, fine-tuned model)
