---
title: Padding, Truncation and Pad Masking
type: concept
tags: [nlp, llm, preprocessing, batching, fine-tuning]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Padding, Truncation and Pad Masking

> **One-liner:** To stack texts of different lengths into one neat table (a batch), we either cut long ones short (truncation) or fill short ones with a filler token (padding) — and then make sure the model and the loss ignore the filler (masking).

## 🟢 Intuition (Dummy)
A GPU loves rectangles. A batch has to be a grid with the same number of
columns in every row. But text messages come in all sizes: "K. I will sent it
again" is short, a spam advert is long.

Two fixes:
- **Truncate:** chop everything to the same short length. Fast, but you may
  cut off the important part.
- **Pad:** keep everything and add "blank" tokens to the short ones until they
  match the longest. Nothing is lost; you just carry some empty boxes.

The book chooses **padding** (book p. 176). Then comes the important part: you
tell the model "those blanks mean nothing — don't learn from them" (**pad
masking**).

## 🟡 Mechanics (Practitioner)

### The idea in Chapter 2 (book p. 32)
With batch sizes larger than one, shorter texts are padded with a [PAD] token
up to the length of the longest text in the batch. GPT has no [PAD]; it pads
with `<|endoftext|>` (ID 50256, see [[end-of-text-token]]), and a **mask**
keeps padded tokens from being attended to, so the exact pad token doesn't
matter.

### The two formulas
For a token sequence $x = (x_1,\dots,x_\ell)$ and a target length $L$:

**Padding** (when $\ell \le L$):
$$\tilde{x} = (x_1,\dots,x_\ell,\underbrace{p,\dots,p}_{L-\ell \text{ times}})$$

**Truncation** (when $\ell > L$):
$$\tilde{x} = (x_1,\dots,x_L)$$

Symbol by symbol:
- $x_i$ = the $i$-th [[token-id]] of the text,
- $\ell$ ("ell") = the text's real length,
- $L$ = the length every row must have (`max_length`),
- $p$ = the pad token ID ($50256$),
- the underbrace = "this block has $L-\ell$ copies of $p$",
- $\tilde{x}$ ("x tilde") = the fixed-length result.

In code: `encoded_text + [pad_token_id] * (max_length - len(encoded_text))` and
`encoded_text[:max_length]` (book p. 177).

### Classification: pad to the longest *training* text (Ch. 6)
Truncating to the shortest message is cheaper but can lose a lot of
information, so the book pads (book pp. 175–176). Figure 6.6 (book p. 176):

| Message | Token IDs | Pads added |
|---|---|---|
| "This is the first text message" | 1212, 318, 262, 717, 2420, 3275 | 5 × 50256 |
| "This is another text message" | 1212, 318, 1194, 2420, 3275 | 6 × 50256 |
| "This is the third text message, which is very long" | 11 IDs | 0 (longest) |

The `SpamDataset` class (Listing 6.4, book p. 177) does both jobs:
```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length]        # truncate
                                  for encoded_text in self.encoded_texts]
        self.encoded_texts = [encoded_text + [pad_token_id] *            # pad
                              (self.max_length - len(encoded_text))
                              for encoded_text in self.encoded_texts]
```
- Training set: $L = \max_i \ell_i = 120$ tokens (book p. 178).
- Validation and test sets reuse $L = 120$; longer samples are truncated
  (optional, as long as nothing exceeds the 1,024-token [[context-length]]).
- One batch is then an $8 \times 120$ integer [[matrix]] plus 8 labels
  (Figure 6.7, book pp. 179–180).
- The same `SpamDataset` setup is reused for LoRA fine-tuning (Appendix E.2,
  book p. 325).

**At inference** (`classify_review`, Listing 6.12, book p. 201) a new text is
treated the same way:
```python
supported_context_length = model.pos_emb.weight.shape[0]           # 1024
input_ids = input_ids[:min(max_length, supported_context_length)]   # truncate
input_ids += [pad_token_id] * (max_length - len(input_ids))         # pad
```
Here the number of pads is $n_{\text{pad}} = \text{max\_length} - \text{len(input\_ids)}$,
and `max_length=train_dataset.max_length` (120), so test inputs look like
training inputs (book p. 202).

**Too much padding hurts.** Exercise 6.1 (book p. 179) pads everything to the
full 1,024 tokens. Result: test accuracy drops from **95.67% to 78.33%**
(Appendix C, book p. 306).

### Instruction fine-tuning: dynamic padding + masking (Ch. 7)
Here a custom [[collate-function]] pads each batch only to *its own* longest
example ([[dynamic-padding]]), so different batches have different lengths
(Figure 7.8, book p. 214):
```
Batch 1: [0,1,2,3,4]   [5,6,50256,50256,50256]   [7,8,9,50256,50256]   (length 5)
Batch 2: [8,1,50256,50256]   [10,3,11,6]   [5,22,13,13]                (length 4)
```
Then, in the shifted targets, every pad except the first becomes `-100`, so the
[[cross-entropy-loss]] skips them ([[ignore-index-loss-masking]], book
pp. 218–222):
$$\mathcal{L} = -\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}} \log p_\theta(y_i \mid x_{\le i}),\qquad \mathcal{I}=\{i : y_i \neq -100\}$$
- $\mathcal{L}$ = the loss, $\theta$ = model weights,
- $y_i$ = target token at position $i$; $p_\theta(y_i\mid x_{\le i})$ = the
  probability the model gives it,
- $\mathcal{I}$ = the set of positions that are **not** masked,
  $|\mathcal{I}|$ = how many there are,
- so the average is taken over real tokens only.

An optional `allowed_max_length` truncates examples longer than GPT-2's 1,024
tokens (book p. 218). Batches come out as e.g. $8 \times 61$, $8 \times 76$, …
(§7.4, book pp. 224–226).

## 🔴 Theory (Expert)
- **Shape, not meaning.** Padding embeds variable-length sequences into a fixed
  space $\{0,\dots,|V|-1\}^L$ so a batch is a $B \times L$ [[tensor]]. It adds
  no information; the whole point of masking is to make the model's output for
  real tokens (and the loss) **invariant** to what sits in the pad slots.
- **Two kinds of mask.** (1) An *attention mask* sets scores to $-\infty$ at pad
  keys before the [[softmax]], giving them zero weight (book p. 32 mentions
  this; the book's GPT code relies on the [[causal-attention-mask]] instead).
  (2) A *loss mask* (`ignore_index`) drops pad positions from the average.
- **Right padding + causal attention.** The book pads on the right. Under
  causal attention a real token at position $i$ only attends to positions
  $\le i$, so it never sees later pads — the real tokens' hidden states are
  exactly what they would be without padding. (Analysis beyond the book.)
- **A subtle catch in classification.** The classifier reads the **last**
  position ([[last-token-pooling]]). With right padding, for short messages the
  last position *is* a pad token. It still works, because that pad attends to
  all real tokens before it, but the model must learn to summarise through
  many pads. This is one plausible reason why padding to 1,024 tokens
  (Exercise 6.1) drops accuracy so much. (Interpretation, not stated in the
  book.)
- **Cost.** Self-attention costs $O(L^2)$ per sequence, so wasted pad slots are
  expensive. Dynamic per-batch padding cuts waste; sorting/bucketing examples
  by length cuts it further (background).
- **Truncation is information loss.** Truncation is a projection that throws
  away $x_{L+1},\dots,x_\ell$; it is unavoidable only when $\ell$ exceeds the
  model's context length (1,024 for GPT-2).

## Connections
- Builds on: [[tokenization]], [[special-tokens]], [[end-of-text-token]], [[mini-batch]], [[tensor]], [[context-length]]
- Leads to: [[dynamic-padding]], [[collate-function]], [[ignore-index-loss-masking]], [[instruction-masking]]
- Used by: [[pytorch-dataset-and-dataloader]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]], [[last-token-pooling]]
- Related: [[causal-attention-mask]], [[cross-entropy-loss]], [[softmax]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book p. 32 ([PAD], masking, `<|endoftext|>` as pad)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3, book pp. 175–181 (Figure 6.6, Listing 6.4, Figure 6.7, Exercise 6.1)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.8, book pp. 200–202 (Listing 6.12 `classify_review`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 214–222 (Figure 7.8, `custom_collate_fn`, `allowed_max_length`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise 6.1 solution, book p. 306 (78.33% vs 95.67%)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix E.2, book p. 325 (SpamDataset reused for LoRA)
