---
title: Dynamic (Per-Batch) Padding
type: concept
tags: [pytorch, batching, padding, efficiency, instruction-fine-tuning]
difficulty: beginner
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Dynamic (Per-Batch) Padding

> **One-liner:** Dynamic padding stretches each batch only to the length of *its own* longest example, instead of padding everything to the longest example in the whole dataset.

## 🟢 Intuition (Dummy)
Say you're mailing letters and every envelope in a box must be the same size. The lazy way: buy
one giant envelope size that fits the single longest letter you'll *ever* send, and use it for
everything. Wasteful — most envelopes are mostly air.

The smart way: sort into boxes, and for each box pick the envelope size that fits the longest
letter **in that box**. Box 1 might use medium envelopes, box 2 small ones. Less air, less postage.

That's dynamic padding. Each batch gets its own width. The filler ("air") is the token `50256`,
and the model has to process every filler token — so less filler means faster training.

## 🟡 Mechanics (Practitioner)

### The rule
For a batch $b$ containing sequences with lengths $\ell_1, \dots, \ell_B$:

$$T_b = \max_{i \in b} \ell_i$$

- $b$ = one particular batch,
- $\ell_i$ = the number of tokens in example $i$,
- $\max_{i \in b}$ = "the biggest one among the examples in this batch",
- $T_b$ = the padded width of batch $b$ — every example in $b$ is padded with `50256` up to $T_b$.

Compare with **static** padding to the global maximum $T^\star = \max_{\text{all } i} \ell_i$:
dynamic padding always satisfies $T_b \le T^\star$.

### The book's picture (Figure 7.8, book p. 214)
| Batch | Raw inputs | Padded | Width |
|---|---|---|---|
| 1 | `[0,1,2,3,4]`, `[5,6]`, `[7,8,9]` | `[0,1,2,3,4]`, `[5,6,50256,50256,50256]`, `[7,8,9,50256,50256]` | 5 |
| 2 | `[8,1]`, `[10,3,11,6]`, `[5,22,13,13]` | `[8,1,50256,50256]`, `[10,3,11,6]`, `[5,22,13,13]` | 4 |

Token ID **50256** is GPT-2's `<|endoftext|>`, reused as the padding token (as in chapter 6). The
book appends the **ID** directly rather than the text, after checking it with
`tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})` → `[50256]` (book p. 214).

### Where it happens in code
Inside the [[collate-function]] (book p. 215):
```python
batch_max_length = max(len(item)+1 for item in batch)   # longest in THIS batch (+1)
padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
```
The `+1` makes room for one extra end-of-text token used when building shifted targets.

### What you see in practice (book pp. 225–226)
With `batch_size = 8` on the 935-example training set:
```
torch.Size([8, 61]) torch.Size([8, 61])
torch.Size([8, 76]) torch.Size([8, 76])
torch.Size([8, 73]) torch.Size([8, 73])
...
torch.Size([8, 74]) torch.Size([8, 74])
torch.Size([8, 69]) torch.Size([8, 69])
```
Batch size stays 8, but the width changes: 61, 76, 73, … Inputs and targets always share the same
shape within a batch. The width can never exceed **1,024** because the collate function is created
with `allowed_max_length=1024` (GPT-2's [[context-length]]).

## 🔴 Theory (Expert)
- **Cost argument.** Per batch, compute grows with $B\cdot T_b$ tokens through the
  [[feed-forward-network]] and with $B\cdot T_b^2$ in [[self-attention]] score matrices. Wasted
  pad tokens per batch are $\sum_{i\in b}(T_b-\ell_i)$; dynamic padding minimizes this for a fixed
  batch assignment, while static padding wastes $\sum_{i\in b}(T^\star-\ell_i)\ge$ that.
- **Correctness is unaffected.** Padding goes on the right. Under the [[causal-attention-mask]],
  a real token at position $t$ attends only to positions $\le t$, so it never sees pad tokens, and
  pad positions in the targets (beyond the first) are set to `-100` and excluded from the loss
  ([[ignore-index-loss-masking]]). So the width of the batch doesn't change what is learned —
  only how much is computed.
- **Shuffling interacts with waste.** With `shuffle=True` (training loader), batch membership is
  random, so $T_b$ is the max of 8 random lengths. *Length bucketing* (grouping similar lengths
  before batching) would shrink waste further; the book doesn't do this — mentioned here as a
  standard extension.
- **Contrast with chapter 6.** For classification, the book padded all messages to the longest one
  in the training set (static padding). Exercise 6.1 (padding to the full 1,024 context) made test
  accuracy drop to **78.33%** vs. 95.67% (appendix C, book p. 306) — a vivid reminder that heavy
  padding can hurt when the prediction is read from the last token. In chapter 7, the loss covers
  many positions and pads are masked, so this effect doesn't apply the same way.

## Connections
- Builds on: [[padding-and-truncation]], [[end-of-text-token]], [[mini-batch]], [[tensor]]
- Leads to: [[collate-function]], [[ignore-index-loss-masking]]
- Used by: [[instruction-fine-tuning]], [[pytorch-dataset-and-dataloader]]
- Related: [[context-length]], [[causal-attention-mask]], [[self-attention]], [[last-token-pooling]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 214–215 (per-batch padding, Figure 7.8, `custom_collate_draft_1`, 50256 check).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.4, book pp. 224–226 (`allowed_max_length=1024`, batch shapes 8×61, 8×76, …).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, book p. 306 (Exercise 6.1: 78.33% vs 95.67%).
