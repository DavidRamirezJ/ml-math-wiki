---
title: Loss Masking with ignore_index (-100)
type: concept
tags: [loss-function, cross-entropy, pytorch, instruction-fine-tuning, masking]
difficulty: intermediate
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Loss Masking with ignore_index (-100)

> **One-liner:** Putting `-100` in a target slot tells PyTorch's cross-entropy "skip this position — don't grade it," so padding filler doesn't pollute training.

## 🟢 Intuition (Dummy)
Imagine a teacher grading a fill-in-the-blank test. Some students wrote short answers, so the
rest of their answer sheet is just blank filler lines. If the teacher graded those filler lines
too, every short answer would get weirdly scored — and students would learn "write lots of filler!"

So the teacher crosses out the filler lines with a big **"N/A"** stamp and only averages the real
answers. In PyTorch, that stamp is the number **`-100`**. Any target position labeled `-100` is
simply left out of the loss.

One clever twist: the book keeps **one** end-of-text token un-stamped at the end of every answer.
That way the model still learns "after the answer, say *I'm done*."

## 🟡 Mechanics (Practitioner)

### The masked cross-entropy formula
For positions $i$ with logits $\mathbf z_i\in\mathbb R^V$ and target token $y_i$:

$$\mathcal{L} = -\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}} \log \frac{e^{z_{i,y_i}}}{\sum_{k=1}^{V} e^{z_{i,k}}},\qquad \mathcal{I}=\{\,i : y_i \neq -100\,\}$$

Symbol by symbol:
- $i$ = a position in the batch (one predicted token),
- $z_{i,k}$ = the [[logits|logit]] (raw score) the model gives token $k$ at position $i$,
- $V$ = vocabulary size (50,257 for GPT-2; just 2 in the toy example below),
- $\frac{e^{z_{i,y_i}}}{\sum_k e^{z_{i,k}}}$ = the [[softmax]] probability of the *correct* token $y_i$,
- $-\log(\cdot)$ = penalty: big when that probability is small (see [[log-probability]]),
- $\mathcal{I}$ = the set of positions **not** marked `-100`,
- $|\mathcal{I}|$ = how many such positions there are — we average only over them.

This is standard [[cross-entropy-loss]], just with the average restricted to a subset.

### The book's toy demo (book pp. 221–222)
```python
logits_1  = torch.tensor([[-1.0, 1.0], [-0.5, 1.5]])
targets_1 = torch.tensor([0, 1])
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)   # tensor(1.1269)

logits_2  = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])
targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)   # tensor(0.7936)

targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)   # tensor(1.1269)
print("loss_1 == loss_3:", loss_1 == loss_3)                      # tensor(True)
```

Check by hand:
- position 1, target 0: $-\log\frac{e^{-1}}{e^{-1}+e^{1}} = \log(1+e^{2}) \approx 2.1269$
- position 2, target 1: $-\log\frac{e^{1.5}}{e^{-0.5}+e^{1.5}} = \log(1+e^{-2}) \approx 0.1269$
- `loss_1` $= (2.1269+0.1269)/2 = 1.1269$
- `loss_2` adds a third 0.1269: $(2.1269+0.1269+0.1269)/3 = 0.7936$ — an extra token changes the loss
- `loss_3` marks the third as `-100`, so $|\mathcal I| = 2$ and the loss is back to **1.1269**

Why it works: PyTorch's default is `cross_entropy(..., ignore_index=-100)` (book p. 222). If you
replace `-100` with some other ID that isn't 0 or 1 here, you get an error — presumably because a
target class must lie in $\{0,\dots,V-1\}$ (our explanation; the book only reports the error).

### Applying it to padding (step 2.5, Figure 7.12, book p. 219)
In the [[collate-function]], targets after the shift look like `[6, 50256, 50256, 50256, 50256]`.
All pads **except the first** become `-100`:

| Before | After |
|---|---|
| `[1, 2, 3, 4, 50256]` | `[1, 2, 3, 4, 50256]` (unchanged) |
| `[6, 50256, 50256, 50256, 50256]` | `[6, 50256, -100, -100, -100]` |
| `[8, 9, 50256, 50256, 50256]` | `[8, 9, 50256, -100, -100]` |

Code (Listing 7.5, book p. 220):
```python
mask = targets == pad_token_id               # boolean: where are the pads?
indices = torch.nonzero(mask).squeeze()      # their positions
if indices.numel() > 1:                      # more than one pad?
    targets[indices[1:]] = ignore_index      # all but the first -> -100
```
Keeping that first `50256` teaches the model to emit end-of-text when its answer is finished —
which `generate(..., eos_id=50256)` later uses as the stop signal (book pp. 218, 234).

(Small book slip: Figure 7.12's caption says "Step 2.4", but it shows step 2.5.)

In chapter 6 this wasn't needed, because classification trained only on the **last** output token
(book p. 218; see [[last-token-pooling]]).

### Optional: masking the instruction too (Figure 7.13, book p. 222)
You can also set the target IDs of the instruction (and input) tokens to `-100`, so the loss
covers **only the response**. Target looks like `[-100, -100, -100, -100, -100, …, 13, 50256]`.
The idea is that the model focuses on writing answers instead of memorizing instructions, which can
reduce [[overfitting]]. The book does **not** do this by default — see [[instruction-masking]].

## 🔴 Theory (Expert)
- **Masked ERM.** With mask $m_i = \mathbb 1[y_i\neq -100]$, the loss is
  $\mathcal L = \frac{\sum_i m_i\,\ell_i}{\sum_i m_i}$ with $\ell_i=-\log p_\theta(y_i\mid x_{\le i})$.
  The gradient $\nabla_\theta\mathcal L = \frac{1}{\sum m_i}\sum_i m_i\nabla_\theta \ell_i$ — masked
  positions contribute exactly zero gradient, so pad positions are invisible to
  [[backpropagation]]. This is still [[maximum-likelihood-estimation]] of the response tokens.
- **Normalization subtlety.** PyTorch's mean reduction divides by the number of *non-ignored*
  targets in the whole batch, so long examples weigh more per batch than short ones (token-level
  averaging, not example-level). That's what the demo's `loss_1 == loss_3` shows.
- **Why `-100`?** It's simply PyTorch's default `ignore_index`; any integer outside
  $\{0,\dots,V-1\}$ could be configured, but `-100` is the convention the library recognizes out of
  the box.
- **The masking debate.** Shi et al. (2024), *Instruction Tuning With Loss Over Instructions*
  (arXiv:2405.14394), found **not** masking instructions can help — especially with long
  instructions and short outputs, or few training examples (appendix B, book p. 297). The book's
  own Exercise 7.2 confirms it on this setup: masking scored about **4 points worse** with the
  Llama 3 judge (appendix C, book pp. 308–309). The masked version adds
  `targets[:instruction_length-1] = -100`, the `-1` accounting for the one-position shift.
- **Interaction with dynamic padding.** Because masked positions contribute nothing,
  [[dynamic-padding]] can vary batch width freely without changing the objective.

## Connections
- Builds on: [[cross-entropy-loss]], [[softmax]], [[logits]], [[log-probability]], [[end-of-text-token]], [[padding-and-truncation]]
- Leads to: [[instruction-masking]], [[instruction-fine-tuning]]
- Used by: [[collate-function]], [[dynamic-padding]], [[llm-training-loop]]
- Related: [[loss-function]], [[maximum-likelihood-estimation]], [[empirical-risk-minimization]], [[overfitting]], [[next-token-prediction]]
- People: [[zhengyan-shi]] (Shi et al. 2024), [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 218–223 (step 2.5, Figures 7.11–7.13, Listing 7.5, cross-entropy demo 1.1269 / 0.7936 / 1.1269, Shi et al. 2024, Exercise 7.2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book p. 297 (Shi 2024 finding).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, book pp. 308–309 (Exercise 7.2 solution, ~4 points worse).
