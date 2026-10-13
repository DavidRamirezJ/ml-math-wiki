---
title: Custom Collate Function
type: concept
tags: [pytorch, data-loading, instruction-fine-tuning, batching, padding]
difficulty: intermediate
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Custom Collate Function

> **One-liner:** A collate function takes a handful of separate examples and glues them into one neat batch tensor the model can chew on in one go.

## 🟢 Intuition (Dummy)
Picture packing eggs into a carton. A carton only works if every slot is the same size. But your
training sentences are all different lengths — some are 5 tokens, some 76.

The **collate function** is the packer. It takes, say, 8 sentences and:
1. stretches the short ones with "filler" tokens so they all fit the same carton width,
2. makes a second carton with the *answers* (each sentence shifted by one word, since the model's
   job is to guess the next word),
3. marks the filler in the answer carton with a special "don't grade this" sticker (`-100`).

PyTorch's `DataLoader` ships with a default packer, but it only works if everything is already the
same size. Instruction data isn't, so the book writes its own.

## 🟡 Mechanics (Practitioner)

### What a collate function is
In PyTorch, a `DataLoader` pulls a list of samples from a `Dataset` and calls
`collate_fn(batch)` to merge them into one batch. In chapter 6 the **default collate function**
did this automatically. For instruction fine-tuning we need a **custom collate function** that
handles the specific formatting needs of the [[instruction-dataset]] (book pp. 211–212).

### The five sub-steps of batching (Figure 7.6, book p. 212)
| Step | What happens | Example |
|---|---|---|
| 2.1 | Format with the [[prompt-template]] | "Below is an instruction…" |
| 2.2 | Tokenize | `[21106, 318, 281, 12064, …]` |
| 2.3 | Pad to equal length with `50256` | `[…, 50256, 50256, 50256]` |
| 2.4 | Make targets = inputs shifted by one, plus a pad | `[318, 281, 12064, …, 50256]` |
| 2.5 | Replace extra pads in targets with `-100` | `[…, 50256, -100, -100, -100]` |

Steps 2.1–2.2 live in the `InstructionDataset` class (it pre-tokenizes everything in `__init__`).
Steps 2.3–2.5 live in the collate function.

### The shift-by-one formula (step 2.4)
For one padded sequence $p = (p_0, p_1, \dots, p_L)$ of length $L+1$:

$$x_t = p_t,\qquad y_t = p_{t+1},\qquad t = 0,\dots,L-1$$

- $p_t$ = the token ID at position $t$ of the padded list,
- $x_t$ = the **input** at position $t$ → code `padded[:-1]` (everything except the last),
- $y_t$ = the **target** at position $t$ → code `padded[1:]` (everything except the first),
- so "the target at $t$ is the input at $t+1$": the model learns [[next-token-prediction]].

### Listing 7.5 — the final version (book p. 220)
```python
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]                      # one extra end-of-text
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs  = torch.tensor(padded[:-1])             # drop last token
        targets = torch.tensor(padded[1:])              # shift +1
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index         # all but first pad -> -100
        if allowed_max_length is not None:
            inputs  = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs); targets_lst.append(targets)
    inputs_tensor  = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
```

### Worked example (book pp. 215–221)
```python
inputs_1 = [0, 1, 2, 3, 4]; inputs_2 = [5, 6]; inputs_3 = [7, 8, 9]
inputs, targets = custom_collate_fn((inputs_1, inputs_2, inputs_3))
```
```
inputs:  [[0, 1, 2, 3, 4],
          [5, 6, 50256, 50256, 50256],
          [7, 8, 9, 50256, 50256]]
targets: [[1, 2, 3, 4, 50256],
          [6, 50256, -100, -100, -100],
          [8, 9, 50256, -100, -100]]
```
Both are $3\times 5$. `batch_max_length` = 5 + 1 = 6; one token is lost to the shift, so each row
has 5. Each target row keeps exactly **one** `50256` so the model learns when to stop; the rest
become `-100` (see [[ignore-index-loss-masking]]). The book built this up in drafts:
`custom_collate_draft_1` (padding only, p. 215) → `custom_collate_draft_2` (+ targets, pp. 217–218)
→ Listing 7.5.

`allowed_max_length` optionally truncates samples — handy if your data exceeds GPT-2's
**1,024-token** [[context-length]] (book p. 218).

### Plugging it into the DataLoader with `functools.partial` (book p. 224)
`DataLoader` calls `collate_fn(batch)` with *only* the batch. To fix the other arguments in
advance, the book uses **partial function application**:

```python
from functools import partial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
```

In math terms: if $f(\text{batch}, p, i, L, d)$ is the full function (pad ID, ignore index,
max length, device), `partial` builds a new function
$$g(\text{batch}, p, i) = f(\text{batch}, p, i, L{=}1024, d{=}\text{device})$$
- $g$ = the new, "pre-filled" function,
- $L{=}1024$ and $d{=}\text{device}$ = the two arguments now frozen in place,
- $p$, $i$ keep their defaults (50256, −100).

(That's partial application, not strictly currying — currying would turn $f$ into a chain of
one-argument functions.)

**Why move to the device in the collate function?** Earlier chapters moved data to the GPU inside
the training loop. Doing it here lets the transfer happen as a background process outside the
training loop, so it doesn't block the GPU (book p. 224). Device options: `"cpu"`, `"cuda"`, or
`"mps"` (Apple Silicon — may give slight numerical differences since support was experimental).

Then (Listing 7.6, book p. 225):
```python
train_loader = DataLoader(InstructionDataset(train_data, tokenizer), batch_size=8,
                          collate_fn=customized_collate_fn, shuffle=True,
                          drop_last=True, num_workers=0)
```
(The listing first calls `torch.manual_seed(123)`; `val_loader` and `test_loader` are built the same
way but with `shuffle=False, drop_last=False`.)
The resulting batches are `torch.Size([8, 61])`, `[8, 76]`, `[8, 73]`, … — different lengths per
batch thanks to [[dynamic-padding]].

## 🔴 Theory (Expert)
- **Shapes.** For $B$ samples of lengths $\ell_1,\dots,\ell_B$, outputs are
  $\text{inputs},\text{targets}\in\mathbb{Z}^{B\times T}$ with $T=\max_i \ell_i$
  (because `batch_max_length` $=\max_i\ell_i+1$ and the shift drops one), capped at
  `allowed_max_length`. Stacking $B$ row vectors gives a [[matrix]] / [[tensor]].
- **Why pad in inputs but `-100` in targets.** Input pads are harmless: with a
  [[causal-attention-mask]], a real token at position $t$ only attends to positions $\le t$, and
  padding is always to the *right*, so real tokens never see pads. Their *outputs* exist but the
  corresponding targets are `-100` (except the first), so they contribute nothing to the
  [[cross-entropy-loss]].
- **Truncation caveat.** Truncating after the `-100` replacement can cut off the one kept
  end-of-text target for very long examples (our observation, not the book's). With this dataset's
  short examples (printed batches are only ~61–76 tokens wide vs. the 1,024 cap) it doesn't arise,
  but it matters for long-form data.
- **Exercise 7.2 variant (instruction masking).** The dataset returns `(instruction_length, item)`
  and the collate function adds `targets[:instruction_length-1] = -100` (the `-1` because targets are
  shifted by one). This scored about **4 points worse** with the Llama 3 judge, consistent with
  Shi et al. (2024) (book pp. 308–309). See [[instruction-masking]].
- **Design principle.** The collate function is the right place for *batch-dependent* logic
  (padding length depends on which samples land together), while the `Dataset` holds
  *sample-dependent* logic (formatting, tokenizing).

## Connections
- Builds on: [[pytorch-dataset-and-dataloader]], [[tensor]], [[mini-batch]], [[padding-and-truncation]], [[end-of-text-token]], [[next-token-prediction]]
- Leads to: [[dynamic-padding]], [[ignore-index-loss-masking]], [[instruction-masking]], [[llm-training-loop]]
- Used by: [[instruction-fine-tuning]], [[training-loop]]
- Related: [[prompt-template]], [[instruction-dataset]], [[context-length]], [[gpu-computing]], [[random-seed-reproducibility]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.2, book p. 211 (default collate is not enough).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 211–223 (Figure 7.6, Listings 7.4–7.5, drafts 1–2, worked example).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.4, book pp. 223–226 (device in collate, `functools.partial`, Listing 7.6, batch shapes).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.9, book p. 249 (summary: custom collate pads, creates targets, masks padding).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, book pp. 308–309 (Exercise 7.2 masking collate).
