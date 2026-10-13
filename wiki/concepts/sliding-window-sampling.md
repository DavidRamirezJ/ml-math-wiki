---
title: Sliding-Window Data Sampling (Stride)
type: concept
tags: [llm, data-preparation, pytorch, next-token-prediction]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Sliding-Window Data Sampling (Stride)

> **One-liner:** Slide a fixed-size "window" along a long list of tokens; what's inside the window is the input, and the same window nudged one token to the right is the answer the model must predict.

## 🟢 Intuition (Dummy)
Imagine a whole book written on one very long ribbon. You hold a little cardboard frame that shows
exactly 4 words at a time. Whatever you see in the frame is a **question**: "what comes next?" The
**answer** is the same frame shifted one word to the right. Then you slide the frame along the
ribbon and get another question, and another, until the ribbon runs out.

How far you slide the frame each time is the **stride**:
- Slide by 1 word → the frames overlap a lot, you get tons of examples, but they're nearly copies.
- Slide by 4 words (the frame's own width) → no overlap, every word is used once, no near-duplicates.

That's all a GPT training set is: millions of these frames cut from text. Nobody has to write the
answers by hand — the text itself supplies them. (This is why pretraining is called
[[self-supervised-learning]].)

## 🟡 Mechanics (Practitioner)

### The shift-by-one rule
Take a token sequence $t_0, t_1, \dots, t_{N-1}$ (here, token IDs from the BPE tokenizer). A window
that starts at index $s$ with length $L$ gives
$$\mathbf{x} = (t_s, t_{s+1}, \dots, t_{s+L-1}), \qquad \mathbf{y} = (t_{s+1}, t_{s+2}, \dots, t_{s+L})$$
Reading it:
- $\mathbf{x}$ = the **input** chunk the model sees; $\mathbf{y}$ = the **target** chunk.
- $t_k$ = the [[token-id]] at position $k$ of the full text.
- $s$ = where the window starts; $L$ = window length (`max_length`, the [[context-length]] used for training).
- So $y_j = x_{j+1}$: the targets are "the inputs shifted by 1" (book p. 36).

One window holds **$L$ prediction tasks**, not one: target $y_j$ is the answer for the prefix
$x_0,\dots,x_j$. The book's example on "The Verdict" (5,145 BPE tokens; first 50 removed;
`context_size = 4`, book pp. 35–36):

```
x: [290, 4920, 2241, 287]
y:      [4920, 2241, 287, 257]

[290] ----> 4920                     " and" ----> " established"
[290, 4920] ----> 2241               " and established" ----> " himself"
[290, 4920, 2241] ----> 287          " and established himself" ----> " in"
[290, 4920, 2241, 287] ----> 257     " and established himself in" ----> " a"
```

Fig. 2.12 (book p. 35) shows the same idea on "LLMs learn to predict one word at a time": the model
"can't access words past the target" — a preview of the [[causal-attention-mask]].

### The stride
Windows start at $s_k = k\cdot\text{stride}$, $k=0,1,2,\dots$. Consecutive windows share
$$\text{overlap} = \max(0,\ L - \text{stride})\ \text{tokens}$$
and the number of windows (rows) in the dataset is
$$\text{rows} = \left\lceil \frac{N-L}{\text{stride}} \right\rceil$$
- $N$ = total number of tokens; $\lceil\cdot\rceil$ = "round up to a whole number".
- These two formulas are our derivation from the code `range(0, len(token_ids) - max_length, stride)`; the book doesn't print them.
- Example: $N=5145$, $L=4$, stride $1$ → $5141$ rows; $L=256$, stride $128$ → $\lceil 4889/128\rceil = 39$ rows (our arithmetic).

Fig. 2.14 (book p. 40) on "In the heart of the city stood the old library…":
stride 1 → batch 1 = "In the heart of", batch 2 = "the heart of the"; stride 4 → batch 2 = "the city stood the".

### The book's code (Listing 2.5, book pp. 37–38)
```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):   # the sliding window
            input_chunk  = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]    # shifted by one
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):            return len(self.input_ids)
    def __getitem__(self, idx):   return self.input_ids[idx], self.target_ids[idx]
```
Listing 2.6 wraps it in `create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
shuffle=True, drop_last=True, num_workers=0)` with the GPT-2 tokenizer (book p. 38). `drop_last=True`
drops a short final batch, which "prevents loss spikes during training" (book p. 38). See
[[pytorch-dataset-and-dataloader]].

**Worked runs:**
- `batch_size=1, max_length=4, stride=1` → first batch `[[40, 367, 2885, 1464]]` / targets
  `[[367, 2885, 1464, 1807]]`; second batch input `[[367, 2885, 1464, 1807]]` — shifted by one (book p. 39).
- `batch_size=8, max_length=4, stride=4` → an $8\times4$ input tensor starting
  `[[40, 367, 2885, 1464], [1807, 3619, 402, 271], …]` and targets `[[367, 2885, 1464, 1807], …]`.
  Stride 4 = window width means "we don't skip a single word" and avoid overlap, "since more overlap
  could lead to increased overfitting" (book pp. 40–41).
- Exercise 2.2 (book p. 39; solution App. C, book p. 301): `max_length=2, stride=2` gives rows
  `[40, 367], [2885, 1464], …` (no overlap); `max_length=8, stride=2` gives rows that overlap by 6.

**Batch shapes:** inputs and targets are both `(batch_size, max_length)` [[tensor|tensors]] — read off
the printed `(1, 4)` and `(8, 4)` tensors.

### Where the book reuses it
- **Pretraining (ch. 5):** 90/10 train/validation split of "The Verdict"; `max_length = stride = 256`
  (`GPT_CONFIG_124M["context_length"]`), `batch_size=2` → **9 training batches** and **1 validation
  batch** of shape `[2, 256]` (book pp. 141–144). Real LLMs use batch sizes "of 1,024 or larger"
  (book p. 143). Fig. 5.9 shows chunks of length 6, stride 6, shuffled into batches of 2 (book p. 142).
- **Appendix D:** the same loaders, `max_length = stride = 256`, for warmup/cosine-decay experiments (book p. 314).
- **Classification fine-tuning (ch. 6):** no sliding window any more — each SMS is one example,
  padded/truncated to 120 tokens; the book notes ch. 2's sliding-window chunks are the "conceptually
  similar" ancestor (book p. 175). See [[padding-and-truncation]].

## 🔴 Theory (Expert)
- **Why the targets give an MLE objective.** The windows implement the autoregressive factorization
  $P(t_1,\dots,t_T)=\prod_{i}P(t_i\mid t_{<i})$; summing cross-entropy over all $L$ positions of every
  window is the (negative) log-likelihood of the corpus, truncated to contexts of at most $L$ tokens.
  This links to [[maximum-likelihood-estimation]], [[cross-entropy-loss]] and [[next-token-prediction]].
  (Framing ours; the book introduces the loss in ch. 5.)
- **Stride as a bias–variance knob.** Stride $<L$ creates correlated samples: more gradient steps per
  epoch but less new information per step and a higher risk of memorization ([[overfitting]]) — the
  book's stated reason for stride $=L$ (book p. 41). Stride $>L$ would skip tokens entirely.
- **Context truncation.** A target at window position $j$ sees only $j$ tokens of context, so early
  positions in each window are trained on short contexts. With non-overlapping windows, dependencies
  that cross a window boundary are never learned in that epoch; shuffling plus different start
  offsets across epochs is a common mitigation (not discussed in the book).
- **Fixed-length chunks are a simplification.** The book notes that "in practice, it can also be
  beneficial to train an LLM with variable-length inputs to help the LLM to better generalize"
  (book p. 142).
- **Mini-batch view.** A batch is a $B\times L$ integer [[matrix]]; small $B$ uses less memory but
  gives "noisier model updates" (book pp. 39–40) — the gradient-variance $\propto 1/B$ story of
  [[stochastic-gradient-descent]] and [[mini-batch]].

## Connections
- Builds on: [[token-id]], [[tokenization]], [[byte-pair-encoding]], [[tensor]]
- Leads to: [[pytorch-dataset-and-dataloader]], [[mini-batch]], [[token-embedding]], [[llm-training-loop]], [[train-validation-test-split]]
- Used by: [[next-token-prediction]], [[pretraining]], [[self-supervised-learning]], [[causal-attention-mask]]
- Related: [[context-length]], [[hyperparameter]], [[overfitting]], [[padding-and-truncation]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1 (chapter goal: sampling with a sliding window), book p. 17; §2.6 (Figs. 2.12–2.14, Listings 2.5–2.6, Exercise 2.2), book pp. 35–41; §2.8 (loader with `stride = max_length = 4`), book p. 46.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3 (train/val loaders, Fig. 5.9), book pp. 141–144.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3 (contrast with padded per-example datasets), book p. 175.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` App. C (Exercise 2.2 solution), book p. 301; App. D §D.1 (loaders), book p. 314.
