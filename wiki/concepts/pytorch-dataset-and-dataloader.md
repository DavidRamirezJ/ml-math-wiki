---
title: PyTorch Dataset and DataLoader
type: concept
tags: [pytorch, data-pipeline, training, llm]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# PyTorch Dataset and DataLoader

> **One-liner:** A `Dataset` knows how to fetch one example; a `DataLoader` shuffles those examples, groups them into batches, and hands the batches to your training loop.

## 🟢 Intuition (Dummy)
Think of a restaurant kitchen.
- The **`Dataset`** is the pantry clerk. Ask for item number 7 and you get exactly that one
  item: one email and its label, or one chunk of text and its next-word targets. It also
  knows how many items there are in total.
- The **`DataLoader`** is the waiter. It shuffles the order, picks up a tray of, say, 8 items
  from the clerk, stacks them neatly into one tray (a **batch**), and carries it to the model.
  Next trip, next tray. When the pantry has been fully served once, that is one **[[epoch]]**.

Splitting the job like this means you write the "how do I read one example" part once, and
reuse the same waiter logic for training, validation and test data (Figure A.10, book p. 270).

A practical bonus: the waiter can have helpers (`num_workers`). Without helpers, the model
(often on a fast GPU) sits idle while the CPU prepares the next tray. With helpers, the next
trays are already being prepared in the background (Figure A.11, book p. 274).

## 🟡 Mechanics (Practitioner)

### The `Dataset` contract (Listing A.6, book p. 271)
A custom dataset subclasses `torch.utils.data.Dataset` and defines three methods:

```python
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):          # set up attributes (tensors, file paths, DB handles...)
        self.features = X
        self.labels = y
    def __getitem__(self, index):      # return exactly ONE example, chosen by index
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    def __len__(self):                 # total number of examples
        return self.labels.shape[0]
```
With the 5-example toy set (2 features, labels `[0, 0, 0, 1, 1]`), `len(train_ds)` prints `5`
(book pp. 270–272). The data loader supplies the index to `__getitem__`.

### The `DataLoader` (Listing A.7, book p. 272)
```python
from torch.utils.data import DataLoader
torch.manual_seed(123)
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0)
test_loader  = DataLoader(dataset=test_ds,  batch_size=2, shuffle=False, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
```
Output (book p. 272):
```
Batch 1: tensor([[-1.2000, 3.1000], [-0.5000, 2.6000]]) tensor([0, 0])
Batch 2: tensor([[ 2.3000, -1.1000], [-0.9000, 2.9000]]) tensor([1, 0])
Batch 3: tensor([[ 2.7000, -1.5000]]) tensor([1])
```
The loader stacks the per-item outputs along a new first axis (the batch axis), so `x` has
shape `[2, 2]` and `y` shape `[2]`; the last batch is short, `[1, 2]`, because 5 is not
divisible by 2.

### The knobs, and what they mean mathematically
| Argument | Meaning | Book's reason |
|---|---|---|
| `batch_size` $B$ | examples per batch | memory vs. update noise trade-off (book pp. 39–40); see [[mini-batch]] |
| `shuffle=True` | new random order each epoch (train only) | avoid "repetitive update cycles" (book p. 272); not needed for test (p. 272) |
| `drop_last=True` | drop a final incomplete batch (train only) | a much smaller last batch "can disturb convergence" (p. 273) / causes "loss spikes" (p. 38) |
| `num_workers` | background processes that load data | avoid the GPU-idle bottleneck (pp. 273–274) |
| `collate_fn` | how a list of items becomes one batch | needed for variable-length instruction data (Ch. 7) |

**Number of batches per epoch.** With $N$ examples and batch size $B$:

$$\#\text{batches} = \begin{cases}\lfloor N/B\rfloor & \texttt{drop\_last=True}\\[2pt] \lceil N/B\rceil & \texttt{drop\_last=False}\end{cases}$$

- $\lfloor\cdot\rfloor$ ("floor") = round down; $\lceil\cdot\rceil$ ("ceiling") = round up.
- Toy set: $\lceil 5/2\rceil=3$ batches (sizes 2, 2, 1); with `drop_last=True`, $\lfloor 5/2\rfloor=2$ (book pp. 272–273).
- Spam data (Ch. 6): 1,045 / 149 / 300 examples, $B=8$ → $\lfloor 1045/8\rfloor=130$ training, $\lceil 149/8\rceil=19$ validation, $\lceil 300/8\rceil=38$ test batches. The book prints 130 / 19 / 38 (book p. 181); the split sizes are derived from the 70/10/20 split of 1,494 rows.

**Shuffling** draws a fresh random permutation $\pi$ of $\{1,\dots,N\}$ each epoch; batch $k$ is
$\{\pi((k-1)B+1),\dots,\pi(kB)\}$. `torch.manual_seed(123)` makes the sequence of permutations
reproducible (see [[random-seed-reproducibility]]).

### The four datasets of the book
**1. Pretraining text, `GPTDatasetV1` (Listing 2.5, book pp. 37–38).** Cuts one long token
sequence into overlapping windows with [[sliding-window-sampling]]:
```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.target_ids[idx]
```
Window $k$ starts at $s_k=k\cdot\text{stride}$ and gives
$$x=(t_{s},\dots,t_{s+L-1}),\qquad y=(t_{s+1},\dots,t_{s+L})$$
- $t_i$ = the $i$-th token ID of the whole text; $L$ = `max_length` ([[context-length]]); $s$ = window start.
- $y$ is $x$ shifted one step to the left: the target at each position is the next token ([[next-token-prediction]]).

`create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True,
drop_last=True, num_workers=0)` wraps it with the GPT-2 tokenizer (Listing 2.6, book p. 38).
With `batch_size=1, max_length=4, stride=1` the first batch is
`[tensor([[40, 367, 2885, 1464]]), tensor([[367, 2885, 1464, 1807]])]` (book p. 39); with
`batch_size=8, max_length=4, stride=4` both tensors have shape `[8, 4]` (book pp. 40–41).

**2. Toy classification, `ToyDataset`** (above, Appendix A).

**3. Spam classification, `SpamDataset` (Listing 6.4, book pp. 176–178).** Reads a CSV,
pre-tokenizes every message, finds the longest one (120 tokens), and pads all others with token
ID 50256 (`<|endoftext|>`). `__getitem__` returns `(input_ids, label)` as `torch.long` tensors.
Validation/test sets are padded or truncated to the training max length. Batches: inputs
`[8, 120]`, labels `[8]` (book p. 180). Appendix E reuses this exact pipeline for LoRA
(Listings E.1–E.3, book pp. 324–325). See [[padding-and-truncation]].

**4. Instructions, `InstructionDataset` + custom collate (Listings 7.4–7.6, book pp. 213–225).**
`__getitem__` returns an *unpadded* list of token IDs. A [[collate-function]] does the batching:
pad each batch to its own longest example ([[dynamic-padding]]), build the shifted targets,
replace all but the first padding token in the targets with −100 ([[ignore-index-loss-masking]]),
optionally truncate to 1,024 tokens, and move tensors to the device:
```python
from functools import partial
customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=customized_collate_fn,
                          shuffle=True, drop_last=True, num_workers=0)
```
`partial` pre-fills the `device` and `allowed_max_length` arguments, because the loader calls
`collate_fn(batch)` with the batch only (book p. 224). Result: batches of shape `[8, 61]`,
`[8, 76]`, `[8, 73]`, … (the length changes from batch to batch) (book pp. 225–226).

### `num_workers`: parallel loading (book pp. 273–274)
- `num_workers=0`: data is loaded in the main process; on a GPU this can leave the GPU idle while
  the CPU prepares the next batch.
- `num_workers>0`: several worker processes load batches in parallel and queue them.
- Caveats: for tiny datasets or in Jupyter notebooks, the overhead of starting workers can exceed
  the loading time, and multiprocessing in notebooks can cause errors or crashes.
- The author's experience: `num_workers=4` is often optimal on real datasets, depending on hardware
  and on the `Dataset` code. The book's chapters use `num_workers=0` "to ensure compatibility with
  most computers" (book p. 180).
- In Ch. 7 the device transfer is moved into the collate function so it can happen outside the
  training loop and not block the GPU (book p. 224).

## 🔴 Theory (Expert)
- **Sampling model.** With `shuffle=True` an epoch is sampling *without replacement*: each example
  appears once per epoch (book p. 272 defines an epoch exactly this way). Every mini-batch gradient
  is then an unbiased estimate of the full-data gradient when averaged over the random permutation,
  which is what [[stochastic-gradient-descent]] needs. Fixed order (`shuffle=False`) correlates
  consecutive updates; in a sorted dataset this can bias the trajectory, the "repetitive update
  cycles" the book warns about.
- **Why drop the last batch.** The variance of a mean over $b$ per-example gradients scales like
  $\sigma^2/b$. A last batch with $b=N\bmod B\ll B$ gives a much noisier step, and in mean-reduced
  losses it gets the *same* step weight as a full batch. Dropping it removes that outlier (the
  "loss spikes" of book p. 38). Evaluation loaders keep it (`drop_last=False`) so that every example
  is counted. Note that with `drop_last=True` and shuffling, a different example is skipped each
  epoch.
- **Variable-length sequences.** The default collate calls `torch.stack`, which requires equal
  shapes. Ch. 6 solves this by padding to a global length (simple, but wastes compute on padding);
  Ch. 7 pads per batch (less padding, variable $T_b$). The masked mean loss over
  $\mathcal{I}=\{i: y_i\neq-100\}$ makes the padding invisible to the gradient.
- **Sliding-window overlap.** Consecutive windows share $\max(0, L-\text{stride})$ tokens and there are
  $\lceil (N-L)/\text{stride}\rceil$ windows (derived from `range(0, N-L, stride)`). Overlap creates
  near-duplicate examples; the book sets `stride = max_length` in its demo so it "doesn't skip a
  single word" yet avoids overlap, which "could lead to increased overfitting" (book p. 41).
- **Distributed loading.** Under [[distributed-data-parallel]] a `DistributedSampler` partitions the
  index set into disjoint shards, one per GPU, and takes over shuffling (`shuffle=False` in the
  loader, `sampler.set_epoch(epoch)` each epoch); `pin_memory=True` speeds host-to-GPU copies
  (book p. 285).
- **Throughput view.** Training is a producer–consumer pipeline: if the time to produce a batch
  $t_{\text{load}}$ exceeds the time to consume it $t_{\text{step}}$, the accelerator utilisation is at
  most $t_{\text{step}}/t_{\text{load}}$. Workers raise the production rate roughly linearly until CPU
  cores or I/O saturate.

## Connections
- Builds on: [[tensor]], [[pytorch]], [[train-validation-test-split]], [[tokenization]], [[token-id]]
- Produces: [[mini-batch]], [[epoch]]
- Uses: [[sliding-window-sampling]], [[collate-function]], [[dynamic-padding]], [[padding-and-truncation]], [[end-of-text-token]], [[ignore-index-loss-masking]], [[random-seed-reproducibility]]
- Feeds: [[training-loop]], [[llm-training-loop]], [[stochastic-gradient-descent]], [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]]
- Related: [[gpu-computing]], [[distributed-data-parallel]], [[overfitting]], [[instruction-dataset]], [[label-encoding]]
- People: [[sebastian-raschka]]

> **From [[llm-appD-training-loop-bells-and-whistles]]:** Appendix D reuses `create_dataloader_v1` from Ch. 2 with `batch_size=2`, `max_length = stride = 256` (no overlap), `drop_last=True, shuffle=True` for training and `drop_last=False, shuffle=False` for validation, on a 90/10 split of "The Verdict" (book p. 314). The schedule code uses `len(train_loader) * n_epochs` as the total number of steps (book p. 315).

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** §5.1.3 reuses `create_dataloader_v1` from Ch. 2 to build two loaders from a 90/10 split of *The Verdict*. The training loader uses `drop_last=True, shuffle=True`. The validation loader uses `drop_last=False, shuffle=False`, so every validation token is used and the order is fixed. Both use `batch_size=2`, `max_length=stride=256` and `num_workers=0`, after `torch.manual_seed(123)` (book p. 143). `calc_loss_loader` (Listing 5.2) iterates over a loader and returns `float("nan")` if the loader is empty (book p. 144). The NOTE on p. 142 says that the book uses equal-length chunks for simplicity, while training with variable-length inputs usually helps the LLM generalize.

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6 "Data sampling with a sliding window", book pp. 35–41 (Listings 2.5–2.6, `GPTDatasetV1`, `create_dataloader_v1`, drop_last/num_workers annotations, printed batches).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.2–6.3, book pp. 175–181 (70/10/20 split, `SpamDataset`, Listing 6.5, 130/19/38 batches).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3–7.4, book pp. 211–226 (default vs custom collate, `InstructionDataset`, `partial`, Listing 7.6, variable batch shapes).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.6 "Setting up efficient data loaders", book pp. 270–274 (Figures A.10–A.11, Listings A.5–A.8, epoch definition, num_workers caveats).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.9.3, book p. 285 (DistributedSampler, pin_memory).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.2, book pp. 324–325 (Listings E.1–E.3, reuse of the Ch. 6 pipeline).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D intro, book p. 314 (data loaders); §D.1, book p. 315 (`len(train_loader)` for total steps).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3, book pp. 142–145 (loader settings, variable-length NOTE, Listing 5.2).
