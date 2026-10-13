---
title: Distributed Data Parallel (DDP)
type: concept
tags: [deep-learning, pytorch, gpu, distributed-training, optimization, llm]
difficulty: intermediate
sources: [llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Distributed Data Parallel (DDP)

> **One-liner:** DDP trains one model on several GPUs at once by giving every GPU its own copy of the model and a different slice of the data, then averaging their gradients so all copies stay identical.

## 🟢 Intuition (Dummy)
Imagine grading 1,000 exam papers. One teacher takes ages. So you hire four
teachers, give each one an identical copy of the answer key, and split the
pile so nobody grades the same paper twice. At the end of each round they get
together, compare notes on "what we should change in the answer key", agree on
the **average** suggestion, and all four update their keys the same way. Then
they keep going.

That's DDP. The answer key is the model, the teachers are GPUs, the piles are
data batches, and the "compare notes and average" step is gradient
synchronization. Four teachers → roughly four times faster, minus a little time
spent talking to each other.

## 🟡 Mechanics (Practitioner)

### Why bother (book p. 282)
Even when one GPU *can* train a model, it may take too long. Splitting the work
across several GPUs, and possibly several machines, can cut training time a
lot. This matters most during experimentation, when you have to run training
many times to tune settings and architecture. (The book is clear that you
don't need multiple GPUs to follow it; this section is for the curious.)

### How DDP works, step by step (book pp. 282–283, Figs. A.12–A.13)
1. PyTorch starts **one separate process per GPU**.
2. The model is created on the CPU and then **copied to every GPU** (each copy is a *replica*).
3. A **`DistributedSampler`** splits the data so that each GPU gets a **different, non-overlapping** minibatch each iteration.
4. Each GPU runs its own forward pass (its own [[logits]]) and its own backward pass, so each one ends up with **different gradients**.
5. Before updating the weights, the gradients are **averaged and synchronized** across all GPUs.
6. Every replica applies the **same** averaged update, so the copies never drift apart.

### The averaging formula (the book says this in words)
$$
\mathbf{g} \;=\; \frac{1}{W}\sum_{r=0}^{W-1} \nabla_\theta \mathcal{L}_r(\theta),
\qquad
\theta \;\leftarrow\; \theta - \eta\,\mathbf{g}
$$

Reading it symbol by symbol:
- $W$ = the **world size**, i.e. the number of processes / GPUs (`world_size` in the code)
- $r$ = the **rank**, the ID of one GPU, running from $0$ to $W-1$
- $\sum_{r=0}^{W-1}$ = "add up one term per GPU"
- $\mathcal{L}_r(\theta)$ = the loss that GPU $r$ computed on *its own* minibatch
- $\nabla_\theta \mathcal{L}_r$ = that GPU's gradient (what `loss.backward()` puts in `.grad`; see [[gradient]])
- $\frac{1}{W}$ = divide by the number of GPUs, so the sum becomes an average
- $\mathbf{g}$ = the averaged gradient, identical on every GPU after synchronization
- $\eta$ = the [[learning-rate]] (`lr=0.5` in the book's example)
- $\theta \leftarrow \theta - \eta\,\mathbf{g}$ = the usual [[stochastic-gradient-descent]] step, done the same way on every replica

The operation that computes $\mathbf{g}$ and hands the result back to every GPU
is called an **all-reduce** (here: an all-reduce *mean*). DDP does it for you
during `backward()`; the book just calls it "averaged and synchronized".

### Ideal speed-up (book pp. 283–284)
$$T_W \approx \frac{T_1}{W}$$
- $T_1$ = time for one epoch on a single GPU
- $T_W$ = time for one epoch on $W$ GPUs
- $\approx$ = "roughly", because GPUs spend a little time communicating

So two GPUs could in theory finish an epoch in half the time, and eight GPUs
eight times faster.

### The code, piece by piece
**Listing A.12: the utilities** (book p. 284)
```python
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
```
- `mp.spawn` launches one training process per GPU.
- `DistributedSampler` divides the dataset among the processes.
- `init_process_group` / `destroy_process_group` set up and tear down the communication group.
- `DDP(...)` wraps the model so gradients get synced.

**Listing A.13: training with DDP** (book pp. 284–285, condensed)
```python
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"   # address of the main node
    os.environ["MASTER_PORT"] = "12345"       # any free port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def prepare_dataset():
    # insert dataset preparation code
    train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=False,
                              pin_memory=True, drop_last=True,
                              sampler=DistributedSampler(train_ds))
    return train_loader, test_loader

def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            # insert model prediction and backpropagation code
            ...
    model.eval()
    ...  # compute_accuracy on train/test, printed per GPU
    destroy_process_group()

if __name__ == "__main__":
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
```
What the key bits mean (book pp. 285–286):
- **`rank`** = the GPU / process ID. `mp.spawn` passes it in automatically, which is why it isn't listed in `args`.
- **`world_size`** = how many GPUs, here `torch.cuda.device_count()`.
- **`backend="nccl"`** = NVIDIA Collective Communication Library, built for GPU-to-GPU communication.
- **`shuffle=False`** because `DistributedSampler` now handles shuffling; **`pin_memory=True`** speeds up memory transfer to the GPU.
- **`set_epoch(epoch)`** is called at the start of each epoch. The book doesn't say why; in PyTorch it reseeds the sampler so the shuffle changes from epoch to epoch.
- **`.to(rank)`** moves model and data to "GPU number `rank`".
- **`DDP(model, device_ids=[rank])`** is the line that turns on gradient synchronization.

### What the output looks like (book p. 287)
The toy model is the 2-input, 2-output network from Appendix A
(Linear 2→30 → ReLU → Linear 30→20 → ReLU → Linear 20→2, 752 parameters), on the
5-example toy dataset, batch size 2, SGD with `lr=0.5`, 3 epochs.

| Setup | Epoch 1 | Epoch 2 | Epoch 3 | Train / test acc. |
|---|---|---|---|---|
| 1 GPU (`ch02-DDP-script.py`) | 0.62, 0.32 | 0.11, 0.07 | 0.02, 0.03 | 1.0 / 1.0 |
| 2 GPUs | GPU1 0.60, GPU0 0.59 | GPU0 0.16, GPU1 0.17 | GPU0 0.05, GPU1 0.05 | 1.0 / 1.0 on each GPU |

The script (`python ch02-DDP-script.py`, run from the terminal) works on both
single- and multi-GPU machines and first prints `Number of GPUs available`. The
book notes the 1-GPU output looks similar to the plain single-GPU run of §A.9.2,
"which is a good sanity check". With 2 GPUs, some batches are processed on GPU0
and others on GPU1, and the accuracy lines appear **twice**, because each process runs
independently and prints on its own. The fix is to print only from the first
process: `if rank == 0: print("Test accuracy: ", accuracy)`.

### Practical gotchas (book pp. 284, 286)
- **No Jupyter.** DDP needs to spawn several processes, each with its own Python interpreter, and notebooks handle multiprocessing differently. Run it as a script.
- **Choosing GPUs.** `CUDA_VISIBLE_DEVICES=0 python some_script.py` uses only GPU 0; `CUDA_VISIBLE_DEVICES=0,2 python some_script.py` uses the first and third of four GPUs, without editing the script.

### Beyond DDP: FSDP and friends (book pp. 288, 299)
DDP needs the **whole model to fit on each GPU**, because every GPU holds a full
copy. When the model is too big for that, the book points to PyTorch's
**Fully Sharded Data Parallel (FSDP)**, which "performs distributed data
parallelism and distributes large layers across different GPUs" (book p. 299;
see "Introducing PyTorch Fully Sharded Data Parallel (FSDP) API",
https://mng.bz/EZJR). For an easier on-ramp, the book mentions the open-source
**Fabric** library and the author's article "Accelerating PyTorch Model Training:
Using Mixed-Precision and Fully Sharded Data Parallelism" (https://mng.bz/jXle)
(book p. 288). For more DDP detail the book points to the official API documentation
(https://mng.bz/9dPr, book p. 288). The Appendix A summary calls DDP "the simplest way in PyTorch to
accelerate training when several GPUs are available" (book p. 288).

## 🔴 Theory (Expert)
- **DDP = large-batch SGD.** If each of the $W$ GPUs uses per-example-mean loss on
  a micro-batch of size $B$, then by linearity of the gradient,
  $\frac{1}{W}\sum_r \nabla\mathcal{L}_r = \nabla\big(\frac{1}{WB}\sum_{i=1}^{WB}\ell_i\big)$,
  the gradient of the mean loss on one batch of size $WB$. So DDP with $W$ GPUs is
  mathematically the same optimizer as single-GPU SGD with a $W$-times-larger
  batch (when all shards have equal size and the model has no cross-example
  layers like [[batch-normalization]]). This is a math annotation; the book only
  says the gradients are averaged. It's the same identity that underlies
  [[gradient-accumulation]], which averages over *time* instead of *devices*.
- **Consequences of the larger batch.** Averaging $W$ independent shard gradients
  divides the gradient-noise variance by roughly $W$
  ([[monte-carlo-estimation]]). In practice people often raise the learning rate
  as $W$ grows (e.g. linear scaling) and add [[learning-rate-warmup]] to keep early
  training stable; none of this is in the book.
- **Why replicas stay identical.** Identical initial weights (the book uses a
  fixed seed; PyTorch's `DDP` wrapper also broadcasts rank 0's weights to every
  replica when it's created) + identical averaged gradient + deterministic
  optimizer ⇒ identical weights after every step, by induction. Anything that
  breaks one of these (e.g. different optimizer state) would make replicas
  diverge.
- **All-reduce cost.** A ring all-reduce moves about $2\frac{W-1}{W}\,P$ values
  per GPU per step for $P$ parameters, which is nearly independent of $W$; that's
  why the communication overhead stays "small" and speed-up is close to linear.
  DDP also overlaps communication with the backward pass by reducing gradients in
  buckets as soon as they're ready.
- **Memory picture and FSDP.** Under DDP each GPU stores the full parameters,
  gradients and optimizer state: for [[adamw-optimizer]] that's about
  $4P$ numbers (weights, grads, two moments), which for GPT-2 124M is about
  $4 \times 124\text{M} \times 4$ bytes $\approx 2$ GB in FP32 before activations
  ([[parameter-count-and-memory]], [[floating-point-precision]]). FSDP
  *shards* these across GPUs, gathering a layer's full weights only while that
  layer runs, so per-GPU memory falls roughly as $1/W$ at the cost of extra
  communication.

## Connections
- Builds on: [[stochastic-gradient-descent]], [[gradient]], [[backpropagation]], [[mini-batch]], [[gpu-computing]], [[pytorch]], [[pytorch-dataset-and-dataloader]], [[expected-value-and-mean]]
- Related: [[gradient-accumulation]], [[learning-rate]], [[parameter-count-and-memory]], [[random-seed-reproducibility]], [[train-vs-eval-mode]]
- Leads to: [[pretraining]] at scale, [[llm-training-loop]]
- Used by: training [[large-language-model]]s such as [[gpt-2]] and [[gpt-3]] across many GPUs
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.9.3 "Training with multiple GPUs" (Figs. A.12–A.13, Listings A.12–A.13, outputs, sidebars "Selecting available GPUs on a multi-GPU machine" and "Alternative PyTorch APIs for multi-GPU training"), book pp. 282–288 (PDF pp. 304–310).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix A Summary, book p. 288 (PDF p. 310).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, "Appendix A" further reading (FSDP, https://mng.bz/EZJR), book p. 299 (PDF p. 321).
- The averaging formula is the book's verbal description written in symbols. The large-batch equivalence, all-reduce cost, linear-scaling and memory figures in Theory are standard results added for context, not book claims.
