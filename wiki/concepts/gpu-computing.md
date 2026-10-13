---
title: GPU Computing and PyTorch Devices
type: concept
tags: [pytorch, hardware, gpu, cuda, training, deep-learning]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# GPU Computing and PyTorch Devices

> **One-liner:** A GPU is a chip that can do thousands of small multiplications at once, and in PyTorch you just tell each tensor (and the model) which chip — "device" — it should live on.

## 🟢 Intuition (Dummy)
Picture a CPU as a handful of brilliant professors: each one can solve any problem,
but there are only a few of them. A GPU is like a stadium full of school kids who can
each only do simple arithmetic — but there are thousands of them, all working at the
same time. Training a neural network is mostly *simple arithmetic repeated billions of
times* (multiply, add, multiply, add…), so the stadium wins by a mile.

There's one catch: the kids and the professors sit in different buildings. Before the
kids can work on your numbers, you have to *carry the numbers over* (a memory
transfer). For a tiny job, the walk takes longer than the work itself, so the CPU is
just as fast. For a big job — like an LLM — the GPU is dramatically faster.

PyTorch hides almost all of this. You pick a **device** ("cpu", "cuda" for NVIDIA GPUs,
or "mps" for Apple Silicon Macs), move your model and data there with `.to(device)`, and
everything else in your code stays the same. The book says it takes only **three changed
lines** to move a training loop from CPU to GPU.

## 🟡 Mechanics (Practitioner)

### The core rules
1. **A device is where a tensor lives and where its math runs.** CPU and GPU are both
   devices. By default, tensors are created on the CPU (book pp. 279–280).
2. **Check for a GPU:** `torch.cuda.is_available()` returns `True` if PyTorch can see a
   CUDA GPU (book pp. 256, 279). On Apple Silicon (M1/M2/M3 or newer) use
   `torch.backends.mps.is_available()` (book pp. 257, 282).
3. **Move things with `.to()`.** The same `.to()` method that changes a tensor's dtype
   also moves it: `.to("cuda")`, `.to("cpu")` (book p. 280).
4. **All tensors in one operation must be on the same device.** Otherwise you get an error.

### The book's tensor example (book pp. 279–280)
The math is plain element-wise vector addition:

$$\mathbf{a}+\mathbf{b} = (a_1+b_1,\ a_2+b_2,\ a_3+b_3)$$

Reading it symbol by symbol: $\mathbf{a}$ = `tensor_1`, $\mathbf{b}$ = `tensor_2`; the bold
letters mean "a whole [[vector]]"; $a_1, a_2, a_3$ are its three entries; the $+$ inside
the parentheses happens **entry by entry**. With the book's numbers:
$(1,2,3)+(4,5,6)=(5,7,9)$.

```python
tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
print(tensor_1 + tensor_2)        # tensor([5., 7., 9.])                  -> CPU

tensor_1 = tensor_1.to("cuda")
tensor_2 = tensor_2.to("cuda")
print(tensor_1 + tensor_2)        # tensor([5., 7., 9.], device='cuda:0') -> GPU

tensor_1 = tensor_1.to("cpu")
print(tensor_1 + tensor_2)
# RuntimeError: Expected all tensors to be on the same device,
# but found at least two devices, cuda:0 and cpu!
```

`device='cuda:0'` means "the first GPU". With several GPUs you pick one by ID:
`"cuda:0"`, `"cuda:1"`, … (book p. 280).

### Single-GPU training: the three changed lines (Listing A.11, book p. 281)
Starting from the CPU training loop of [[training-loop|§A.7]]:

```python
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)

device = torch.device("cuda")          # (1) choose the device
model.to(device)                       # (2) move the model's parameters

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)   # (3) move each batch
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
```

The loss log is the same as on the CPU: 0.75, 0.65, 0.44, 0.13, 0.03, 0.00 (book p. 281).
The update the optimizer runs is ordinary [[stochastic-gradient-descent]]:

$$\theta \leftarrow \theta - \eta\,\nabla_\theta L(\theta)$$

Symbol by symbol: $\theta$ (theta) = all the model's weights and biases; $\leftarrow$ = "is
replaced by"; $\eta$ (eta) = the [[learning-rate]] (`lr=0.5` here); $\nabla_\theta L$ =
the [[gradient]] of the loss $L$ with respect to $\theta$ (what `loss.backward()` computes).
The GPU doesn't change *what* is computed — only *where*.

**Best practice — device-agnostic code** (book p. 281):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Apple Silicon (book p. 282):
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

**Don't expect a speed-up for tiny models.** For the 2-input toy network you probably
won't see one, because of the cost of copying memory from CPU to GPU. The big wins come
with deep networks, "especially LLMs" (book p. 281).

### How fast? Exercise A.4 (book pp. 282, 311–312)
The exercise asks you to time [[matrix-multiplication]] on CPU vs GPU. The solution in
Appendix C uses a $100\times200$ matrix times a $200\times300$ matrix:

```python
a = torch.rand(100, 200)
b = torch.rand(200, 300)
%timeit a @ b                    # CPU:  63.8 µs ± 8.7 µs per loop
a, b = a.to("cuda"), b.to("cuda")
%timeit a @ b                    # V100: 13.8 µs ± 425 ns per loop
```

The book calls this "approximately four times faster" on a Google Colab V100 (book p. 312;
the exact ratio is $63.8/13.8 \approx 4.6$). The author reports "significant speed-ups even for
small matrix multiplications" like this one, and stresses that results depend on the hardware
(book p. 311).

### In LLM fine-tuning (chapter 7)
- **Device choice** (book p. 224): `"cpu"`, `"cuda"` (NVIDIA), or optionally `"mps"`. The
  book warns that `"mps"` may give **small numerical differences** from the printed
  results, because Apple Silicon support in PyTorch was still experimental.
- **Move data inside the collate function.** In chapter 7 the
  [[collate-function]] itself calls `.to(device)` (e.g. `torch.stack(inputs_lst).to(device)`).
  The book's reason: the device transfer can then run as a background process outside the
  training loop, so it does not block the GPU during training (book p. 224). The device is
  pre-filled with `functools.partial(custom_collate_fn, device=device, allowed_max_length=1024)`.
- **Keep the GPU fed.** With `num_workers=0`, data loading happens in the main process
  and the **GPU can sit idle** waiting for the next batch; with `num_workers > 0`, worker
  processes queue up batches in the background (book pp. 273–274, Fig. A.11). The author
  finds `num_workers=4` often works well, but warns about Jupyter problems.
- **Timing examples.** LoRA fine-tuning of GPT-2 small: about 15 minutes on an M3 MacBook
  Air vs. less than half a minute on a V100 or A100 (book p. 334).

### Hardware and install notes (book pp. 254–257)
- Two PyTorch builds: CPU-only and CPU+GPU. If you have a CUDA GPU (ideally NVIDIA **T4,
  RTX 2080 Ti, or newer**), install the GPU build; `pip install torch` picks it
  automatically on a CUDA machine. The book uses **PyTorch 2.4.0**.
- Experimental AMD support via **ROCm** (as of writing).
- No GPU? Rent one, or use **Google Colab** (Runtime → Change runtime type → GPU).

## 🔴 Theory (Expert)
- **Why GPUs fit deep learning.** Almost all the FLOPs in a transformer are dense
  [[matrix-multiplication|matrix multiplications]]: $C = AB$ with
  $C_{ij} = \sum_k A_{ik}B_{kj}$. Every one of the $m\cdot p$ output entries is an
  independent dot product, so the work is *embarrassingly parallel*. For $n\times n$
  matrices the cost is $O(n^3)$ arithmetic on $O(n^2)$ data — high arithmetic intensity,
  which is exactly what thousands of GPU cores (and tensor cores) are built for.
  (Background, not stated in the book.)
- **The crossover point (Exercise A.4).** Total GPU time ≈ transfer + kernel launch +
  compute. Transfer is $O(n^2)$ and launch is a fixed overhead; compute is $O(n^3)$ but
  divided across many cores. For small enough $n$ the overheads dominate, so the CPU wins or
  ties; as $n$ grows, the cubic compute term dominates and the GPU pulls ahead. This explains
  the book's "no speed-up for this small example" remark about the toy training loop (book
  p. 281). Note: the crossover can be very low. In the book's Exercise A.4 solution, a
  $100\times200$ by $200\times300$ product (tensors already on the GPU, so `%timeit` measures
  no transfer) was already ~4× faster on a V100 (book pp. 311–312). (Cost model is background,
  not stated in the book.)
- **float32 by default.** PyTorch's default float type is 32-bit partly because GPU
  architectures are optimized for 32-bit computation (book p. 259). Lower precisions
  (bfloat16, mixed precision) push throughput further — see [[floating-point-precision]];
  the book points to the author's article on mixed precision and FSDP (book p. 288).
- **Same-device rule as a type system.** You can think of the device as part of a tensor's
  type: `+` is only defined between tensors of the same device type, which is why PyTorch
  raises instead of silently copying (silent copies would hide expensive transfers).
- **Numerical non-determinism across devices.** Floating-point addition is not
  associative: $(a+b)+c \ne a+(b+c)$ in general. GPUs reduce sums in a different order
  than CPUs (and MPS differently again), so results can differ in the last digits — the
  root cause of the book's MPS warning (book p. 224). See [[random-seed-reproducibility]]
  and [[numerical-stability]].
- **Many GPUs.** When one GPU is not enough, [[distributed-data-parallel]] (DDP) runs one
  process and one model replica per GPU, gives each a different minibatch, and *averages
  the gradients* so all replicas stay identical:
  $g = \frac{1}{W}\sum_{r=0}^{W-1}\nabla_\theta L_r(\theta)$, where $W$ = number of GPUs
  (`world_size`) and $r$ = the GPU's index (`rank`). Ideal speed-up ≈ $W\times$ minus
  communication overhead — "half the time with 2 GPUs" (book pp. 282–284). Restrict GPUs with
  `CUDA_VISIBLE_DEVICES=0,2 python script.py` (book p. 286).

## Connections
- Builds on: [[tensor]], [[pytorch]], [[vector]], [[matrix-multiplication]]
- Leads to: [[distributed-data-parallel]], [[floating-point-precision]], [[gradient-accumulation]]
- Used by: [[training-loop]], [[llm-training-loop]], [[pretraining]], [[instruction-fine-tuning]], [[low-rank-adaptation]], [[pytorch-dataset-and-dataloader]], [[collate-function]]
- Related: [[stochastic-gradient-descent]], [[random-seed-reproducibility]], [[numerical-stability]], [[parameter-count-and-memory]]
- People: [[sebastian-raschka]]

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** The cost sidebar in §5.1.3 puts LLM pretraining on GPUs into numbers. Llama 2 7B needed **184,320 A100 GPU-hours** for 2 trillion tokens. With an 8 × A100 AWS server at about \$30 per hour, the cost is $184{,}320 / 8 \times 30 \approx \$690{,}000$ (book p. 141). The same section shows the one-line device pattern `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, then `model.to(device)` and `input_batch.to(device)` inside `calc_loss_batch`. With this, the code runs on a CUDA GPU with no other change (book pp. 144–145). The 10-epoch training run in §5.2 takes about 5 minutes on a MacBook Air CPU (book p. 149).

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1.3 (installing PyTorch, CUDA/ROCm/MPS, Colab), book pp. 254–257
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2.2 (float32 and GPU optimization), book p. 259
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.6 (`num_workers`, GPU idle time, Fig. A.11), book pp. 273–274
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.9.1–A.9.3 (devices, Listing A.11, DDP), book pp. 279–288
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise A.4 solution (CPU vs GPU matmul timing), book pp. 311–312
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.4 (device setting in the collate function, MPS note), book p. 224
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4 (training time M3 vs V100/A100), book p. 334
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3, book pp. 141, 144–145 (sidebar "The cost of pretraining LLMs"; device selection in Listing 5.2 usage); §5.2, book p. 149 (5-minute laptop run).
