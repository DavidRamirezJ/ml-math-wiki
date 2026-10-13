---
title: Random Seed and Reproducibility
type: concept
tags: [pytorch, randomness, reproducibility, training, data-loading]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-appD-training-loop-bells-and-whistles, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Random Seed and Reproducibility

> **One-liner:** A random seed is the starting number for the computer's "fake dice", so if you use the same seed you get exactly the same "random" results every time.

## 🟢 Intuition (Dummy)
Computers can't really roll dice. Instead they run a recipe that turns one starting
number into a long list of numbers that *look* random. That starting number is the
**seed**. Same seed means the same list, every single time — a bit like shuffling a deck
of cards with a robot that always shuffles the same way if you press the same button.

Why should you care? Training a neural network uses randomness all over the place:
the starting weights are random, and the order you show the examples in is random. If
you and I both run the book's code, we'd get different numbers — unless we both press
the same button first. That button in PyTorch is `torch.manual_seed(123)`. With it,
your printouts match the book's exactly, bugs become repeatable, and a comparison like
"LoRA vs. full fine-tuning" is fair because both start from the same dice rolls.

Randomness itself is *good* — we just want it to be **controllable**.

## 🟡 Mechanics (Practitioner)

### Where the book uses randomness
| What is random | Why it's random | Where in the book |
|---|---|---|
| Initial weights of `nn.Linear` layers | to **break symmetry** | §A.5, book p. 268 |
| Toy input `X = torch.rand((1, 50))` | demo data | §A.5, book p. 268 |
| Order of training examples (`shuffle=True`) | avoid repetitive update cycles | §A.6, book p. 272 |
| Train/val/test split (`random_split`, reused from ch. 6) | unbiased split; seeded with pandas `df.sample(frac=1, random_state=123)` (Listing 6.3, book p. 175) | §E.2, book p. 324 |
| Mini-batches drawn for evaluation / training | stochastic gradients | §7.4, §E.2, §E.4 |

### Seeding the weights (book p. 268)
Without a seed, every `NeuralNetwork(50, 3)` gets different small random weights, so your
printed numbers won't match the book's. With a seed (the book prints the first layer's
$30\times50$ weight matrix, abbreviated; middle rows omitted here):

```python
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)
# Parameter containing:
# tensor([[-0.0577,  0.0047, -0.0702,  ...,  0.0222,  0.1260,  0.0865],
#         [ 0.0502,  0.0307,  0.0333,  ...,  0.0951,  0.1134, -0.0297],
#         ...
#         [ 0.0790,  0.1343, -0.0293,  ...,  0.0344, -0.0971, -0.0509]],
#        requires_grad=True)
```

And seeding the random input:

```python
torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
# tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)
```

**Why random at all? Symmetry breaking** (book p. 268). If every weight started at the
same value, every node in a layer would compute the same thing and get the same update in
[[backpropagation]], forever. The network could never learn different features. Random
starting values make the nodes different from step one. (See [[weight-initialization]].)

### Seeding the shuffle (Listing A.7, book p. 272)
```python
torch.manual_seed(123)
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0)
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
# Batch 1: tensor([[-1.2000,  3.1000], [-0.5000,  2.6000]]) tensor([0, 0])
# Batch 2: tensor([[ 2.3000, -1.1000], [-0.9000,  2.9000]]) tensor([1, 0])
# Batch 3: tensor([[ 2.7000, -1.5000]])                      tensor([1])
```

Two facts from the book:
- With the seed, you get "the exact same shuffling order" as the book.
- If you loop over the loader **a second time**, the order is **different** — and that's on
  purpose: it keeps the network from "getting caught in repetitive update cycles" during
  training (book p. 272).

You can see this in the book itself: Listing A.8 (book p. 273) builds a new loader with
`drop_last=True` **without** re-seeding, and its batches come out in a different order
(`[-0.9, 2.9], [2.3, -1.1]` first) — the generator has moved on.

So the seed fixes the *whole sequence* of shuffles, not one frozen order. Each [[epoch]]
gets a new permutation, but run-to-run the sequence of permutations is identical.

In maths terms, shuffling picks a random **permutation** $\pi$ of the indices:

$$\pi : \{1,\dots,N\} \to \{1,\dots,N\}\ \text{(one-to-one)}$$

Symbol by symbol: $\pi$ (pi) = the reordering rule; $\{1,\dots,N\}$ = the example numbers
(here $N = 5$); "$\to$" = "maps to"; one-to-one = every example appears exactly once. There
are $N! = 5! = 120$ possible orders; the seed decides which one you get each epoch.

### Seeds in the LLM chapters
- **Instruction fine-tuning data loaders** (Listing 7.6, book p. 225): `torch.manual_seed(123)`
  before building `train_loader` (`batch_size=8`, `shuffle=True`, `drop_last=True`);
  validation and test loaders use `shuffle=False` so evaluation order is fixed.
- **Baseline generation before fine-tuning** (§7.5, book p. 228): `torch.manual_seed(123)`
  before formatting `val_data[0]` and calling `generate(..., max_new_tokens=35, eos_id=50256)`.
  (The book does not explain this seed. Note: without temperature or top-k, `generate` is
  greedy [[argmax]] decoding, which is deterministic anyway; the seed matters once you
  switch on [[multinomial-sampling]] / [[top-k-sampling]].)
- **LoRA spam classifier** (Appendix E): `torch.manual_seed(123)` appears before
  - the data loaders (Listing E.3, book p. 325) — reproducible training-loader shuffling;
  - the new random classification head `model.out_head = torch.nn.Linear(768, 2)` (§E.3,
    book p. 327) — reproducible head initialization;
  - each `calc_accuracy_loader(..., num_batches=10)` call, giving the same baseline accuracies
    twice — **46.25% / 45.00% / 48.75%** (train/val/test) — before and after inserting LoRA
    layers (book pp. 328, 333–334). The book's explanation: $B$ is initialized to zeros, so
    $AB$ is the zero matrix and does not change the weights (book p. 334). Because the seeded
    numbers match exactly, you can trust that adding LoRA with $B = 0$ really was a no-op;
  - the LoRA training run (Listing E.7, book p. 334).
- **Multi-GPU** (Listing A.13, book p. 285): `torch.manual_seed(123)` in `__main__`, and
  `train_loader.sampler.set_epoch(epoch)` each epoch so the `DistributedSampler` reshuffles
  differently per epoch (the reason is standard PyTorch behaviour, not stated in the book).

### Recipe
```python
torch.manual_seed(123)   # before creating the model  -> same initial weights
torch.manual_seed(123)   # before creating the loader -> same shuffle sequence
```
Seed **right before** the random thing you want to pin down; any extra random call in
between shifts everything after it.

## 🔴 Theory (Expert)
- **Pseudo-random number generators (PRNGs).** A PRNG is a deterministic state machine
  $s_{t+1} = T(s_t)$, $u_t = O(s_t)$, where $s_0$ is set by the seed, $T$ updates the
  state and $O$ outputs a number that is (statistically) uniform on $[0,1)$. PyTorch's CPU
  generator is a Mersenne-Twister–style generator; GPU generators use counter-based
  Philox. Same $s_0$ ⇒ same stream $u_0, u_1, \dots$. Every "random" PyTorch op
  (`rand`, `randperm`, init, dropout masks, `multinomial`) consumes part of this stream, so
  results are reproducible **only if the same ops run in the same order**. (Background, not
  in the book.)
- **Shuffling = sampling a uniform permutation.** A correct shuffle (Fisher–Yates /
  `randperm`) makes every one of the $N!$ orders equally likely. Reshuffling each epoch turns
  mini-batch gradients into (approximately) unbiased estimates of the full-batch gradient —
  the formal reason behind the book's "avoid repetitive update cycles" (book p. 272). See
  [[stochastic-gradient-descent]], [[mini-batch]].
- **Symmetry breaking, formally.** If all weights of a layer start equal, the gradients of
  the hidden units are equal by symmetry, so the update preserves equality; the set of
  "all-equal" weights is invariant under gradient descent. Random initialization starts the
  dynamics off that invariant set (book p. 268). See [[weight-initialization]].
- **Seeds are not the whole story.** Bit-exact reproducibility also needs: same library
  versions, same hardware, deterministic kernels. Floating-point addition is not associative,
  so GPU reductions in a different order can change low-order bits; the book warns that the
  `"mps"` device can give numerical differences from its printouts (book p. 224). See
  [[gpu-computing]] and [[floating-point-precision]].
- **Seeds and science.** One seed is one sample from the training distribution. A difference
  like "LoRA 1.30 min vs full 1.80 min" or "~4 points worse with instruction masking"
  (Appendix C) is a single-seed result; a careful comparison averages over several seeds and
  reports the spread — exactly the [[monte-carlo-estimation]] idea applied to experiments.

> **From [[llm-ch04-implementing-a-gpt-model]]:** every Ch. 4 demo calls `torch.manual_seed(123)` before it builds a layer or model (book pp. 100, 111, 116, 120). That is why your LayerNorm means (0.1324, 0.2170), shortcut gradients (0.0002 vs 0.22) and untrained output ("Featureiman Byeswickattribute argue") can match the book exactly.

## Connections
- Builds on: [[random-variable]], [[probability-distribution]], [[pytorch]]
- Leads to: [[weight-initialization]], [[pytorch-dataset-and-dataloader]], [[stochastic-gradient-descent]], [[multinomial-sampling]]
- Used by: [[training-loop]], [[instruction-fine-tuning]], [[classification-fine-tuning]], [[low-rank-adaptation]], [[baseline-evaluation]], [[distributed-data-parallel]], [[train-validation-test-split]]
- Related: [[gpu-computing]], [[floating-point-precision]], [[dropout]], [[temperature-scaling]], [[epoch]]
- People: [[sebastian-raschka]]

> **From [[llm-appD-training-loop-bells-and-whistles]]:** Appendix D calls `torch.manual_seed(123)` before it builds the 124M GPT model and the data loaders (book pp. 313–314), again before the gradient-clipping demo (book p. 318), and before the final `train_model` run (book p. 320). The same seed makes the warmup/decay/clipping run comparable with Ch. 5's `train_model_simple` run.

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** Ch. 5 sets `torch.manual_seed(123)` before it builds the untrained model in §5.1.1 (book p. 130), before it creates the data loaders in §5.1.3 (book p. 143), and before training in §5.2 (book p. 149). The seed also matters for *sampling*. In §5.3.1 the seed is set before `torch.multinomial`, so the 1,000-sample counts (582 forward, 343 toward, 73 closer, 2 inches) can be reproduced (book pp. 153–154). §5.3.3 sets it again before `generate` (book p. 158). Without a seed, every sampled text would be different.

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (random init, symmetry breaking, `torch.manual_seed(123)`), book p. 268
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.6 (seeded shuffling, Listing A.7, reshuffle per epoch), book pp. 272–273
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.9.3 (seed and `set_epoch` in DDP, Listing A.13), book p. 285
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.4 (Listing 7.6, seeded loaders; MPS numerical differences), book pp. 224–225
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5 (seed before baseline generation), book p. 228
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3 (Listing 6.3, `random_split` with `random_state=123`), book p. 175
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.2 (Listing E.3, seeded loaders), book p. 325
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.3–E.4 (seeded head init, seeded accuracy baselines 46.25/45.00/48.75%, Listing E.7), book pp. 327–328, 333–334
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D intro, book pp. 313–314; §D.3, book p. 318; §D.4, book p. 320 (seeded runs).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book p. 100; §4.4, book p. 111; §4.5, book p. 116; §4.6, book p. 120 (`torch.manual_seed(123)` before each randomly initialized demo so printed tensors and gradients match the book).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.1, book p. 130; §5.1.3, book p. 143; §5.2, book p. 149; §5.3.1, book pp. 153–154; §5.3.3, book p. 158 (`torch.manual_seed(123)`).
