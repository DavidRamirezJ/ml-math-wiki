---
title: "LLM App. A — Introduction to PyTorch"
type: source
tags: [llm, pytorch, deep-learning, autograd, training, gpu, source]
created: 2026-10-13
updated: 2026-10-13
---

# LLM App. A — Introduction to PyTorch

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix A, **book pp. 251–288** (PDF pages 273–310)
- **Type:** book appendix — *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025)
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13 (every listing, printed output and number checked line by line against the page text; Figures A.7, A.8, A.9 and book pp. 274, 279 checked from rendered or re-extracted PDF pages)
- *Page rule:* PDF page = book page + 22.
- **Related:** Appendix B, book pp. 298–299 (7 references for Appendix A). Appendix C, book pp. 310–312 (solutions to Exercises A.1–A.4).

This appendix is the PyTorch "toolbox" chapter of the book. You do not need it to understand
LLMs, but you need it to *build* one. It starts with what [[pytorch]] is and where
[[deep-learning]] sits inside [[machine-learning]] and [[artificial-intelligence]]. Then it
teaches [[tensor|tensors]], [[computational-graph|computation graphs]] and
[[automatic-differentiation]], and it builds a small [[multilayer-perceptron]]. It finishes
with data loaders, a full [[training-loop]], saving models, and training on one or many GPUs.
The same pieces come back, at a much bigger size, in the GPT model of Ch. 4–7.

## Section coverage (all of App. A accounted for)
| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Appendix intro (PyTorch as the main tool; tensors, autograd, backprop) | 251 | ✅ [[pytorch]], [[tensor]], [[automatic-differentiation]], [[backpropagation]] |
| A.1 | What is PyTorch? (Papers With Code, Kaggle 2022 ~40%) | 251–252 | ✅ [[pytorch]] |
| A.1.1 | The three core components of PyTorch (Fig. A.1) | 252 | ✅ [[pytorch]], [[tensor]], [[automatic-differentiation]], [[gradient]], [[loss-function]] |
| A.1.2 | Defining deep learning (Figs. A.2–A.3) | 252–254 | ✅ [[artificial-intelligence]], [[machine-learning]], [[deep-learning]], [[neural-network]], [[supervised-learning]], [[text-classification]], [[next-token-prediction]], [[pretraining]], [[self-supervised-learning]], [[empirical-risk-minimization]] |
| A.1.3 | Installing PyTorch (Figs. A.4–A.5; CUDA, ROCm, Colab, Apple Silicon MPS) | 254–257 | ✅ [[pytorch]], [[gpu-computing]] |
| A.2 | Understanding tensors (Fig. A.6; tensor rank) | 258 | ✅ [[tensor]], [[vector]], [[matrix]], [[basis-and-rank]] (rank ≠ tensor order) |
| A.2.1 | Scalars, vectors, matrices, and tensors (Listing A.1) | 258–259 | ✅ [[tensor]], [[vector]], [[matrix]] |
| A.2.2 | Tensor data types (int64, float32, `.to`) | 259–260 | ✅ [[floating-point-precision]], [[tensor]] |
| A.2.3 | Common PyTorch tensor operations (`.shape`, `.reshape`, `.view`, `.T`, `.matmul`, `@`) | 260–261 | ✅ [[tensor-reshaping-and-transposing]], [[matrix-transpose]], [[matrix-multiplication]], [[gram-matrix]], [[inner-product]] |
| A.3 | Seeing models as computation graphs (Listing A.2, Fig. A.7) | 261–262 | ✅ [[computational-graph]], [[forward-pass]], [[logistic-regression]], [[sigmoid-function]], [[binary-cross-entropy]], [[model-parameters]], [[bias-vector]] |
| A.4 | Automatic differentiation made easy (Fig. A.8, Listing A.3, sidebar "Partial derivatives and gradients") | 263–265 | ✅ [[automatic-differentiation]], [[backpropagation]], [[chain-rule]], [[partial-derivative]], [[derivative]], [[gradient]], [[gradient-descent]], [[computational-graph]] |
| A.5 | Implementing multilayer neural networks (Fig. A.9, Listing A.4) | 265–269 | ✅ [[multilayer-perceptron]], [[pytorch-nn-module]], [[linear-layer]], [[relu]], [[activation-function]], [[logits]], [[softmax]], [[weight-initialization]], [[random-seed-reproducibility]], [[forward-pass]], [[parameter-count-and-memory]], [[numerical-stability]] |
| A.6 | Setting up efficient data loaders (Figs. A.10–A.11, Listings A.5–A.8) | 270–274 | ✅ [[pytorch-dataset-and-dataloader]], [[mini-batch]], [[epoch]], [[random-seed-reproducibility]], [[train-validation-test-split]] |
| A.7 | A typical training loop (Listings A.9–A.10, Exercise A.3) | 274–278 | ✅ [[training-loop]], [[stochastic-gradient-descent]], [[learning-rate]], [[hyperparameter]], [[cross-entropy-loss]], [[train-vs-eval-mode]], [[gradient-accumulation]], [[argmax]], [[classification-accuracy]], [[train-validation-test-split]], [[dropout]], [[batch-normalization]] |
| A.8 | Saving and loading models (`state_dict`) | 278–279 | ✅ [[model-checkpointing]], [[model-parameters]] |
| A.9 | Optimizing training performance with GPUs (intro) | 279 | ✅ [[gpu-computing]] |
| A.9.1 | PyTorch computations on GPU devices | 279–280 | ✅ [[gpu-computing]], [[tensor]] |
| A.9.2 | Single-GPU training (Listing A.11, sidebar on macOS, Exercise A.4) | 280–282 | ✅ [[gpu-computing]], [[training-loop]], [[matrix-multiplication]] |
| A.9.3 | Training with multiple GPUs (Figs. A.12–A.13, Listings A.12–A.13, sidebars) | 282–288 | ✅ [[distributed-data-parallel]], [[gradient]], [[expected-value-and-mean]], [[mini-batch]] |
| — | Summary | 288 | ✅ (all of the above) |
| App. B | References for Appendix A | 298–299 | ✅ see "References cited" below |
| App. C | Solutions to Exercises A.1–A.4 | 310–312 | ✅ see "Exercises" below |

> ✅ **Every section and subsection of Appendix A maps to at least one wiki page.** Only A.1,
> A.2 and A.9 have numbered subsections. A.3–A.8 are single sections with no A.x.y headings.
>
> **Coverage audit (2026-10-13):** all linked pages exist, and all 32 concept pages owned by this
> unit cite `llm-appA-introduction-to-pytorch` and have all three tiers.
>
> **Final audit (2026-10-13):** five linked pages owned by other units had no App. A content:
> [[basis-and-rank]], [[gram-matrix]], [[expected-value-and-mean]], [[linear-mapping]] and
> [[text-classification]]. Each one now has a short "From llm-appA-introduction-to-pytorch"
> note with page citations. The notes cover tensor rank vs matrix rank, the $\mathbf X\mathbf X^\top$ example,
> DDP gradient averaging, `nn.Linear` as an affine map, and the spam-classifier workflow.
> 74 concept and person pages now cite this source.

## TL;DR (5 bullets)
- **PyTorch has three parts (A.1):** a [[tensor]] library (NumPy-like arrays that can run on a GPU), an [[automatic-differentiation]] engine ("autograd"), and a deep-learning library (layers, losses, optimisers). [[deep-learning]] is [[machine-learning]] with [[neural-network|neural networks]] that have many layers, and it sits inside [[artificial-intelligence]].
- **Tensors generalise numbers, vectors and matrices (A.2):** a scalar is rank 0, a [[vector]] is rank 1, a [[matrix]] is rank 2. The default float type is 32-bit ([[floating-point-precision]]). Transpose is *not* the same as reshape, and `@` does [[matrix-multiplication]].
- **A model is a computation graph, and autograd walks it backwards (A.3–A.4):** for a one-input [[logistic-regression]], PyTorch records $u = w_1x_1$, $z = u+b$, $a = \sigma(z)$, $L(a,y)$, then applies the [[chain-rule]] from right to left ([[backpropagation]]) to get $\partial L/\partial w_1 = -0.0898$ and $\partial L/\partial b = -0.0817$.
- **Networks are `nn.Module` classes (A.5–A.6):** a [[multilayer-perceptron]] is a stack of [[linear-layer|linear layers]] with [[relu|ReLU]] between them, and it returns [[logits]]. A `Dataset` says how to load *one* example; a `DataLoader` shuffles the data and groups it into [[mini-batch|mini-batches]].
- **The training loop is five lines that repeat (A.7–A.9):** forward pass → [[cross-entropy-loss]] → `zero_grad()` → `backward()` → `step()` (plain [[stochastic-gradient-descent]]). Save the weights with `state_dict` ([[model-checkpointing]]). Move model and data to `"cuda"` for [[gpu-computing]], and use [[distributed-data-parallel]] (DDP) to split the data across several GPUs and average their gradients.

## Key concepts introduced
**What PyTorch is, and where deep learning fits (§A.1)**
- [[pytorch]] — open-source Python deep-learning library; the main tool of the book
- [[artificial-intelligence]] ⊃ [[machine-learning]] ⊃ [[deep-learning]] — the nested sets of Fig. A.2
- [[neural-network]] — layers of artificial neurons; "deep" means many hidden layers that can model nonlinear relationships
- [[supervised-learning]] — train on (example, label) pairs, then predict labels for new data (Fig. A.3)
- [[text-classification]], [[next-token-prediction]], [[pretraining]], [[self-supervised-learning]] — how LLM training fits the supervised workflow (labels come from the text itself)
- [[empirical-risk-minimization]] — the formal reading of "train on a dataset" (wiki link, not book wording)
- [[gpu-computing]] — CUDA, ROCm, Google Colab and Apple Silicon (MPS) set-ups

**Tensors (§A.2)**
- [[tensor]] — multidimensional array; rank/order = number of axes (not the [[basis-and-rank|matrix rank]])
- [[vector]], [[matrix]] — rank-1 and rank-2 tensors
- [[floating-point-precision]] — int64 and float32 defaults; `.to(torch.float32)`
- [[tensor-reshaping-and-transposing]] — `.reshape`, `.view` (needs contiguous memory), `.T`
- [[matrix-transpose]] — $(\mathbf X^\top)_{ij} = X_{ji}$
- [[matrix-multiplication]] — `.matmul` and `@`; $(\mathbf{AB})_{ij} = \sum_k A_{ik}B_{kj}$
- [[gram-matrix]], [[inner-product]] — every entry of $\mathbf X\mathbf X^\top$ is a row–row dot product

**Graphs and gradients (§A.3–A.4)**
- [[computational-graph]] — a directed graph of operations, built on the fly ("dynamic")
- [[forward-pass]] — run inputs through the graph to get the output and the loss
- [[logistic-regression]] — "a single-layer neural network"; the running example
- [[sigmoid-function]] — $\sigma(z) = 1/(1+e^{-z})$, squashes the net input into $(0,1)$
- [[binary-cross-entropy]] — `F.binary_cross_entropy(a, y)`
- [[model-parameters]], [[bias-vector]] — the trainable weight $w_1$ and bias $b$
- [[derivative]], [[partial-derivative]], [[gradient]] — the sidebar's definitions
- [[chain-rule]] — multiply the local derivatives along the graph
- [[backpropagation]], [[automatic-differentiation]] — reverse-mode AD; `grad(...)` and `loss.backward()`
- [[gradient-descent]] — what the gradients are used for

**Building networks (§A.5)**
- [[pytorch-nn-module]] — subclass `torch.nn.Module`; layers in `__init__`, data flow in `forward`
- [[multilayer-perceptron]] — fully connected network; Listing A.4 is 50 → 30 → 20 → 3
- [[linear-layer]] — `torch.nn.Linear`, computes $XW^\top + \mathbf b$; also called a feedforward or fully connected layer
- [[relu]], [[activation-function]] — the nonlinearity between hidden layers
- [[logits]], [[softmax]] — raw scores, and how to turn them into class probabilities
- [[numerical-stability]] — why the loss functions take logits and apply softmax inside
- [[weight-initialization]] — small random weights break symmetry
- [[random-seed-reproducibility]] — `torch.manual_seed(123)`
- [[parameter-count-and-memory]] — $d_{\text{in}}d_{\text{out}} + d_{\text{out}}$ per linear layer

**Data and training (§A.6–A.8)**
- [[pytorch-dataset-and-dataloader]] — `__init__`, `__getitem__`, `__len__`; `batch_size`, `shuffle`, `drop_last`, `num_workers`
- [[mini-batch]], [[epoch]] — one epoch visits every training example once
- [[training-loop]] — forward → loss → `zero_grad` → `backward` → `step`
- [[stochastic-gradient-descent]] — `torch.optim.SGD(model.parameters(), lr=0.5)`
- [[learning-rate]], [[hyperparameter]] — `lr` and the number of epochs are tuned by hand
- [[cross-entropy-loss]] — `F.cross_entropy(logits, labels)`
- [[gradient-accumulation]] — PyTorch *adds* new gradients to `.grad`, so you reset them each step
- [[train-vs-eval-mode]], [[dropout]], [[batch-normalization]] — `model.train()` / `model.eval()` matter for layers that act differently at training time
- [[train-validation-test-split]] — use the validation set many times, the test set exactly once
- [[argmax]], [[classification-accuracy]] — predicted class and fraction correct
- [[model-checkpointing]] — `torch.save(model.state_dict(), "model.pth")` and `load_state_dict`

**GPUs (§A.9)**
- [[gpu-computing]] — devices, `.to("cuda")`, all tensors on the same device
- [[distributed-data-parallel]] — one process and one model copy per GPU, `DistributedSampler`, gradient averaging, NCCL

## Notable results / numbers
- **Popularity (book p. 251):** PyTorch is the most used research library since 2019 (Papers With Code); about **40%** of Kaggle 2022 survey respondents use it. The book uses **PyTorch 2.4.0** (book p. 256) and suggests a Python one or two releases behind the newest (3.11 or 3.12 if 3.13 is newest; book p. 254).
- **Tensor dtypes (book p. 259):** `torch.tensor([1, 2, 3])` → `torch.int64`; `torch.tensor([1.0, 2.0, 3.0])` → `torch.float32`. float32 is the default because it is precise enough, uses less memory and compute than 64-bit, and GPUs are optimised for it.
- **Reshape vs transpose (book pp. 260–261):** for
  $$\mathbf X = \begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad \texttt{reshape(3,2)} = \begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix},\qquad \mathbf X^\top = \begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix},\qquad \mathbf X\mathbf X^\top = \begin{bmatrix}14&32\\32&77\end{bmatrix}$$
  Both 3×2 results have the same shape but different entries. Check: $1+4+9 = 14$, $4+10+18 = 32$, $16+25+36 = 77$.
- **Logistic-regression forward pass (book pp. 262, 264):** $y = 1.0$, $x_1 = 1.1$, $w_1 = 2.2$, $b = 0.0$. So $z = 2.42$, $a = \sigma(2.42) \approx 0.9183$, and the loss $= -\ln 0.9183 \approx 0.0852$ (wiki arithmetic; the book does not print these three numbers).
- **Chain rule in Fig. A.8 (book p. 263):**
  $$\frac{\partial L}{\partial w_1} = \frac{\partial u}{\partial w_1}\cdot\frac{\partial z}{\partial u}\cdot\frac{da}{dz}\cdot\frac{\partial L}{\partial a},\qquad \frac{\partial L}{\partial b} = \frac{\partial z}{\partial b}\cdot\frac{da}{dz}\cdot\frac{\partial L}{\partial a}$$
  Autograd prints $\partial L/\partial w_1 = -0.0898$ and $\partial L/\partial b = -0.0817$ (book p. 264). Hand check: sigmoid + BCE gives $\partial L/\partial z = a - y \approx -0.0817$, and $\partial L/\partial w_1 = x_1(a-y) \approx -0.0898$.
- **`retain_graph=True` (book p. 264):** PyTorch frees the graph after computing gradients, so you must keep it if you call `grad` more than once.
- **Parameter count (book p. 267):** `NeuralNetwork(50, 3)` has **2,213** trainable parameters: $(50\cdot30+30) + (30\cdot20+20) + (20\cdot3+3) = 1530 + 620 + 63$. The first weight matrix has shape `torch.Size([30, 50])` (out × in).
- **Untrained outputs (book pp. 268–269):** with seed 123 and `X = torch.rand((1, 50))`, the logits are `[-0.1262, 0.1080, -0.1792]` with `grad_fn=<AddmmBackward0>` (matrix multiply + add). Softmax gives `[0.3113, 0.3934, 0.2952]`, almost uniform, as expected before training.
- **Toy data and loaders (book pp. 270–273):** 5 training examples with 2 features (3 of class 0, 2 of class 1), 2 test examples. With `batch_size=2` there are $\lceil 5/2\rceil = 3$ batches (the last has 1 example); with `drop_last=True` there are $\lfloor 5/2\rfloor = 2$. Class labels must start at 0, and the largest label must be at most (output nodes − 1). The author finds `num_workers=4` often best on real datasets (book p. 274).
- **Training run (book p. 275):** `NeuralNetwork(2, 2)`, SGD with `lr=0.5`, 3 epochs × 2 batches. Loss: 0.75, 0.65, 0.44, 0.13, 0.03, **0.00**. The model converged on the training set. Row 1 of the predicted probabilities is `[0.9991, 0.0009]` (99.91% class 0). Train and test accuracy are both **1.0** (book p. 278).
- **SGD update (book p. 276, in words):**
  $$\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal L(\theta),\qquad \eta = 0.5$$
- **GPU (book pp. 279–281):** `tensor([1., 2., 3.]) + tensor([4., 5., 6.])` → `tensor([5., 7., 9.], device='cuda:0')` on a GPU. Mixing devices raises `RuntimeError: Expected all tensors to be on the same device...`. Moving the loop to a GPU takes **three changed lines**. For this tiny model you will probably not see a speed-up, because copying data from CPU to GPU costs time.
- **DDP (book pp. 282–287):** each GPU gets a different, non-overlapping mini-batch; gradients are averaged before the update, so all replicas stay identical:
  $$g = \frac{1}{W}\sum_{r=0}^{W-1}\nabla_\theta L_r(\theta),\qquad \theta \leftarrow \theta - \eta\,g$$
  ($W$ = `world_size`, $r$ = `rank`; the book says this in words.) In theory 2 GPUs halve the epoch time and 8 GPUs make it 8× faster, minus communication overhead. Settings: `MASTER_ADDR="localhost"`, `MASTER_PORT="12345"`, `backend="nccl"`. The run prints `PyTorch version: 2.2.1+cu117`; 1-GPU losses 0.62 → 0.03 and 2-GPU losses 0.60/0.59 → 0.05/0.05, all accuracies 1.0. DDP does not work in Jupyter, so run it as a script.

> ⚠️ **Book typos (no effect on the method):** Fig. A.8's caption says "reverse-model" (for
> reverse-mode) and a callout says "trainable derivative" (for the bias) (book p. 263). The book
> prints `w1.grad` with a tuple wrapper `(tensor([-0.0898]),)`; real PyTorch prints a plain
> tensor (book p. 264). The softmax output has an extra `)` (book p. 269). One training log line
> says "Trainl Loss" (book p. 275). The `__len__` prose says "feature array" but the code uses
> `self.labels.shape[0]`; both are 5 here (book p. 271). §A.9.1 points to "2.2.2" for `.to()`;
> it means §A.2.2 (book p. 280). "multiprocessesing.spawn" and "nproces" mean
> `multiprocessing.spawn` and `nprocs` (book p. 286).

> ⚠️ **Name clash — "rank":** the book uses *rank* for the number of tensor axes (§A.2) and,
> in DDP, for the process/GPU index (§A.9.3). Neither is the linear-algebra matrix rank on
> [[basis-and-rank]].

## Exercises
| # | Book p. | Task | Solution idea (App. C, book pp. 310–312) |
|---|---|---|---|
| A.1 | 257 | Install and set up PyTorch on your computer. | Follow the "Python Setup Tips" in the repo (`setup/01_optional-python-setup-preferences`). |
| A.2 | 257 | Run the supplementary code (mng.bz/o05v) that checks your environment. | Use the checks in "Installing Libraries Used In This Book" (`setup/02_installing-python-libraries`). |
| A.3 | 275 | How many parameters does the network in Listing A.9 have? | `NeuralNetwork(2, 2)` has **752**: $(2\cdot30+30) + (30\cdot20+20) + (20\cdot2+2) = 90 + 620 + 42$. Count with `sum(p.numel() for p in model.parameters() if p.requires_grad)`. See [[parameter-count-and-memory]]. |
| A.4 | 282 | Compare CPU and GPU run time of [[matrix-multiplication]]. At what size does the GPU win? Hint: `%timeit a @ b`. | With `a` 100×200 and `b` 200×300: CPU **63.8 µs**, V100 GPU **13.8 µs**, about 4.6× faster on Colab. Results depend on hardware. For small matrices, copy and launch overhead hides the gain, because the work grows like $O(n^3)$. |

## References cited
Appendix B, Appendix A group (book pp. 298–299):
1. Raschka, Liu & Mirjalili (2022), *Machine Learning with PyTorch and Scikit-Learn* — a fuller introduction to deep learning.
2. Stevens, Antiga & Viehmann (2021), *Deep Learning with PyTorch* — a fuller introduction to deep learning.
3. Raschka, "Lecture 4.1: Tensors in Deep Learning" (15-minute video) — more on [[tensor|tensors]].
4. Raschka (2018), "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning", arXiv 1811.12808 — model evaluation ([[train-validation-test-split]], [[cross-validation]]).
5. Raschka, "Introduction to Calculus" (mng.bz/WEyW) — a free calculus refresher ([[derivative]], [[chain-rule]]).
6. Raschka, "Finetuning Large Language Models on a Single GPU Using Gradient Accumulation" (mng.bz/8wPD) — why PyTorch does not call `zero_grad()` for you ([[gradient-accumulation]]).
7. "Introducing PyTorch Fully Sharded Data Parallel (FSDP) API" (mng.bz/EZJR) — beyond [[distributed-data-parallel]]: shard a model that does not fit on one GPU.

Also named in the text: Papers With Code trends and the Kaggle 2022 survey (book p. 251); the
author's blog post "Scientific Computing in Python: Introduction to NumPy and Matplotlib"
(book p. 258); the PyTorch tensor docs (book pp. 260–261); the DDP docs (mng.bz/9dPr, book
p. 288); and the author's article "Accelerating PyTorch Model Training: Using Mixed-Precision
and Fully Sharded Data Parallelism" plus the Fabric library (book p. 288).

## People
- [[sebastian-raschka]] — the author; the source of the references above and of the advice on `num_workers=4`. No other person is named in Appendix A. PyTorch is described as a continuation of the Lua-based Torch library (book p. 256).

## What this changes in the wiki
- **New "how it runs in code" pages:** this unit owns the PyTorch-side pages — [[pytorch]], [[tensor]], [[pytorch-nn-module]], [[linear-layer]], [[pytorch-dataset-and-dataloader]], [[training-loop]], [[gpu-computing]], [[distributed-data-parallel]], [[random-seed-reproducibility]], [[gradient-accumulation]] — that the Ch. 2–7 source pages link to.
- **Ties MML calculus to working code:** [[derivative]], [[partial-derivative]], [[gradient]], [[chain-rule]], [[backpropagation]] and [[automatic-differentiation]] get a fully worked numeric example (the logistic-regression graph with $-0.0898$ and $-0.0817$). [[computational-graph]], [[forward-pass]], [[logistic-regression]], [[sigmoid-function]] and [[binary-cross-entropy]] are built around the same example.
- **Ties MML linear algebra to code:** [[matrix-multiplication]], [[matrix-transpose]] and [[gram-matrix]] get the $\mathbf X\mathbf X^\top$ example, and [[linear-layer]] shows that `nn.Linear` is an affine map ([[linear-mapping]] plus a bias, see [[affine-space]]) stored as an out × in matrix.
- **Strengthens optimisation pages:** [[stochastic-gradient-descent]], [[mini-batch]], [[learning-rate]] and [[hyperparameter]] now have a tiny end-to-end run (loss 0.75 → 0.00). [[distributed-data-parallel]] adds the fact that averaging per-GPU gradients equals the gradient of a bigger combined batch (wiki annotation, by linearity of the gradient; see [[expected-value-and-mean]]).
- **Clarification added to [[tensor]] and [[basis-and-rank]]:** tensor rank (number of axes) is not matrix rank (number of linearly independent columns).
- **No contradictions** with existing wiki pages. The book typos listed above are flagged here; none of them changes a result.
