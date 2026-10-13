---
title: Floating-Point Precision and Tensor Data Types
type: concept
tags: [numerical-computing, pytorch, deep-learning, llm]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Floating-Point Precision and Tensor Data Types

> **One-liner:** A computer stores every number in a fixed number of bits, so it can only keep so many digits — and the "data type" (like `float32`) decides how many digits you get, how much memory each number costs, and how fast the math runs.

## 🟢 Intuition (Dummy)
Imagine writing numbers on sticky notes that only have room for about seven
digits. Most of the time that's plenty. But if you add up a bunch of numbers
that should cancel out to exactly zero, the tiny bits you had to round off
along the way don't quite cancel — you end up with something like
0.00000006 instead of 0. That's not a bug in your code; it's the sticky note
being too small.

A **data type** (`dtype`) is the size of the sticky note. Bigger notes
(`float64`) hold more digits but take twice the space and are slower to
process. Smaller notes (`float32`, or even 16-bit types) are cheaper and
faster. Deep learning almost always picks `float32` as the default, because
neural networks don't need 15 exact digits — they need lots of numbers,
crunched quickly. Raschka's book puts it simply: 32-bit floats balance
precision and efficiency (book p. 259).

Integers get their own type too: token IDs and class labels are stored as
whole numbers (`int64`), because "token number 6109" must never become
"6108.9999".

## 🟡 Mechanics (Practitioner)

### PyTorch's default dtypes (§A.2.2, book pp. 259–260)
```python
import torch
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)          # torch.int64   (from Python ints)

floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)          # torch.float32 (from Python floats)

floatvec = tensor1d.to(torch.float32)   # change the type with .to
print(floatvec.dtype)          # torch.float32
```
- Python integers → **`torch.int64`** (64-bit integers, Python's default).
- Python floats → **`torch.float32`** (32-bit floating point).
- `.to(...)` converts between types. The **same** `.to` method also moves a
  tensor to a device, e.g. `.to("cuda")` for a GPU (§A.9, p. 280; see
  [[gpu-computing]]).

**Why `float32` is the default** — the book's four reasons (p. 259):
1. it balances precision against computational efficiency;
2. it is precise enough for most deep-learning tasks;
3. it uses less memory and compute than 64-bit floats;
4. GPU architectures are optimized for 32-bit math, which "can significantly
   speed up model training and inference."

### How a float stores a number (general background, not from the book)
A floating-point number is scientific notation in base 2:
$$x=(-1)^{s}\times(1.f)_2\times2^{\,e-\text{bias}}$$
Read it symbol by symbol:
- $s$ — the **sign bit** (0 = positive, 1 = negative); $(-1)^s$ is $+1$ or $-1$.
- $(1.f)_2$ — the **mantissa** (significand): "1 point *f*" written in binary, where $f$ is the fraction bits. More fraction bits = more digits of precision.
- $e$ — the stored **exponent bits**; subtracting the bias (127 for float32) allows negative exponents.
- $2^{e-\text{bias}}$ — the scale: it slides the binary point left or right. More exponent bits = wider range.

| dtype | bits (sign / exponent / fraction) | bytes | ≈ decimal digits | machine epsilon $\varepsilon_{\text{mach}}$ |
|---|---|---|---|---|
| `float64` | 1 / 11 / 52 | 8 | ~16 | $2^{-52}\approx2.2\times10^{-16}$ |
| `float32` | 1 / 8 / 23 | 4 | ~7 | $2^{-23}\approx1.19\times10^{-7}$ |
| `float16` | 1 / 5 / 10 | 2 | ~3 | $2^{-10}\approx9.8\times10^{-4}$ |
| `bfloat16` | 1 / 8 / 7 | 2 | ~2–3 | $2^{-7}\approx7.8\times10^{-3}$ |
| `int64` | (whole numbers) | 8 | exact | — |

**Machine epsilon** $\varepsilon_{\text{mach}}$ = the gap between 1 and the next
representable number. It is the "tick size" of the number grid near 1.

### The book's rounding example (§4.2, book p. 102)
After [[layer-normalization]] / [[standardization-z-score|standardization]], each
row should have mean exactly 0. PyTorch prints:
```
Mean:
 tensor([[-5.9605e-08],
         [ 1.9868e-08]])
```
- `-5.9605e-08` is scientific notation for $-5.9605\times10^{-8}=-0.000000059605$.
- The book: "very close to 0, but it is not exactly 0 due to small numerical
  errors that can accumulate because of the finite precision with which
  computers represent numbers."
- `torch.set_printoptions(sci_mode=False)` hides this; the means then print as `0.0000`.

**Where those exact numbers come from (our check).** The mean is the sum of 6
values divided by 6. Multiply back:
- row 1: $-5.9605\times10^{-8}\times6\approx-3.58\times10^{-7}=-3\times2^{-23}$,
- row 2: $1.9868\times10^{-8}\times6\approx1.19\times10^{-7}=1\times2^{-23}$.

So the two sums were off by exactly **three** and **one** float32 "ticks"
($2^{-23}$) — rounding error at the scale of machine epsilon, just as the
table predicts.

A related trick from the same section: LayerNorm divides by
$\sqrt{\sigma^2+\varepsilon}$ with `eps = 1e-5` (p. 103), a small constant that
prevents division by zero (see [[numerical-stability]]). The checked
variances come out as 0.9995 and 0.9997 — "approximately 1" in the book's
words (p. 104). Our explanation (not the book's): the output variance is
$\sigma^2/(\sigma^2+\varepsilon)$, a little under 1 because of `eps`.

### Precision = memory (§4.6, book p. 122)
The book sizes its GPT model by multiplying the parameter count by 4 bytes
(one float32):
```python
total_size_bytes = total_params * 4          # float32 = 4 bytes / parameter
total_size_mb = total_size_bytes / (1024 * 1024)
```
$$\text{size}_{\text{MB}}=\frac{N_{\text{params}}\times b}{1024^2}=\frac{163{,}009{,}536\times4}{1{,}048{,}576}\approx621.83\ \text{MB}$$
- $N_{\text{params}}$ = number of parameters (163,009,536 for the untied GPT-2 124M config);
- $b$ = bytes per number (4 for float32);
- $1024^2$ = bytes per MB (mebibyte).

Same model in other dtypes (our arithmetic): float64 ≈ 1243.66 MB,
float16/bfloat16 ≈ 310.92 MB. This is why halving the precision is the first
lever people pull for big models (see [[parameter-count-and-memory]]).

## 🔴 Theory (Expert)
- **Relative error model.** Every float operation is exact-then-rounded:
  $\operatorname{fl}(a\circ b)=(a\circ b)(1+\delta)$ with
  $|\delta|\le\varepsilon_{\text{mach}}/2$ (round-to-nearest). Summing $n$
  numbers naively can accumulate error up to about $(n-1)\varepsilon_{\text{mach}}\sum|x_i|$;
  that is the "errors that can accumulate" of the book's p. 102 remark.
- **Cancellation.** Subtracting nearly equal numbers (like a sum that should
  be 0 after mean-centering) keeps the *absolute* rounding error but destroys
  the *relative* accuracy — the result is pure noise at the
  $\varepsilon_{\text{mach}}$ scale. Hence "mean = $-5.96\times10^{-8}$" is
  float32's way of saying "zero."
- **Range vs precision.** Exponent bits set range, fraction bits set
  precision. `float16` maxes out at 65,504 and overflows easily (e.g. in
  $e^{z}$ inside [[softmax]]); `bfloat16` keeps float32's 8 exponent bits — same
  range, far less precision — which is why it became the standard for LLM
  training. Fused, numerically careful kernels (log-sum-exp inside
  `F.cross_entropy`, the max-subtraction in softmax) exist to survive these
  limits ([[numerical-stability]]).
- **Mixed precision.** Practical training stores/computes most tensors in 16
  bits but keeps a float32 copy of the weights and accumulates sums in
  float32. Raschka points to his article "Accelerating PyTorch Model
  Training: Using Mixed-Precision and Fully Sharded Data Parallelism" for this
  (§A.9, p. 288); the book's main code stays in float32.
- **Integers are exact.** `int64` represents every integer up to
  $2^{63}-1\approx9.2\times10^{18}$ exactly, which is why indices such as
  [[token-id|token IDs]] and class labels use it (embedding lookups and
  `cross_entropy` targets expect integer tensors).
- **Not the same "precision" as in classification.** "Precision" here means
  significant digits, not the precision/recall metric.

> The bit layouts, epsilon values, error model, bfloat16/float16 facts, and
> the $2^{-23}$ decomposition of the book's printout are standard
> numerical-computing background added for this wiki; they are not stated in
> the cited sections.

## Connections
- Builds on: [[tensor]], [[pytorch]]
- Leads to: [[numerical-stability]], [[parameter-count-and-memory]], [[gpu-computing]]
- Used by: [[layer-normalization]], [[standardization-z-score]], [[softmax]], [[cross-entropy-loss]], [[gpt-model-configuration]], [[distributed-data-parallel]]
- Related: [[token-id]], [[gradient]] (tiny gradients can underflow in 16-bit types)
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2.2 "Tensor data types", book pp. 259–260 (PDF pp. 281–282).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2 (scientific notation, finite-precision error, `sci_mode=False`, `eps=1e-5`), book pp. 102–104 (PDF pp. 124–126).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6 (model memory in float32, 621.83 MB), book p. 122 (PDF p. 144).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.9 (`.to` for devices; mixed-precision article pointer), book pp. 280, 288.
