---
title: Gradient Clipping
type: concept
tags: [optimization, deep-learning, llm, training, norms]
difficulty: intermediate
sources: [llm-appD-training-loop-bells-and-whistles, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Gradient Clipping

> **One-liner:** If the gradient gets too big, shrink it down to a fixed maximum size — keep the direction, cap the length.

## 🟢 Intuition (Dummy)
The [[gradient]] is an arrow that tells the model "go this way, this much" to reduce its error.
Usually the arrow is reasonable. But once in a while — an odd batch, a steep cliff in the loss
landscape — the arrow becomes enormous, and following it would fling the weights far away and
wreck everything learned so far (the "exploding gradients" problem).

Gradient clipping puts a leash on that arrow. You pick a maximum length. If the arrow is shorter,
nothing happens. If it is longer, you shrink it until its length equals the maximum, **but it still
points in the same direction**. The model still moves the right way, just not dangerously far.
Raschka's book uses it as the third "bell and whistle" for stable LLM training, next to
[[learning-rate-warmup]] and [[cosine-decay]].

## 🟡 Mechanics (Practitioner)
**Measuring "how big": the L2 norm.** The book measures the gradient's size with the L2 (Euclidean)
[[norm]] (book p. 317):
$$\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$
- $\mathbf v$ = the gradient written as one long [[vector]]; $v_i$ = its $i$-th entry (one [[partial-derivative]] of the loss with respect to one parameter).
- $n$ = the number of entries — one per trainable parameter. For the book's `GPTModel` with `GPT_CONFIG_124M` (no weight tying, `context_length=256` in appendix D) that is about 162 million; the "124M" name counts weights with tying (book §4.6).
- $\|\cdot\|_2$ = "the length of"; square every entry, add them up, take the square root — Pythagoras in $n$ dimensions.
- For a [[matrix]] of gradients you do the same: square every entry, sum, square-root (book p. 318).

**The clipping rule (book p. 318).** If $\|\mathbf G\|_2 > \text{max\_norm}$:
$$c = \frac{\text{max\_norm}}{\|\mathbf{G}\|_2}, \qquad \mathbf{G}' = c\,\mathbf{G}$$
- $\mathbf G$ = the gradient (matrix or vector); $\mathbf G'$ = the clipped gradient.
- max_norm = the cap you choose (the book uses 1.0).
- $c$ = the scaling factor, a number smaller than 1. Multiplying every entry by the same $c$ keeps the direction and makes $\|\mathbf G'\|_2 = \text{max\_norm}$ exactly.

Compact form covering both cases:
$$\mathbf g \leftarrow \mathbf g \cdot \min\!\left(1, \frac{\text{max\_norm}}{\|\mathbf g\|_2}\right)$$
"multiply by 1 (no change) if already small enough, otherwise by the shrink factor."

**Worked example (book p. 318).** $\mathbf G = \begin{bmatrix}1 & 2\\ 3 & 4\end{bmatrix}$, max_norm $= 1$.

> ⚠️ **Book typo:** the book writes $\|\mathbf G\|_2 = \sqrt{1^2 + 2^2 + 2^2 + 4^2} = \sqrt{25} = 5$ — it uses $2^2$
> where the entry is 3. The correct value is $\sqrt{1 + 4 + 9 + 16} = \sqrt{30} \approx 5.477$. The book's
> $\mathbf G' = \tfrac15 \mathbf G$ would then have norm $\sqrt{30}/5 \approx 1.095$, not 1. The method is right; only the arithmetic slipped.

Correct calculation:
- $\|\mathbf G\|_2 = \sqrt{30} \approx 5.477 > 1$, so clip.
- $c = 1/\sqrt{30} \approx 0.1826$
- $\mathbf G' \approx \begin{bmatrix}0.183 & 0.365\\ 0.548 & 0.730\end{bmatrix}$, and $\|\mathbf G'\|_2 = 1$ ✓.
- Book's printed answer: $\mathbf G' = \tfrac15\mathbf G = \begin{bmatrix}1/5 & 2/5\\ 3/5 & 4/5\end{bmatrix}$, whose norm is $\sqrt{30}/5 \approx 1.095$, not 1. (The book's $\sqrt{25}=5$ would be the correct norm only for $\begin{bmatrix}1&2\\2&4\end{bmatrix}$.)

**PyTorch demo (book pp. 318–319).** Build a fresh GPT-2 124M model, compute the loss on one batch, and
backpropagate so every parameter's `.grad` is filled:

```python
from chapter05 import calc_loss_batch
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()

def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad

print(find_highest_gradient(model))        # tensor(0.0411)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(find_highest_gradient(model))        # tensor(0.0185)
```

- `clip_grad_norm_` (trailing underscore = in-place) computes **one** norm over *all* parameters' gradients
  together, as if they were one giant vector, and scales every gradient by the same factor.
- The largest single gradient entry drops from **0.0411** to **0.0185**. The ratio $0.0185/0.0411 \approx 0.45$
  is the common factor $c$, which implies the total gradient norm for that batch was about $1/0.45 \approx 2.2$
  (derived, not stated in the book).

**In the full training function (§D.4, book p. 320).** Clipping goes **between** `loss.backward()` and
`optimizer.step()`, and the book only turns it on after warmup:

```python
loss.backward()
if global_step >= warmup_steps:            # avoid exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Chapter 5's simple loop only names gradient clipping and points to appendix D (book p. 146).

## 🔴 Theory (Expert)
- **Geometry.** Norm clipping is the Euclidean projection of $\mathbf g$ onto the closed ball
  $B = \{\mathbf x : \|\mathbf x\|_2 \le \text{max\_norm}\}$: points inside stay put, points outside go radially to
  the boundary. It changes magnitude, never direction (unlike *value* clipping, `clip_grad_value_`, which clamps
  each entry separately and can change direction).
- **Frobenius = flattened L2.** The matrix "L2" in the book's example is the Frobenius norm
  $\|\mathbf G\|_F = \sqrt{\sum_{ij} G_{ij}^2} = \sqrt{\operatorname{tr}(\mathbf G^\top \mathbf G)}$, equal to the vector
  2-norm of $\operatorname{vec}(\mathbf G)$. It is **not** the operator 2-norm (largest singular value); for
  $\begin{bmatrix}1&2\\3&4\end{bmatrix}$ that would be $\approx 5.465$. See [[norm]].
- **Global vs per-tensor.** PyTorch's global clipping preserves the direction of the *whole* update vector.
  Clipping each tensor separately would change the relative step sizes between layers.
- **Why gradients explode.** By the [[chain-rule]], a deep network's gradient is a product of many layer
  Jacobians; if their norms exceed 1 the product can grow exponentially in depth (see
  [[vanishing-and-exploding-gradients]], [[backpropagation]]). Clipping is a cheap safety net, not a cure — residual
  [[shortcut-connections]] and [[layer-normalization]] address the cause.
- **Interaction with the optimizer.** With plain [[stochastic-gradient-descent|SGD]], clipping caps the step
  length at $\eta \cdot \text{max\_norm}$. With [[adamw-optimizer|AdamW]] (used in the book), per-parameter
  normalisation by $\sqrt{\hat v_t}$ means the step is not capped in that direct way; clipping still damps outlier
  batches in the moment estimates. The book does not discuss this (background).
- **Implementation detail.** PyTorch computes $c = \text{max\_norm}/(\|\mathbf g\|_2 + \varepsilon)$ with a tiny
  $\varepsilon$ for [[numerical-stability]] and returns the pre-clip total norm, which is handy for logging.

## Connections
- Builds on: [[norm]], [[gradient]], [[vector]], [[matrix]], [[partial-derivative]], [[backpropagation]]
- Leads to: [[llm-training-loop]]
- Used by: [[pretraining]], [[training-loop]], [[adamw-optimizer]]
- Related: [[vanishing-and-exploding-gradients]], [[learning-rate-warmup]], [[cosine-decay]], [[learning-rate-schedule]], [[automatic-differentiation]], [[chain-rule]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.3 Gradient clipping, book pp. 317–319 (L2 norm p. 317; matrix example p. 318; code and outputs pp. 318–319).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.4 The modified training function, book p. 320 (clipping after warmup, `max_norm=1.0`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2, book p. 146 (pointer to appendix D).
