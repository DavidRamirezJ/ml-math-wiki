---
title: Numerical Stability (Softmax Overflow/Underflow)
type: concept
tags: [numerical-methods, floating-point, softmax, deep-learning]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-ch05-pretraining-on-unlabeled-data, llm-appA-introduction-to-pytorch, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Numerical Stability (Softmax Overflow/Underflow)

> **One-liner:** Computers store numbers with limited room, so a formula that is perfect on paper can blow up to "infinity" or collapse to "zero" in code — numerical stability is the art of rewriting it so that never happens.

## 🟢 Intuition (Dummy)
Imagine a calculator that can only show numbers up to a certain size. Ask it for
$e^{1000}$ and it just shows "∞". Ask it for $e^{-1000}$ and it shows "0". Now you try to
compute a fraction like $\frac{\infty}{\infty}$ or $\frac{0}{0}$ — the answer is garbage
("NaN", *not a number*).

That is exactly what can happen to a naive [[softmax]], which is full of $e^{x}$ terms. The
book warns that its hand-written `softmax_naive` "may encounter numerical instability
problems, such as overflow and underflow, when dealing with large or small input values",
and recommends PyTorch's built-in `torch.softmax` instead (book p. 60).

The fix is a neat trick: before exponentiating, **subtract the biggest score from all the
scores**. The answer does not change at all (softmax doesn't care about a common shift),
but now the biggest exponent is $e^0 = 1$ — nothing can overflow.

## 🟡 Mechanics (Practitioner)

### The problem
Naive softmax (book p. 60):
```python
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
```
On the book's small attention scores this is fine — it gives
`[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]`, identical to `torch.softmax` (book p. 60).
Trouble starts with extreme inputs. In 32-bit floats ([[floating-point-precision]]):
- **Overflow:** the largest float32 is about $3.4\times10^{38}$, and $\ln(3.4\times10^{38}) \approx 88.7$.
  So $e^{x}$ becomes `inf` for any $x > 88.7$. If two scores overflow you get
  $\infty/\infty$ = `nan`.
- **Underflow:** for very negative $x$ (below about −104 in float32), $e^{x}$ rounds to exactly 0.
  If *all* scores are that negative, the denominator is 0 and you get $0/0$ = `nan`.

(Thresholds are standard IEEE-754 facts, not from the book.)

### The fix: subtract the max
$$\operatorname{softmax}(\mathbf z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_j z_j$$

Symbol by symbol:
- $\mathbf z$ = the input scores; $z_i$ = one of them.
- $m$ = the largest score in the list ("max over $j$ of $z_j$").
- $z_i - m$ = the shifted score; it is always $\le 0$, so $e^{z_i - m} \le 1$ (no overflow).
- At least one term equals $e^{0} = 1$, so the denominator is $\ge 1$ (no $0/0$).

Why it's allowed: $e^{z_i - m} = e^{z_i}\,e^{-m}$, and the factor $e^{-m}$ appears in the top
and in every term of the bottom, so it cancels.

**Worked example (book's scores).** For `[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]`,
$m = 1.4950$; the shifted exponentials are `[0.5824, 1.0, 0.9806, 0.5212, 0.4548, 0.6646]`.
Dividing by their sum (4.2036) gives the same weights `[0.1385, 0.2379, …]`. (Our computation.)

```python
def softmax_stable(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    return torch.exp(x) / torch.exp(x).sum(dim=dim, keepdim=True)
```
(Our sketch — in practice just call `torch.softmax`, which is implemented stably.)

### The log version: log-sum-exp
Losses need $\log$ of a softmax probability. Computing `log(softmax(z))` can give
$\log 0 = -\infty$ when a probability underflows. The stable form is
$$\log \operatorname{softmax}(\mathbf z)_y = z_y - \operatorname{LSE}(\mathbf z), \qquad \operatorname{LSE}(\mathbf z) = m + \log\sum_j e^{z_j - m}$$
- $y$ = index of the correct class/token; $\operatorname{LSE}$ = "log-sum-exp".

This is why PyTorch models return raw **[[logits]]** and loss functions combine softmax and
the negative log-likelihood in one step "for numerical efficiency and stability" (book
p. 269), and why `torch.nn.functional.cross_entropy` takes logits and applies softmax
internally (book p. 276). Never feed it probabilities you already softmaxed.

### Related stability tricks in the book
- **Log-probabilities** instead of raw probabilities in the loss: tiny numbers like
  $7.45\times10^{-5}$ become manageable logs like −9.5042 (book pp. 137–138; see [[log-probability]]).
- **Scaling attention scores by $1/\sqrt{d_k}$** keeps softmax away from its saturated
  "step function" regime, where gradients vanish (book p. 69; see
  [[scaled-dot-product-attention]], [[vanishing-and-exploding-gradients]]).
- **Masking with $-\infty$** is numerically safe as long as at least one entry per row is
  finite: $e^{-\infty} = 0$ exactly (book p. 77). In causal attention the diagonal is never
  masked, so every row has a finite entry.

## 🔴 Theory (Expert)
- **Shift invariance** is the algebraic fact behind the trick:
  $\operatorname{softmax}(\mathbf z + c\mathbf 1) = \operatorname{softmax}(\mathbf z)$ for every $c\in\mathbb R$.
  Choosing $c = -\max_j z_j$ is the unique shift that makes the largest exponent exactly 1.
- **Error analysis.** With $c = -m$, every term $e^{z_j - m}\in(0,1]$ and the sum lies in
  $[1, K]$, so no intermediate overflows, and the relative error of the result is $O(K\,\varepsilon)$
  in machine epsilon $\varepsilon$. Terms that underflow to 0 were below $\varepsilon$
  relative to the max term anyway, so dropping them is harmless.
- **LSE bounds.** $m \le \operatorname{LSE}(\mathbf z) \le m + \log K$, so the log-sum-exp can never
  overflow if $m$ is finite.
- **Fully masked rows.** If an entire row is $-\infty$ (e.g. a padding query with every key
  masked), $m = -\infty$ and $z - m$ is $-\infty-(-\infty)$ = NaN. Implementations guard
  this case; the book's causal mask avoids it because the diagonal stays unmasked.
- **Lower precision.** In bfloat16/float16 ([[floating-point-precision]]) the dynamic range or
  mantissa is smaller, so stable softmax/LSE and float32 accumulation matter even more —
  fused kernels such as [[flashattention]] compute a running max and running sum for
  exactly this reason ("online softmax").
- **Conditioning of softmax.** Its Jacobian $\operatorname{diag}(\mathbf s) - \mathbf s\mathbf s^\top$ satisfies
  $0 \preceq J \preceq \operatorname{diag}(\mathbf s)$, so its spectral norm is at most $\max_i s_i \le 1$:
  softmax itself is well behaved (1-Lipschitz); the instability is purely
  in the naive *floating-point* evaluation of $e^x$, not in the function.

> **From [[llm-ch04-implementing-a-gpt-model]]:** `LayerNorm` computes $(x-\mu)/\sqrt{\sigma^2+\varepsilon}$ with $\varepsilon = 10^{-5}$ (book p. 103). Read it as: "subtract the mean $\mu$, then divide by the square root of the variance $\sigma^2$ plus a tiny $\varepsilon$". The $\varepsilon$ stops a division by zero when a row has (almost) no variance. Side effect: the normalized variance is slightly below 1 (0.9995, 0.9997).

## Connections
- Builds on: [[softmax]], [[floating-point-precision]], [[logits]]
- Leads to: [[cross-entropy-loss]], [[log-probability]], [[flashattention]]
- Used by: [[attention-weights]], [[scaled-dot-product-attention]], [[causal-attention-mask]], [[llm-training-loop]]
- Related: [[vanishing-and-exploding-gradients]], [[maximum-likelihood-estimation]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3.1, book p. 60 (`softmax_naive`, overflow/underflow warning, use `torch.softmax`).
- Same, §3.4.1 sidebar, book p. 69 (softmax saturation and small gradients); §3.5.1, book p. 77 ($-\infty$ masking).
- Same, §5.1, book pp. 137–138 (log probabilities are easier to optimize).
- Same, §A.5, book p. 269 (models return logits; fused softmax + NLL for stability); §A.7, book p. 276 (`F.cross_entropy` applies softmax internally).
- Same PDF, §4.2 Listing 4.2, book p. 103 (`eps = 1e-5` added to the variance in `LayerNorm` to prevent division by zero; see [[layer-normalization]]).
