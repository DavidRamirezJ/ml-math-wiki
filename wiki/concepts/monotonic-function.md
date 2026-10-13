---
title: Monotonic Function
type: concept
tags: [functions, calculus, analysis, decoding]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, llm-ch06-fine-tuning-for-classification, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Monotonic Function

> **One-liner:** A monotonic function never changes direction, so if one input is bigger than another, its output stays bigger too (or at least never ends up smaller).

## 🟢 Intuition (Dummy)
Think of a ranking of runners. If you add 10 seconds to everyone's time, or convert all times from seconds to minutes, the ranking doesn't change. The winner is still the winner. Those conversions are **monotonic**: they move every number, but they keep the order.

The book uses exactly this idea. A GPT model gives a score to every possible next token. Turning those scores into probabilities with [[softmax]] changes the numbers a lot, but it keeps them in the same order. So the token with the highest score is also the token with the highest probability. In the book's words, softmax "is monotonic, meaning it preserves the order of its inputs when transformed into outputs" (book p. 125). That's why you can skip softmax if all you want is the top token.

A non-example: a hill-shaped function goes up and then down, so two quite different inputs can end up with the same output and the order gets scrambled.

## 🟡 Mechanics (Practitioner)

### The definitions
A function $f$ is **(monotonically) increasing** if
$$x_1 < x_2 \;\Rightarrow\; f(x_1) \le f(x_2)$$
and **strictly increasing** if
$$x_1 < x_2 \;\Rightarrow\; f(x_1) < f(x_2).$$

Symbol by symbol:
- $x_1, x_2$ = any two inputs.
- $<$ = "is smaller than"; $\le$ = "is smaller than or equal to".
- $\Rightarrow$ = "implies", i.e. "whenever the left side is true, the right side is true too".
- So: "a smaller input never gives a bigger output" (increasing), or "a smaller input always gives a smaller output" (strictly increasing).

Flip the inequalities and you get **decreasing** and **strictly decreasing**. "Monotonic" means one of these four.

### Quick test with derivatives
If $f$ is differentiable, then $f'(x) \ge 0$ everywhere ⇒ $f$ is increasing, and $f'(x) > 0$ everywhere ⇒ strictly increasing ([[derivative]]). Examples:
- $e^x$: derivative $e^x > 0$, so **strictly increasing**. This is the key piece inside softmax.
- $\log x$ on $x>0$: derivative $1/x > 0$, so **strictly increasing** ([[log-probability]]).
- $\sigma(x) = \frac{1}{1+e^{-x}}$, the [[sigmoid-function]]: strictly increasing.
- [[relu]] $= \max(0,x)$: increasing but **not strictly** (all negatives map to the same 0).
- [[gelu]]: **not monotonic**. Its curve dips below 0 for negative inputs and then rises. The book says its gradient is non-zero "for almost all negative values (except at approximately x = –0.75)", i.e. the bottom of the dip is at $x \approx -0.75$ (book Fig. 4.8, p. 106). The dip depth, about $-0.17$, is our calculation, not printed in the book.

### Why softmax keeps the order
$$\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
- $\mathbf{z}$ = the vector of [[logits]]; $z_i$ = the logit of token (or class) $i$; $K$ = how many there are (50,257 for GPT-2, 2 for the spam classifier).
- $e^{z_i}$ = "e to the power $z_i$", always positive.
- $\sum_{j=1}^{K} e^{z_j}$ = add up all $K$ exponentials. **Every** entry is divided by this same number.

Since $e^x$ is strictly increasing and every entry shares the same positive denominator:
$$z_i > z_j \iff \mathrm{softmax}(\mathbf{z})_i > \mathrm{softmax}(\mathbf{z})_j$$
($\iff$ = "if and only if"). So the [[argmax]] of the probabilities is the argmax of the logits.

### The book's numbers
- **§4.7 (book p. 124):** last-row logits `[-0.4929, ..., 2.4812, ..., -0.6093]` → probabilities `[0.0001, ..., 0.0200, ..., 0.0001]`. The largest logit (2.4812) becomes the largest probability (0.0200), at index 257 = "a".
- **§A.7 (book pp. 276–277):** logits row `[2.8569, -4.1618]` → probabilities `[0.9991, 0.0009]`. Check: $e^{2.8569}/(e^{2.8569}+e^{-4.1618}) = 1/(1+e^{-7.0187}) \approx 0.9991$. Order kept, so `torch.argmax(probas, dim=1)` and `torch.argmax(outputs, dim=1)` both give `tensor([0, 0, 0, 1, 1])`.

```python
probas = torch.softmax(logits, dim=-1)
torch.argmax(probas, dim=-1) == torch.argmax(logits, dim=-1)   # always True (ties aside)
```

The book keeps the softmax line in `generate_text_simple` anyway, "to illustrate the full process of transforming logits to probabilities" (book p. 125).

## 🔴 Theory (Expert)
- **Order-preserving maps.** A strictly increasing $g:\mathbb{R}\to\mathbb{R}$ is an order embedding: $a<b \iff g(a)<g(b)$. So it is injective (one-to-one), and $\arg\max_s g(f(s)) = \arg\max_s f(s)$ for any $f$. Non-strict monotone maps keep the maximizer *inside* the argmax set but can create new ties (e.g. ReLU sends all negative logits to 0).
- **Softmax is "monotonic" in a vector sense.** Softmax maps $\mathbb{R}^K \to \Delta^{K-1}$ (the probability simplex), so the scalar definition doesn't apply directly. The precise statements are: (1) it is **order-preserving across coordinates**, $z_i>z_j \iff p_i>p_j$; (2) each $p_i$ is strictly increasing in its own logit, $\partial p_i/\partial z_i = p_i(1-p_i) > 0$, and decreasing in the others, $\partial p_i/\partial z_j = -p_ip_j < 0$. The book's informal "monotonic" means (1).
- **Where this fact is used in ML.**
  - *Logs:* because $\log$ is strictly increasing, maximizing a likelihood = maximizing its log. That's what makes [[maximum-likelihood-estimation]] and [[cross-entropy-loss]] workable (products become sums; see [[log-probability]], [[numerical-stability]]).
  - *Temperature:* dividing logits by $T>0$ is strictly increasing, so it changes the *spread* of the probabilities but not the argmax ([[temperature-scaling]]). Greedy output is the same at any temperature.
  - *Shift invariance:* $\mathrm{softmax}(\mathbf{z}+c\mathbf{1}) = \mathrm{softmax}(\mathbf{z})$. Subtracting $\max_j z_j$ before exponentiating is the standard stable implementation.
  - *Calibration:* monotone post-hoc maps (e.g. temperature, isotonic regression) can fix confidence without changing accuracy, since predicted labels stay the same.
- **Monotone ⇒ nice analytic behaviour.** A monotone function on an interval has at most countably many jump points and is differentiable almost everywhere (Lebesgue's theorem). These matter little in deep learning, but they explain why monotone activations (ReLU, sigmoid, tanh) are easy to analyse. The book calls [[gelu]] and [[swiglu]] "more complex and smooth" than ReLU (book p. 105). For GELU it says the smoothness "can lead to better optimization properties during training", and its small non-zero output for negative inputs lets those neurons still contribute to learning (book p. 106). The non-monotone dip is the price of that smooth, non-zero negative branch.

## Connections
- Builds on: [[derivative]], [[softmax]], [[logits]]
- Leads to: [[argmax]], [[greedy-decoding]], [[log-probability]], [[temperature-scaling]]
- Used by: [[autoregressive-text-generation]], [[classification-fine-tuning]], [[maximum-likelihood-estimation]], [[cross-entropy-loss]], [[numerical-stability]]
- Examples and non-examples: [[relu]], [[sigmoid-function]], [[gelu]], [[activation-function]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 124–125 (the book's definition: softmax "is monotonic, meaning it preserves the order of its inputs when transformed into outputs"; the softmax step is redundant before `torch.argmax`; Fig. 4.17 numbers).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6, book p. 192 (largest outputs correspond to highest probabilities, so softmax is optional for class labels).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7, book p. 277 (argmax on logits and on probabilities agree).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book p. 106 (Fig. 4.8: the GELU curve, a non-monotonic example).
- Formal definitions, the derivative test, softmax partial derivatives and Lebesgue's theorem are standard analysis, not printed in the book.
