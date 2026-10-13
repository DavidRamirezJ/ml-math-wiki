---
title: Label Encoding (Zero-Based Class Labels)
type: concept
tags: [classification, data-preparation, pytorch]
difficulty: beginner
sources: [llm-ch06-fine-tuning-for-classification, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Label Encoding (Zero-Based Class Labels)

> **One-liner:** Label encoding swaps word labels like "ham" and "spam" for the numbers 0 and 1, starting from zero, so the model can use them as positions in its output.

## 🟢 Intuition (Dummy)
Computers don't do words, they do numbers. So before training, we give each class a number,
like seats in a row: the first class sits in seat **0**, the next in seat **1**, and so on.

Why start at 0 instead of 1? Because the model's output is a little list of scores, one per
class, and Python counts list positions from 0. Label "1" literally means "look at the score in
position 1". If you started counting at 1, the last class would point to a seat that doesn't exist.

Raschka compares it to [[token-id|token IDs]]: same idea of turning text into integers, just
with a tiny vocabulary of **two** instead of GPT-2's 50,000+ words (book p. 174).

## 🟡 Mechanics (Practitioner)

### In the spam example (book p. 174)
```python
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
```
Ham (not spam) becomes 0, spam becomes 1. Appendix E does exactly the same before LoRA
(book p. 324). As a formula:

$$y=\begin{cases}0 & \text{if label = ham}\\ 1 & \text{if label = spam}\end{cases}$$

- $y$ = the numeric label the model trains on.
- The curly brace means "pick the line whose condition is true".

At prediction time we go back the other way: `"spam" if predicted_label == 1 else "not spam"`
(book p. 201).

### PyTorch's rule (Appendix A, book p. 271)
> Class labels must **start at 0**, and the largest label may not exceed
> **(number of output nodes − 1)**.

Example from the book: labels 0, 1, 2, 3, 4 need an output layer with **5 nodes**. The toy dataset
in Listing A.5 uses `y_train = torch.tensor([0, 0, 0, 1, 1])` — three examples of class 0, two of
class 1 (book p. 270).

In symbols:

$$y\in\{0,1,\dots,K-1\},\qquad \mathbf z\in\mathbb R^{K}$$

- $K$ = number of classes = number of output nodes of the [[classification-head]].
- $y$ = a class label; $\{0,1,\dots,K-1\}$ = "one of the whole numbers from 0 to $K-1$".
- $\mathbf z$ = the output vector of $K$ [[logits]]; $\mathbb R^K$ = "a list of $K$ real numbers".

### Why the label is an index
The cross-entropy loss literally *looks up* the score of the true class:

$$\ell(\mathbf z,y)=-\log\frac{e^{z_y}}{\sum_{j=0}^{K-1}e^{z_j}}$$

- $z_y$ = entry number $y$ of the logit vector — the label is used as a subscript (an index).
- $e^{z_y}/\sum_j e^{z_j}$ = the [[softmax]] probability of the true class.
- $-\log$ = turn "high probability" into "low loss" (see [[cross-entropy-loss]]).

So if $y=5$ but $\mathbf z$ only has positions 0–4, PyTorch raises an index error. That's the
whole reason for the zero-based rule.

## 🔴 Theory (Expert)
- **A bijection, not a measurement.** Label encoding is an arbitrary one-to-one map
  $\phi:\mathcal C\to\{0,\dots,K-1\}$ from class names to indices. The numbers carry **no order or
  distance** — "spam = 1 > ham = 0" means nothing. Any permutation of $\phi$ gives an equivalent
  model (the head's rows just get permuted). This is why you must not feed these integers to a
  model as *numeric features*; for inputs you'd use [[one-hot-encoding]] or an
  [[embedding-layer]].
- **Integer label = compressed one-hot.** Cross-entropy with integer target $y$ equals
  $-\sum_k e_{y,k}\log p_k$ with the one-hot vector $\mathbf e_y$. PyTorch's
  `cross_entropy(logits, y)` accepts the integer form because the dot product with a one-hot just
  selects entry $y$.
- **Binary special case.** With $K=2$ and $y\in\{0,1\}$, the label is also a Bernoulli outcome, which
  is what lets a single-output sigmoid + [[binary-cross-entropy]] model work as well — the
  alternative the book mentions in its "Output layer nodes" sidebar (book p. 183).
- **Same mechanism as tokenization.** A vocabulary maps tokens to $\{0,\dots,50256\}$, and the LM
  head's 50,257 logits are indexed by the next-token ID exactly the way the 2 class logits are
  indexed by $y$. Classification is "next-token prediction with a 2-word vocabulary".

## Connections
- Builds on: [[token-id]], [[vocabulary]], [[vector]]
- Leads to: [[cross-entropy-loss]], [[classification-head]], [[one-hot-encoding]]
- Used by: [[text-classification]], [[classification-fine-tuning]], [[pytorch-dataset-and-dataloader]]
- Related: [[class-imbalance]], [[logits]], [[softmax]], [[argmax]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.2, book p. 174 (ham → 0, spam → 1; token-ID analogy)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.8, book p. 201 (mapping prediction back to "spam"/"not spam")
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.6, book pp. 270–271 (Listing A.5 toy labels; zero-based label NOTE)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.2, book p. 324 (same mapping before LoRA fine-tuning)
