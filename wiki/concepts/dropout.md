---
title: Dropout
type: concept
tags: [deep-learning, regularization, training, llm]
difficulty: beginner
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Dropout

> **One-liner:** Dropout randomly switches off some of a network's numbers during training so it can't lean too hard on any one of them — and switches the randomness off when you actually use the model.

## 🟢 Intuition (Dummy)
Imagine a study group where one student always answers every question. The others
never learn. A clever teacher starts randomly sending a few students out of the room
each day. Now everyone has to be able to cope, and the group as a whole gets
sturdier. On exam day, everyone is back in the room.

**Dropout** does that to a neural network. During training, each time data flows
through, a random set of units (numbers inside the network) is set to zero. The
network can't rely on one special unit, so it learns more robust patterns and
memorises the training data less — it [[overfitting|overfits]] less. When training
is done, dropout is turned **off**: the full network is used.

One small detail: when half the students are out, the ones still in the room "speak
louder" (are scaled up) so the total amount of talking stays about the same.

## 🟡 Mechanics (Practitioner)
**Definition (book p. 78).** Dropout is a deep-learning technique where randomly
selected hidden-layer units are **ignored during training**, effectively "dropping"
them out. It prevents overfitting by ensuring the model doesn't become overly reliant
on any specific set of hidden units. It is **only used during training** and disabled
afterward.

**The formula ("inverted dropout"):**
$$\tilde a=\frac{m}{1-p}\,a,\qquad m\sim\operatorname{Bernoulli}(1-p)$$
- $a$ = an original value (a hidden unit, or an attention weight).
- $p$ = the **dropout rate**, the chance a value is dropped (e.g. 0.5 or 0.1).
- $m$ = a random coin flip: 1 ("keep") with probability $1-p$, 0 ("drop") with probability $p$ ([[random-variable]]).
- $\frac{1}{1-p}$ = the "speak louder" factor for survivors. With $p=0.5$ it is $1/0.5=2$.
- $\tilde a$ = the value after dropout: either 0 or $a/(1-p)$.

Why the scaling? The [[expected-value-and-mean|expected value]] stays unchanged:
$$\mathbb E[\tilde a]=\frac{(1-p)\cdot a}{1-p}+p\cdot 0=a$$
- $\mathbb E[\cdot]$ = "the average over many random coin flips".

The book's words: scaling keeps the overall balance so the **average influence is
consistent during training and inference** (book pp. 79–80). (The Bernoulli/expectation
algebra is our addition.)

**Demo on a 6×6 matrix of ones (book pp. 78–79):**
```python
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))
# tensor([[2., 2., 0., 2., 2., 0.],
#         [0., 0., 0., 2., 0., 2.],
#         [2., 2., 2., 2., 0., 2.],
#         [0., 2., 2., 0., 0., 2.],
#         [0., 2., 0., 2., 0., 2.],
#         [0., 2., 2., 2., 2., 0.]])
```
About half the entries are 0; survivors became $1\times2=2$.

**Dropout on attention weights (book pp. 78–80).** In transformers, attention dropout
usually goes either (a) **after computing the attention weights** or (b) after
applying them to the value vectors; the book uses (a), the more common variant
(book p. 78). Applied to the causal weights from [[causal-attention]]:
```python
torch.manual_seed(123)
print(dropout(attn_weights))
# [[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#  [0.7599, 0.6194, 0.6206, 0.0000, 0.0000, 0.0000],
#  [0.0000, 0.4921, 0.4925, 0.0000, 0.0000, 0.0000],
#  [0.0000, 0.3966, 0.0000, 0.3775, 0.0000, 0.0000],
#  [0.0000, 0.3327, 0.3331, 0.3084, 0.3331, 0.0000]]
```
E.g. $0.3800\times2\approx0.7599$. Rows no longer sum to 1 (row 2 is all zeros) — that's
expected during training. The book warns that results can differ by operating system
(PyTorch issue #121595, book p. 80). The demo rate is 50%; GPT training uses lower
rates such as **0.1 or 0.2** (book p. 78).

**Where dropout lives in the GPT.** Three places (Exercise 4.3, book p. 127;
solution pp. 302–304):
1. **Embeddings** — `drop_emb` after adding token + positional embeddings (`DummyGPTModel`, book pp. 96–97; `GPTModel`, book p. 119).
2. **Attention weights** — `self.dropout` inside [[multi-head-attention]] (book p. 87).
3. **Shortcut path** — `drop_shortcut` on each sub-layer's output before the residual add (book p. 115):
$$\mathbf h=\mathbf x+\operatorname{Dropout}\big(\operatorname{MHA}(\operatorname{LN}_1(\mathbf x))\big),\qquad
\mathbf y=\mathbf h+\operatorname{Dropout}\big(\operatorname{FFN}(\operatorname{LN}_2(\mathbf h))\big)$$
   - $\mathbf x$ = block input; $\operatorname{LN}$ = [[layer-normalization]]; $\operatorname{MHA}$ = masked multi-head attention; $\operatorname{FFN}$ = [[feed-forward-network]]; the `+` is the [[shortcut-connections|shortcut]].

**Rates used across the book.**

| Where | `drop_rate` | Page |
|---|---|---|
| Ch. 3 demo | 0.5 | p. 78 |
| `GPT_CONFIG_124M` (Ch. 4–5) | 0.1 — "10% of hidden units are dropped at random" | pp. 95, 130 |
| Ch. 5 note | "possible and common to set dropout to 0" | p. 130 |
| Ch. 6 classification fine-tuning | 0.0 | p. 182 |
| Ch. 7 instruction fine-tuning | 0.0 | p. 227 |

To shrink a train/validation gap during classification fine-tuning, the book suggests
**raising `drop_rate`** or `weight_decay` — both [[regularization]] methods (book p. 200).

**Train vs. eval mode.** `model.train()` turns dropout on; `model.eval()` turns it off
([[train-vs-eval-mode]]). The book calls `model.eval()` before generation (book p. 126),
inside the evaluation helpers for "stable, reproducible results" (book pp. 147–148),
and after loading weights: "during inference we don't want to randomly drop out any
information the network has learned" (book p. 160). Appendix A notes these calls are
needed for layers like dropout or batch normalization, and are good practice even in
models without them (book pp. 275–276).

## 🔴 Theory (Expert)
- **Origin.** Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov (2014),
  "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 15 —
  cited in appendix B (book p. 292). ([[nitish-srivastava]], [[geoffrey-hinton]].)
- **Implicit ensemble.** A network with $n$ droppable units defines $2^n$ thinned
  sub-networks sharing weights; training samples one per step. Inference with the
  full net (inverted scaling) approximates the geometric-mean ensemble — exact for a
  single linear layer followed by softmax, an approximation otherwise
  ([[monte-carlo-estimation]] if you instead average stochastic passes, "MC dropout").
- **Regularisation effect.** For linear regression $\mathbf y\approx X\mathbf w$
  with inverted dropout on the inputs, the expected loss is
  $\lVert\mathbf y-X\mathbf w\rVert^2+\tfrac{p}{1-p}\sum_j w_j^2\,\lVert X_{:,j}\rVert^2$ —
  the clean loss plus a feature-scaled $L_2$ penalty, so dropout acts like
  data-dependent [[weight-decay]] ([[regularization]]).
- **Variance mismatch.** Inverted dropout preserves the mean but not the variance:
  $\operatorname{Var}[\tilde a]=\tfrac{p}{1-p}a^2$. This train/test variance shift is
  one reason dropout interacts awkwardly with [[batch-normalization]], and a reason
  [[layer-normalization]] + low rates are preferred in transformers.
- **Why LLMs often use 0.** Large-scale pretraining usually sees each token about once,
  so overfitting is less of a risk than under-fitting; many modern LLMs set dropout to
  0, matching the book's remark (book p. 130) and the fine-tuning configs above.
- **Attention dropout semantics.** Dropping an attention weight $\alpha_{ij}$ removes
  token $j$'s contribution to token $i$'s context vector for that step — a form of
  random "edge dropout" on the attention graph; it never leaks future information,
  since masked weights are already 0.

## Connections
- Builds on: [[overfitting]], [[regularization]], [[random-variable]], [[expected-value-and-mean]], [[neural-network]]
- Used by: [[causal-attention]], [[multi-head-attention]], [[transformer-block]], [[gpt-architecture]], [[gpt-model-configuration]]
- Related: [[train-vs-eval-mode]], [[weight-decay]], [[batch-normalization]], [[layer-normalization]], [[shortcut-connections]], [[hyperparameter]]
- People: [[nitish-srivastava]], [[geoffrey-hinton]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1 intro, book p. 50; §3.5.2–3.5.3, book pp. 78–82 (Fig. 3.22, Listing 3.3); §3.6.2, book p. 87; §3.6 summary, book p. 91.
- §4.1, book pp. 95–97; §4.5, book pp. 114–115; §4.6, book pp. 118–119; §4.7, book pp. 126–127 (Exercise 4.3).
- §5.1, book p. 130; §5.2, book pp. 147–148; §5.3, book p. 151; §5.4, book p. 160; §5.5, book p. 164.
- §6.4, book p. 182; §6.7, book pp. 196–197, 200; §6.8, book p. 201 (`model.eval()` before classifying new text); §7.5, book p. 227; appendix A.7, book pp. 274–276.
- Appendix B Ch. 3 ref. 7 (Srivastava et al. 2014), book p. 292; appendix C Exercise 4.3 solution, book pp. 302–304.
- The Bernoulli/expectation algebra, ensemble view, and variance results are standard background, not printed in the book.
