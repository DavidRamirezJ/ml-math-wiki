---
title: Supervised Learning (Training vs Inference)
type: concept
tags: [machine-learning, training, inference, llm, foundations]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-appA-introduction-to-pytorch, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Supervised Learning (Training vs Inference)

> **One-liner:** Supervised learning means teaching a model with examples that come with the right answers, then using it to guess the answers for new examples.

## 🟢 Intuition (Dummy)
Think of flashcards. On the front is a question, on the back is the answer. You study a
big stack: look at the front, guess, flip, see how wrong you were, adjust. That is the
**training stage**. Later someone hands you a brand-new card with no back side, and you
answer it from what you learned. That is the **inference stage**.

The book's running example is a spam filter (book pp. 3, 253). The flashcards are emails,
and someone has written "spam" or "not spam" on the back of each one. Those answers are
called **labels**. Instead of a programmer writing rules like "if it says *free prize*,
it's spam", the model figures out the patterns itself by trying to make fewer mistakes on
the labelled emails. Then it can sort new emails it has never seen.

There is one more step that people forget: **check before you trust it**. Between training
and real use you should test the model on examples it did not train on, to make sure it
is actually good enough (book pp. 253–254).

How does this connect to LLMs? Two ways:
- [[classification-fine-tuning]] (Ch. 6) is plain supervised learning: texts plus class labels.
- [[pretraining]] looks different (no human labels), but it uses the same machinery. The
  "label" for each position is just the next word in the text, so the text labels itself.
  This is [[self-supervised-learning]] (book p. 6).

## 🟡 Mechanics (Practitioner)

### The two stages (Figure A.3, book p. 254)
- **Training:** a learning algorithm fits a model on a **training dataset** of examples
  and their labels.
- **(Evaluation):** measure performance on held-out data (see [[train-validation-test-split]]).
- **Inference:** the trained model predicts labels for **new observations**.

### The math of training
You have $N$ labelled pairs $\{(\mathbf{x}_n, y_n)\}_{n=1}^{N}$ and a model $f_\theta$. Training solves

$$\theta^{*} = \arg\min_{\theta}\ \frac{1}{N}\sum_{n=1}^{N} \ell\big(f_\theta(\mathbf{x}_n),\, y_n\big)$$

Symbol by symbol:
- $\mathbf{x}_n$ = the $n$-th input (e.g. one email turned into numbers).
- $y_n$ = its label (e.g. 0 = not spam, 1 = spam).
- $f_\theta$ = the model, e.g. a [[neural-network]]; $\theta$ = its adjustable [[model-parameters]].
- $f_\theta(\mathbf{x}_n)$ = the model's prediction for that input.
- $\ell(\text{prediction}, \text{label})$ = the [[loss-function]]: a number saying how wrong the prediction was.
- $\frac{1}{N}\sum_{n=1}^{N}$ = average over all $N$ training examples. This average is the "empirical risk".
- $\arg\min_\theta$ = "the setting of $\theta$ that makes this average as small as possible". $\theta^{*}$ is that best setting.

That is exactly the book's sentence "by minimizing the error in its predictions on a
training dataset, the model learns patterns" (book p. 3), written in symbols. It is
[[empirical-risk-minimization]]. In practice you do not minimize over all $N$ at once:
you use [[mini-batch|mini-batches]] and [[stochastic-gradient-descent]].

### The math of inference
$$\hat{y} = f_{\theta^{*}}(\mathbf{x}_{\text{new}})$$
- $\mathbf{x}_{\text{new}}$ = a fresh input the model never saw.
- $f_{\theta^{*}}$ = the model with its trained (frozen) parameters.
- $\hat{y}$ ("y-hat") = the predicted label. The hat means "estimate", not the true value.

For a classifier the network outputs [[logits]] $\mathbf{z}$, and the predicted class is
$\hat y = \arg\max_k z_k$ (see [[argmax]]).

### Worked example from the book (Appendix A)
Toy dataset (Listing A.5, book p. 270): 5 training examples with 2 features each, labels
`[0, 0, 0, 1, 1]`, plus 2 test examples with labels `[0, 1]`:

```python
X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6],
                        [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test  = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
y_test  = torch.tensor([0, 1])
```

Training a small MLP with SGD (learning rate 0.5, 3 epochs) drops the training loss from
0.75 to 0.00 (book p. 275). Then, at inference, `compute_accuracy` gives **1.0** on both
the training and the test loader (book p. 278). Note the PyTorch rule: class labels must
start at 0 and go up to (number of output nodes − 1) (book p. 271).

In PyTorch the two stages even have their own switches: `model.train()` for training mode
and `model.eval()` for evaluation/inference mode, which matter for layers like [[dropout]]
(book pp. 275–276; see [[train-vs-eval-mode]]).

### How LLM training fits (book pp. 6–7, 254)
| | Inputs $\mathbf{x}$ | Labels $y$ | Where in the book |
|---|---|---|---|
| Classification fine-tuning | text message | spam / not spam | Ch. 6 |
| Instruction fine-tuning | instruction (+input) | the desired answer text | Ch. 7 |
| Pretraining | first $t$ tokens | token $t+1$ (taken from the text itself) | Ch. 2, 5 |

At inference an LLM is given a prompt and generates new text, rather than one label
(book p. 254).

## 🔴 Theory (Expert)
- **Statistical framing.** Assume pairs $(\mathbf{x}, y)\sim p_{\text{data}}$. The goal is low
  *true risk* $R(\theta)=\mathbb{E}_{(\mathbf{x},y)\sim p_{\text{data}}}[\ell(f_\theta(\mathbf{x}),y)]$
  (an [[expected-value-and-mean|expectation]]), but only the empirical average $\hat R_N(\theta)$ is
  computable. The gap $R-\hat R_N$ is the generalisation gap, which [[overfitting]] makes large.
  That is why the book insists on evaluation between training and inference, and on a test set
  used "precisely once" (book p. 275; see [[cross-validation]]).
- **Probabilistic view.** With a softmax output and [[cross-entropy-loss]], minimising empirical
  risk equals maximising $\sum_n \log p_\theta(y_n\mid\mathbf{x}_n)$, i.e. conditional
  [[maximum-likelihood-estimation]] of a discriminative model.
- **Self-supervision as supervision.** Pretraining factorises
  $p_\theta(x_1,\dots,x_T)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$ and builds pairs
  $(x_{<t}, x_t)$ from raw text, so every token position becomes a supervised example. Mathematically
  it is the same objective with a $V$-way classifier ($V$ = vocabulary size); the only difference is
  where the labels come from. The book says "self-supervised" on p. 6 (NOTE) and describes the
  "labels derived from the text itself" idea on p. 254.
- **Train/inference asymmetry in LLMs.** During training all positions are predicted in parallel
  with teacher forcing (true previous tokens as context); at inference the model is
  [[autoregressive-text-generation|autoregressive]] and feeds its own predictions back in. This is a
  source of exposure bias and the reason decoding strategies (Ch. 5) matter.
- **Labels are the cost.** The book notes both traditional ML and deep learning still need
  collected labels for spam classification (book p. 4). Self-supervised pretraining sidesteps that
  for the expensive stage and leaves only a small labelled dataset for [[fine-tuning]] (book pp. 6–7).

### From llm-ch06-fine-tuning-for-classification
Ch. 6 is a full supervised-learning run on real data. Each training example is a pair $(\mathbf x_n, y_n)$: $\mathbf x_n$ is the token-ID sequence of one SMS message, and $y_n \in \{0,1\}$ is its label (ham → 0, spam → 1, see [[label-encoding]]). The book balances the 747 spam and 4,825 ham messages ([[class-imbalance]]), splits them 70/10/20 ([[train-validation-test-split]]), and minimizes [[cross-entropy-loss]] on the training part. The goal it reports is [[classification-accuracy]] on the held-out test part: 95.67% (§6.2, book pp. 172–175; §6.7, p. 200). The book calls this "classification fine-tuning, a concept you might already be acquainted with if you have a background in machine learning" (§6.1, p. 170).

## Connections
- Builds on: [[machine-learning]], [[loss-function]], [[empirical-risk-minimization]], [[probability-distribution]]
- Leads to: [[classification-fine-tuning]], [[instruction-fine-tuning]], [[fine-tuning]], [[text-classification]], [[llm-training-loop]]
- Contrast with: [[self-supervised-learning]], [[pretraining]], [[next-token-prediction]]
- Uses: [[neural-network]], [[stochastic-gradient-descent]], [[mini-batch]], [[cross-entropy-loss]], [[train-validation-test-split]], [[classification-accuracy]]
- Risks: [[overfitting]], [[overfitting-and-underfitting]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1, book pp. 3–4 (ML definition; spam filter learned by minimizing prediction error on a training dataset; labels still required).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 6–7 (supervised vs self-supervised learning NOTE; fine-tuning on labelled data).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1.2, book pp. 253–254 (training vs inference, model evaluation, Figure A.3, application to LLMs).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.6–A.7, book pp. 270–278 (toy dataset, label convention, training output, accuracy 1.0).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.1–6.2, book pp. 170–175; §6.7, p. 200 (labeled SMS spam dataset, supervised fine-tuning, test accuracy) — see [[llm-ch06-fine-tuning-for-classification]].
