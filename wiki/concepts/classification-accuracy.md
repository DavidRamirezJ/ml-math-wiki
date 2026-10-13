---
title: Classification Accuracy
type: concept
tags: [evaluation, classification, statistics, llm, pytorch]
difficulty: beginner
sources: [llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Classification Accuracy

> **One-liner:** Accuracy is just the share of examples your model labels correctly: right answers divided by total answers.

## 🟢 Intuition (Dummy)
It's the score on a multiple-choice test. You answered 100 questions and got 96
right, so your accuracy is 96%. A classifier gets graded the same way. For every text
message it says "spam" or "not spam", we check that against the true label, and we
count how many it got right.

One number you should always keep in mind is the **coin-flip baseline**. With two
classes that appear equally often, random guessing gets about **50%**. The book's
spam classifier, *before* any training, scores 46–49% (book p. 193). That's
basically a coin flip, so it hasn't learned anything yet. After fine-tuning it scores
about 96% on texts it has never seen (book p. 200).

## 🟡 Mechanics (Practitioner)

### The formula
$$\text{accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[\hat{y}^{(i)} = y^{(i)}\right]$$

Reading it symbol by symbol:
- $N$ = how many examples we evaluate.
- $\sum_{i=1}^{N}$ = "add up the following thing for example 1, example 2, … up to example $N$".
- $\hat{y}^{(i)}$ ("y-hat i") = the model's **predicted** label for example $i$. The little hat means "estimated".
- $y^{(i)}$ = the **true** label for example $i$. The superscript $(i)$ is just an index, not a power.
- $\mathbb{1}[\cdot]$ = the **indicator function**. It gives 1 if the statement inside the brackets is true, 0 if it's false.
- $\frac{1}{N}\sum$ = take the average. So accuracy is the average of a list of 0s and 1s, which is the fraction of 1s.

In the book's GPT classifier, the prediction is the [[argmax]] of the
[[last-token-pooling|last-token]] [[logits]]:

$$\hat{y}^{(i)} = \arg\max_{k \in \{0,1\}} z^{(i)}_k$$

- $z^{(i)}_k$ = the logit (raw score) for class $k$ on example $i$. Class 0 is "not spam", class 1 is "spam".
- $\arg\max_k$ = "the $k$ that gives the biggest value". You don't need [[softmax]]
  first, because softmax is [[monotonic-function|monotonic]]. It never changes which
  entry is largest (book p. 192).

Fig. 6.14 (book p. 191) shows the pipeline for two texts: last-token logits → softmax
probabilities → argmax. The untrained model gets both wrong ("You won the lottery" →
0, "Do you have time" → 1), which is what a ~50% accuracy looks like in practice. The
book defines classification accuracy as the measure of "the percentage of correct
predictions across a dataset" (book p. 192).

### Tiny example by hand (Appendix A)
Here's a toy network on 5 training points. It outputs probabilities like
`[0.9991, 0.0009]` for the first row and predicts `tensor([0, 0, 0, 1, 1])`
(book p. 277):

```python
predictions == y_train            # tensor([True, True, True, True, True])
torch.sum(predictions == y_train) # 5   -> 5/5 = 100% accuracy
```

`True` counts as 1, so summing the comparison *is* the indicator sum $\sum \mathbb{1}[\cdot]$.
The book then wraps this in **Listing A.10 `compute_accuracy(model, dataloader)`**: loop
over batches, `predictions = torch.argmax(logits, dim=1)`, add up
`torch.sum(labels == predictions)` and `len(compare)`, and return
`(correct / total_examples).item()`, a value between 0 and 1. On the toy data it returns
`1.0` for both the training and the test loader (book p. 278).

### Batch-wise accuracy: Listing 6.8 (book pp. 192–193)
Real datasets don't fit through the model all at once, so you count correct answers
batch by batch. Appendix A gives the reason: we "typically can only call the model on
a small part of the dataset due to memory limitations", and a batch-wise function
"scales to datasets of arbitrary size" (book p. 278). Chapter 6's version also slices
out the last token (book pp. 192–193):

```python
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]   # last-token logits [B, 2]
            predicted_labels = torch.argmax(logits, dim=-1)   # [B]
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples
```

Some details worth noticing:
- It adds up *counts* first and divides only once at the very end. That way, a
  smaller last batch doesn't get the same weight as a full one.
- `num_batches` lets you get a quick *estimate* from just the first few batches.
- [[train-vs-eval-mode|`model.eval()`]] and `torch.no_grad()` are there because we're
  only measuring here, not training.

### The book's numbers (spam classifier, GPT-2 small, 70/10/20 split = 1,045 / 149 / 300 texts)

| Moment | Train | Validation | Test | Where |
|---|---|---|---|---|
| Before fine-tuning (10 batches) | 46.25% | 45.00% | 48.75% | book p. 193 |
| After epoch 1 / 3 / 5 (5 batches, during training) | 70.00 / 90.00 / 100.00% | 72.50 / 90.00 / 97.50% | — | book p. 197 |
| After fine-tuning (full sets) | **97.21%** | **97.32%** | **95.67%** | book p. 200 |
| LoRA fine-tuning (Appendix E, full sets) | 100.00% | 96.64% | 98.00% | book p. 336 |

How to read this table:
- **Before training ≈ 50%** means the model is guessing. A balanced 2-class dataset has a
  chance level of $1/K = 1/2$ (book p. 193; dataset balanced to 747 + 747 in §6.2).
- **Train vs. test.** The book calls training and test performance "almost identical";
  the slight gap "suggests minimal overfitting" (book p. 200).
- **Validation is usually a bit higher than test.** The book explains that we tune
  [[hyperparameter|hyperparameters]] to do well on the validation set, and those settings
  might not carry over to the test set as well. The gap could be reduced by raising the
  dropout rate (`drop_rate`) or the optimizer's `weight_decay` (book p. 200).
- **LoRA starts from exactly the same accuracy.** Before LoRA training, the accuracies
  are again 46.25 / 45.00 / 48.75% (book pp. 328, 334). The book expects "around 50%"
  for the untrained model (book p. 327). They match chapter 6 exactly because LoRA's $B$
  matrix starts at zero, so $AB = 0$ and the model's outputs don't change (book p. 334).
  During LoRA training, the 5-batch estimates go from 100.00% / 92.50% (train / val,
  epoch 1) to 100.00% / 100.00% (epoch 5) (book pp. 334–335).
- **Train 100% with val/test a bit lower** suggests "a small degree of [[overfitting]]"
  (book p. 336, LoRA run).
- ⚠️ *Book inconsistency (Appendix E):* the printed output says test accuracy is
  98.00%, but the paragraph right after it says 97.33% (book p. 336).

### Where accuracy stops being useful (Chapter 7)
Accuracy only works when each output is one of a fixed set of labels. For
[[instruction-fine-tuning|instruction-tuned]] models that write free-form answers,
evaluation "is not as straightforward" as for classification fine-tuning, "where we
simply calculate the percentage of correct spam/non-spam class labels" (book p. 236).
The book lists three alternatives: short-answer and multiple-choice benchmarks such as
MMLU (where accuracy *does* still apply, per question), human preference comparisons
such as the LMSYS chatbot arena, and automated conversational benchmarks where another
LLM judges the answers, such as AlpacaEval (book p. 236). Chapter 7 uses the last kind:
[[llm-as-a-judge]] scoring.

## 🔴 Theory (Expert)
- **Accuracy is one minus the average 0-1 loss.** Define the zero-one loss
  $\ell_{0\text{-}1}(\hat y, y) = \mathbb{1}[\hat y \ne y]$. Then
  $\text{accuracy} = 1 - \frac{1}{N}\sum_i \ell_{0\text{-}1}$, which is one minus the
  *empirical risk* under 0-1 loss ([[empirical-risk-minimization]]). This is the link
  to [[surrogate-loss]]: we can't train on accuracy directly, because it's piecewise
  constant in $\theta$, so we train on [[cross-entropy-loss]] instead (book p. 193).
- **It's an estimate of a probability.** On a test set drawn independently from the
  data distribution, accuracy estimates $P(\hat Y = Y)$ using an
  [[expected-value-and-mean|average]] of Bernoulli outcomes. The count of correct
  answers follows $\text{Binomial}(N, p)$, so the standard error is
  $\text{SE} = \sqrt{p(1-p)/N}$.
  - Test set, $N = 300$, $p \approx 0.957$: $\text{SE} \approx \sqrt{0.957 \cdot 0.043 / 300} \approx 0.012$, so about **±1.2 points**.
    The 95.67% (287/300) vs. 96.67% (290/300) gap from Exercise 6.2 (book p. 306) is
    about one SE. That's suggestive but not conclusive.
  - During training, `eval_iter=5` batches × 8 = 40 examples. So accuracy moves in
    steps of $1/40 = 2.5\%$ (70.00, 72.50, 97.50, …) and has an SE of up to about 8
    points at $p = 0.5$. That's why the book calls these numbers approximations and
    recomputes on the full sets afterwards (book pp. 197, 200). (The arithmetic here
    is ours, not the book's.)
  - The full-set percentages pin down the counts exactly: 97.21% = 1011/1040 (the train
    loader drops the last partial batch, so 130 × 8 = 1040), 97.32% = 145/149,
    95.67% = 287/300.
- **Accuracy and class imbalance.** On the original SMS data (4,825 ham vs. 747 spam,
  book p. 173), a model that *always* says "not spam" already scores 86.6%. Accuracy
  hides that kind of failure, which is one reason the book balances the classes
  first ([[class-imbalance]]). Other metrics like precision, recall, F1 or balanced
  accuracy fix this, but they aren't covered in the book.
- **Invariance.** Accuracy only depends on the argmax, so it doesn't change under any
  strictly increasing transformation of the logits, like softmax, temperature, or
  adding a constant to every logit. It says nothing about *calibration*, meaning
  whether a "0.99" really is right 99% of the time.

## Connections
- Builds on: [[argmax]], [[logits]], [[softmax]], [[monotonic-function]], [[expected-value-and-mean]], [[train-validation-test-split]]
- Leads to: [[surrogate-loss]], [[baseline-evaluation]], [[llm-evaluation]]
- Related: [[empirical-risk-minimization]], [[cross-entropy-loss]], [[class-imbalance]], [[overfitting]], [[classification-fine-tuning]], [[last-token-pooling]], [[llm-as-a-judge]], [[cross-validation]]
- Used by: [[text-classification]], [[training-loop]], [[low-rank-adaptation]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6, book pp. 190–194 (softmax/argmax to labels, definition, Listing 6.8, initial 46.25/45.00/48.75%)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7, book pp. 195–200 (accuracy per epoch, Fig. 6.17, final 97.21/97.32/95.67%, validation-vs-test remark)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.8, book pp. 200–203 (chapter summary: evaluation = classification accuracy)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.7, book p. 236 (why accuracy doesn't apply to free-form responses)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7, book pp. 276–278 (accuracy by hand, Listing A.10 `compute_accuracy`, result 1.0 on train and test)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.3–E.4, book pp. 327–328, 333–336 (expected ~50% and 46.25/45.00/48.75% baseline before LoRA; why it is unchanged after inserting LoRA; per-epoch estimates; 100.00/96.64/98.00% after; 97.33% text inconsistency)
- Derived (not in the book): standard-error arithmetic, exact count reconstructions, the 86.6% majority-class baseline.
