---
title: Class Imbalance and Undersampling
type: concept
tags: [classification, data-preparation, sampling, statistics]
difficulty: beginner
sources: [llm-ch06-fine-tuning-for-classification, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Class Imbalance and Undersampling

> **One-liner:** When one label is far more common than another, a model can look smart just by always guessing the common one, so we even out the classes first, for example by throwing away extra examples of the big class (undersampling).

## 🟢 Intuition (Dummy)
Imagine studying for a quiz where 87 out of every 100 questions have the answer "no". A lazy
student who answers "no" to everything scores 87 %, without learning anything at all.

That's **class imbalance**: one class (here "ham", normal messages) shows up way more than the
other ("spam"). A model trained on that data can drift toward just shouting the majority label.

**Undersampling** is the simplest fix: keep all the rare examples, and randomly keep only as many
of the common ones as you have rare ones. Now it's a 50/50 quiz, and the lazy strategy only
scores 50 %. To do better, the model has to actually learn what spam looks like.

## 🟡 Mechanics (Practitioner)

### The book's numbers
The UCI SMS Spam Collection has 5,572 messages (book pp. 173–174):

```python
print(df["Label"].value_counts())
# ham     4825
# spam     747
```

- Ratio: $4825/747\approx6.46$ ham messages for every spam message (derived).
- Spam share: $747/5572\approx13.4\%$ (derived).
- "Always say ham" accuracy: $4825/5572\approx86.6\%$ (derived) — high, yet useless.

### Undersampling, step by step (Listing 6.2, book p. 174)
```python
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]                 # 747
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df
# ham 747, spam 747  → 1,494 rows
```
- Count the minority class (747 spam).
- Randomly draw that many ham messages, **without replacement**; `random_state=123` makes the draw
  repeatable ([[random-seed-reproducibility]]).
- Glue them together. Note `pd.concat` puts all ham first and all spam after; the shuffle happens
  later in `random_split` (book p. 175).

The book picks undersampling "for simplicity" and because a smaller dataset makes fine-tuning
faster (book p. 174). It mentions that other methods exist and points to the *Imbalanced-Learn
User Guide* in appendix B (book pp. 174, 296). Appendix E reuses the same
`create_balanced_dataset` helper before LoRA fine-tuning (book p. 324).

### The math: class prior
The **class prior** is just the fraction of each class in the data:

$$\hat p(y=k)=\frac{N_k}{N}$$

- $\hat p(y=k)$ = our estimate (the hat means "estimated from data") of the chance a random
  example has label $k$.
- $N_k$ = number of examples with label $k$ (e.g. $N_\text{spam}=747$).
- $N$ = total number of examples (5,572).

Before balancing: $\hat p(\text{spam})=747/5572\approx0.134$. After undersampling:
$\hat p(\text{spam})=747/1494=0.5$.

This is why the book can read its untrained accuracies (46.25 % / 45.00 % / 48.75 %) as "about
random": on a balanced 2-class set, random guessing gives 50 % (book p. 193).

### Common alternatives (not used in the book)
- **Oversampling** the minority class (repeat or synthesize spam examples).
- **Class weights** in the loss: multiply each example's loss by $w_{y}$, larger for the rare class.
- **Better metrics**: precision, recall, F1 instead of plain accuracy.

## 🔴 Theory (Expert)
- **Prior shift.** By [[bayes-theorem|Bayes' rule]],
  $p(y\mid\mathbf x)\propto p(\mathbf x\mid y)\,p(y)$. Undersampling leaves the class-conditional
  $p(\mathbf x\mid y)$ alone but changes $p(y)$. A classifier trained on balanced data therefore
  estimates $p_\text{bal}(y\mid\mathbf x)$. If test data keeps the true prior $\pi_k$, the
  corrected posterior is
  $p(y=k\mid\mathbf x)\propto p_\text{bal}(y=k\mid\mathbf x)\,\pi_k/(1/K)$ — equivalently, add
  $\log\pi_k-\log(1/K)$ to logit $k$. The book doesn't do this; its test split is a random 20 % slice of
  the 50/50 set (so only approximately balanced), and the reported 95.67 % test accuracy is
  therefore a *(near-)balanced-set* accuracy.
- **Information loss.** Undersampling discards $4825-747=4078$ real ham messages (about 73 % of
  the data). With a pretrained LLM the cost is small because most of the "knowing English" comes
  from [[pretraining]]; for training from scratch it can hurt more.
- **Equivalence to reweighting.** In expectation, keeping each majority example with probability
  $r=747/4825$ is the same as weighting the majority-class loss by $r$ in
  [[empirical-risk-minimization]] — but with higher variance, since fewer samples are used.
- **Why accuracy misleads.** Accuracy is the empirical mean of the 0–1 loss; under imbalance it is
  dominated by the majority class. Balanced accuracy $\frac1K\sum_k\text{recall}_k$ equals plain
  accuracy exactly when the evaluation set is balanced.

## Connections
- Builds on: [[probability-distribution]], [[expected-value-and-mean]], [[histogram]]
- Leads to: [[classification-accuracy]], [[train-validation-test-split]], [[label-encoding]]
- Used by: [[text-classification]], [[classification-fine-tuning]], [[low-rank-adaptation]]
- Related: [[bayes-theorem]], [[empirical-risk-minimization]], [[random-seed-reproducibility]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.2, book pp. 173–175 (value counts, Listing 6.2, undersampling rationale)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6, book p. 193 (near-random initial accuracy vs 50 % baseline)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.2, book p. 324 (reused `create_balanced_dataset`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book p. 296 ("Imbalanced-Learn User Guide", mng.bz/2KNa)
