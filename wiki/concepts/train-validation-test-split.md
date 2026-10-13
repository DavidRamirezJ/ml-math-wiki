---
title: Train/Validation/Test Split
type: concept
tags: [machine-learning, generalization, model-selection, evaluation, llm]
difficulty: beginner
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Train/Validation/Test Split

> **One-liner:** Before training, you cut your data into three piles — one to learn from, one to tune your choices on, and one locked away for a single honest final exam.

*Also known as:* train-test split, training/validation split, held-out test set, validation set.

## 🟢 Intuition (Dummy)
Imagine you're preparing a student for a big exam using a box of old exam questions.

- **Training pile** — the questions the student actually practices with. They see these over and over.
- **Validation pile** — questions you use to *check in* every so often: "Is the study plan working? Should we study longer? Change the method?" You peek at these scores many times.
- **Test pile** — questions sealed in an envelope. You open it **once**, at the very end, to get an honest score.

Why not just check the student on the practice questions? Because they might have **memorized** the answers without understanding anything. And why keep the test pile separate from the validation pile? Because once you start making decisions based on the validation scores, those scores become a little bit "yours" — you tuned the plan to look good on them. The sealed test pile is the only score nobody has fiddled with.

That's exactly what Raschka does in *Build a Large Language Model (From Scratch)* — and he does it in every chapter where a model is trained.

## 🟡 Mechanics (Practitioner)

### The three roles
The book gives the cleanest definition in Appendix A (§A.7, book p. 275):

> "In practice, we often use a third dataset, a so-called validation dataset, to find the optimal hyperparameter settings. A validation dataset is similar to a test set. However, while we only want to use a test set precisely once to avoid biasing the evaluation, we usually use the validation set multiple times to tweak the model settings."

| Split | What touches it | How often you look | Book's phrase (§6.2, p. 175) |
|---|---|---|---|
| Training | gradients, `optimizer.step()` | every step | "train" |
| Validation | [[hyperparameter]] choices, monitoring [[overfitting]] | many times | "adjust" |
| Test | nothing — final report only | **once** | "evaluate" |

### The split formula
Every split in the book is the same trick: compute cut points with integer truncation, then slice.

$$n_{\text{train}} = \lfloor f_{\text{train}}\, N \rfloor,\qquad n_{\text{val}} = \lfloor f_{\text{val}}\, N \rfloor,\qquad n_{\text{test}} = N - n_{\text{train}} - n_{\text{val}}$$

Reading it symbol by symbol:
- $N$ = how many examples (or characters) you have in total — `len(df)` or `len(text_data)`.
- $f_{\text{train}}$, $f_{\text{val}}$ = the fractions you choose, e.g. 0.7 and 0.1.
- $\lfloor \cdot \rfloor$ = "floor": round **down** to a whole number. In Python, `int(...)` does this for positive numbers.
- $n_{\text{test}}$ = whatever is left over. This is why the book says the test size is "implied" as the remainder.

### The four splits in the book

**1. Pretraining (Ch. 5, §5.1.3, book pp. 141–144) — 90 / 10, no test set.** The corpus is the short story "The Verdict" (20,479 characters, 5,145 tokens). It is split at the *character* level on the raw string:

```python
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data   = text_data[split_idx:]
```

With `batch_size=2` and `max_length = stride = 256`, this gives **9 training batches** and just **1 validation batch**, each of shape `[2, 256]` (book p. 144). Untrained, both losses are about 11: training **10.98758347829183**, validation **10.98110580444336** (book p. 145). Appendix D reuses exactly this 90/10 split (§D.1, book p. 314).

**2. Spam classification (Ch. 6, §6.2, book p. 175) — 70 / 10 / 20, shuffled.** Listing 6.3:

```python
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # shuffle
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
```

Worked numbers with the balanced $N = 1{,}494$ messages (747 ham + 747 spam): $\lfloor 0.7 \times 1494 \rfloor = \lfloor 1045.8\rfloor = 1045$ train, $\lfloor 0.1\times1494\rfloor = \lfloor 149.4\rfloor = 149$ validation, $1494 - 1045 - 149 = 300$ test. With batch size 8 that's **130 / 19 / 38 batches** (§6.3, book p. 181; the training loader uses `drop_last=True`, so $\lfloor 1045/8\rfloor = 130$, while $\lceil 149/8\rceil = 19$ and $\lceil 300/8\rceil = 38$). The three set sizes themselves are not printed in the book; they follow from the code *(derived)*. Appendix E reuses this exact split for LoRA (§E.2, book p. 324).

**3. Instruction fine-tuning (Ch. 7, §7.2, book p. 210) — 85 / 10 / 5, contiguous slices.** Listing 7.3:

```python
train_portion = int(len(data) * 0.85)   # 85% training
test_portion  = int(len(data) * 0.1)    # 10% testing
val_portion   = len(data) - train_portion - test_portion  # remaining 5% validation
train_data = data[:train_portion]
test_data  = data[train_portion:train_portion + test_portion]
val_data   = data[train_portion + test_portion:]
```

With $N = 1{,}100$ instruction–response pairs: $\lfloor 935.0\rfloor = 935$ train, $\lfloor 110.0 \rfloor = 110$ test, $1100 - 935 - 110 = 55$ validation. Notice the slice order is train → **test** → validation, so here the *validation* set is the remainder. The 110 held-out test instructions are later answered by the fine-tuned model (§7.7, book pp. 233–238) and scored 0–100 by a larger LLM, Llama 3 8B Instruct via Ollama (§7.8, book pp. 238–247; see [[llm-as-a-judge]]).

**4. Toy example (Appendix A, §A.6, book p. 270).** 5 training examples and 2 test examples, each with 2 features — just enough to show that a training `DataLoader` gets `shuffle=True` while a test `DataLoader` doesn't need it.

### What the book does with each split
- **Training split** → `shuffle=True`, `drop_last=True` (book pp. 143, 180, 225).
- **Validation/test splits** → `shuffle=False`, `drop_last=False` — fixed order, every example counted.
- **Padding:** in Ch. 6, validation and test messages are padded/truncated to the *training* set's longest length (120 tokens), so all three splits share one shape (book p. 178).

### Reading the final scores (Ch. 6, §6.7, book p. 200)
After fine-tuning GPT-2 small as a spam classifier, the book evaluates on the **entire** split each time:

| Split | Accuracy |
|---|---|
| Training | 97.21% |
| Validation | 97.32% |
| Test | 95.67% |

Two lessons the book draws:
1. Train ≈ test → **minimal overfitting**.
2. "Typically, the validation set accuracy is somewhat higher than the test set accuracy because the model development often involves tuning hyperparameters to perform well on the validation set" ([[hyperparameter]]). To shrink the gap, increase `drop_rate` ([[dropout]]) or `weight_decay` ([[weight-decay]]) — both [[regularization]].

**The rule of thumb can flip.** The same split re-used for LoRA fine-tuning (§E.4, book p. 336) gives train **100.00%**, validation **96.64%**, test **98.00%** — here *test beats validation*. The book reads it as "a small degree of overfitting" (train > both held-out sets). (The book's prose quotes the test value as "97.33%", which does not match its own printed output of 98.00%.) With only 149 validation and 300 test messages, differences of 1–2 points are within sampling noise — see the 🔴 tier.

## 🔴 Theory (Expert)

**What each split estimates.** Let $R(f) = \mathbb{E}_{(\mathbf{x},y)}[\ell(y, f(\mathbf{x}))]$ be the true risk ([[empirical-risk-minimization]]). For any fixed predictor $f$ and a sample $S$ drawn independently of $f$, the empirical risk $\hat R_S(f) = \frac{1}{|S|}\sum_{i\in S}\ell(y_i,f(\mathbf x_i))$ is an **unbiased** estimate of $R(f)$. The catch is the word *independent*:

- On the **training set**, $f$ was chosen to minimize $\hat R_{\text{train}}$, so $\mathbb E[\hat R_{\text{train}}(\hat f)] \le R(\hat f)$ — optimistically biased. That's why the book's training loss falls to 0.391 while validation sits at 6.452 (§5.2, book p. 149).
- On the **validation set**, each single check is unbiased, but once you *pick* the best of $M$ configurations, $\min_m \hat R_{\text{val}}(f_m)$ is again biased low (the "winner's curse"). This is the mechanism behind the book's "validation set accuracy is somewhat higher than the test set accuracy" (book p. 200).
- The **test set** stays unbiased only if it's used once — the book's "precisely once" rule (book p. 275). This is the same logic as nested [[cross-validation]].

**How noisy is a test score?** For accuracy $\hat p$ on $n$ independent examples, the standard error is $\sqrt{\hat p(1-\hat p)/n}$ (a binomial / [[random-variable]] argument). For the Ch. 6 test set: $\sqrt{0.9567 \times 0.0433 / 300} \approx 0.0118$, i.e. about ±1.2 percentage points. For validation ($n=149$): $\approx 0.0132$. The val–test gap of 1.65 points is under one combined standard error ($\sqrt{0.0118^2+0.0132^2}\approx 0.0177$), so on this data it's consistent with pure noise as well as with selection bias. *(Derived; not computed in the book.)* Likewise, the in-training accuracies use only `eval_iter=5` batches × 8 = 40 examples, so they move in 2.5% steps (book pp. 197, 199–200).

**Contiguous vs shuffled splits.** Ch. 6 shuffles before slicing (`random_state=123`); Ch. 5 and Ch. 7 slice without shuffling. For running text (Ch. 5), a contiguous cut keeps the sliding-window chunks of the validation text from overlapping with training chunks — shuffling characters or windows first would leak near-duplicate context across the boundary. *(Interpretation, not stated in the book.)*

**Contamination.** The split only means something if the held-out data really is unseen. Exercise 5.5 (Appendix C, book pp. 305–306) evaluates the *pretrained* OpenAI GPT-2 on "The Verdict": train 3.7547, validation 3.5596. Raschka notes that if the story was part of GPT-2's training dataset, "we can't tell whether the model is overfitting the training data because the validation set would have been used for training as well" — you'd need "a new dataset generated after OpenAI finished training GPT-2". (He also remarks that validation loss slightly below training loss is "unusual in deep learning" and likely random noise on a small dataset.) For LLMs trained on web-scale data, this **test-set contamination** is the main threat to any benchmark number.

**Why only a hold-out set, not K-fold?** One training run of a real LLM costs far too much to repeat $K$ times (the book's Llama 2 estimate is 184,320 GPU hours, about \$690,000; §5.1, book p. 141). So LLM practice uses a single hold-out validation split. See [[cross-validation]] for the classical alternative.

## Connections
- Builds on: [[empirical-risk-minimization]], [[overfitting-and-underfitting]], [[random-variable]]
- Leads to: [[training-and-validation-loss]], [[cross-validation]], [[baseline-evaluation]], [[llm-evaluation]]
- Used by: [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]], [[pytorch-dataset-and-dataloader]], [[hyperparameter]] tuning, [[llm-as-a-judge]]
- Related: [[overfitting]], [[regularization]], [[dropout]], [[weight-decay]], [[epoch]], [[random-seed-reproducibility]], [[classification-accuracy]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3 (Fig. 5.9, Listing 5.2), book pp. 140–145 — 90/10 character split, 9 + 1 batches, initial losses.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.2 (Listing 6.3), book p. 175; §6.3, book pp. 178–181 — 70/10/20 split, 1,045/149/300 (derived), 130/19/38 batches (p. 181).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7, book p. 200 — final 97.21% / 97.32% / 95.67%, val-vs-test remark.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.2 (Listing 7.3), book p. 210; §7.4, book pp. 223–225; §7.7, book p. 233; §7.8, book pp. 238–247 — 85/10/5 split, 935/110/55, held-out test set scored by an LLM judge.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.9 summary, book p. 249.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.6, book pp. 270–273; §A.7, book p. 275 — validation vs test "precisely once".
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1, book p. 314; §E.2, book pp. 324–325; §E.4, book pp. 335–336 (LoRA final accuracies); Appendix C Exercise 5.5, book pp. 305–306.
