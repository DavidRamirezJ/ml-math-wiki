---
title: Epoch
type: concept
tags: [machine-learning, optimization, training-dynamics, pytorch, llm]
difficulty: beginner
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Epoch

> **One-liner:** One epoch means the model has gone through every training example once.

*Also known as:* training epoch, pass over the data.

## 🟢 Intuition (Dummy)
Picture a deck of flashcards. You shuffle it, go through every card once — that's **one epoch**. Then you shuffle again and go through the whole deck a second time — **epoch two**.

You don't flip one card at a time, though: you grab a small handful (a **batch**), study it, adjust what you think, grab the next handful. So one epoch = (number of cards) ÷ (cards per handful) little study sessions.

Going through the deck many times helps — up to a point. Go through a *tiny* deck fifty times and you'll memorize the cards instead of understanding the subject. That trade-off is the whole story of "how many epochs?"

## 🟡 Mechanics (Practitioner)

### Definition
The book defines it in two places:
- "One epoch is one complete pass over a training set" (§5.2, book p. 146; §6.7, book p. 195).
- An epoch visits "each training example exactly once" (§A.6, book p. 272).

### Batches per epoch
$$N_{\text{batches}} = \left\lfloor \frac{N_{\text{train}}}{B} \right\rfloor \ \ (\texttt{drop\_last=True}) \qquad\text{or}\qquad \left\lceil \frac{N_{\text{train}}}{B} \right\rceil \ \ (\texttt{drop\_last=False})$$

Symbol by symbol:
- $N_{\text{train}}$ = number of training examples (or text chunks).
- $B$ = batch size.
- $\lfloor\cdot\rfloor$ = round down (drop the leftover partial batch); $\lceil\cdot\rceil$ = round up (keep it as a small final batch).

The book phrases it in words: "training set size divided by the size of each batch" (book p. 146). Each batch is one [[gradient]] update (one `optimizer.step()`), so

$$\text{total steps} = N_{\text{batches}} \times \texttt{num\_epochs}.$$

### The epoch loop in PyTorch (§A.7, book pp. 274–275)
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3
for epoch in range(num_epochs):          # outer loop: epochs
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):  # inner loop: batches
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()                         # optional evaluation after each epoch
```

### Worked numbers from the book

| Where | $N_{\text{train}}$ | $B$ | batches/epoch | epochs | total steps |
|---|---|---|---|---|---|
| App. A toy (§A.6–A.7, pp. 272–275) | 5 | 2 | $\lfloor 5/2\rfloor = 2$ | 3 | 6 |
| Ch. 5 pretraining (§5.2, p. 149) | 9 chunks of 256 tokens | 2 | 9 | 10 | 90 |
| App. D schedule (§D.1, p. 315) | same as Ch. 5 | 2 | 9 | 15 | 135 |
| Ch. 6 spam (§6.7, p. 197) | 1,045 | 8 | $\lfloor 1045/8\rfloor = 130$ | 5 | 650 |
| Ch. 7 instructions (§7.6, pp. 231–232) | 935 | 8 | $\lfloor 935/8\rfloor = 116$ | 2 | 232 |

- Ch. 7 check against the log: with `drop_last=True` (book p. 225), epoch 1 covers steps 0–115 and the first logged epoch-2 step is 120; the last logged step is 230 (book pp. 231–232) — consistent with 116 steps per epoch *(derived)*. Likewise Ch. 5's last log line is step 85 of 0–89 and Ch. 6's is step 600 of 0–649.
- In the toy run, the training loss goes 0.75 → 0.65 (epoch 1) → 0.44 → 0.13 (epoch 2) → 0.03 → 0.00 (epoch 3): "a sign that the model converged on the training set" (book p. 275).
- With `drop_last=True` and 5 examples, **one example is skipped every epoch** — and because of shuffling, which one changes each time (book p. 273).
- In Appendix D, `total_steps = len(train_loader) * n_epochs` = 9 × 15 = 135, and 20% warmup gives `int(0.2 * 135)` = **27** steps (book p. 315; see [[learning-rate-warmup]]).

### Reshuffling every epoch
`shuffle=True` in the training `DataLoader` draws a new order each epoch. With `torch.manual_seed(123)` you get the same first order as the book, and iterating again gives a *different* order — "to prevent deep neural networks from getting caught in repetitive update cycles during training" (§A.6, book p. 272).

### Logging in fractional epochs
The book logs losses every `eval_freq` *steps*, not every epoch, and then maps them onto an epoch axis with `torch.linspace(0, num_epochs, len(train_losses))` (book p. 150). That's why loss plots show values at "epoch 1.5". It also tracks a finer clock: `tokens_seen` in pretraining (`input_batch.numel()`, book p. 147) and `examples_seen` in classification (book p. 195).

### How many epochs? (sidebar, §6.7, book p. 199)
- "There is no universal solution or recommendation, although an epoch number of five is usually a good starting point."
- Overfitting after a few epochs on the loss plot → **reduce** epochs.
- Validation loss still trending down → **increase** epochs.

The book's choices: Ch. 5 uses 10 epochs on a tiny story and visibly overfits after epoch 2 (book p. 150); real pretraining uses "a much larger dataset for only one epoch" (book p. 151). Ch. 7 stops at 2 epochs because a third "may even be counterproductive" (book p. 232). The number of epochs is a [[hyperparameter]] (book p. 275), like the [[learning-rate]].

Exercise 5.4 (book p. 160; solution Appendix C, p. 305) resumes from a checkpoint and trains **one more epoch** — which only works properly if the [[adamw-optimizer]] state is restored too.

## 🔴 Theory (Expert)

**Sampling without replacement.** An epoch with reshuffling is *random reshuffling* SGD: each pass samples a random permutation $\pi$ of $\{1,\dots,N\}$ and visits examples in that order. Classical [[stochastic-gradient-descent]] analysis assumes sampling *with* replacement (each step an i.i.d. draw, so each gradient is an unbiased estimate of the full gradient). Without replacement, steps within an epoch are dependent, but every example contributes exactly once per epoch; for smooth strongly convex objectives random reshuffling provably converges faster once the number of epochs $K$ is large enough — its error decays like $O(1/K^2)$ in $K$, versus $O(1/K)$ for with-replacement SGD run for the same number of steps (Gürbüzbalaban, Ozdaglar & Parrilo, 2019; constants depend on $N$ and the step size). *(Background, not in the book.)*

**One epoch ⇒ train loss is a fair estimate.** In a single pass, every batch is unseen when it's used, so the running training loss is an (approximately) unbiased estimate of the population risk at that moment — it behaves like a validation loss. Multi-epoch training breaks this: after the first pass the model has seen each example, so $\hat R_{\text{train}}$ becomes optimistic and the generalization gap appears (see [[training-and-validation-loss]]). This is the formal reason the Ch. 5 curves split after epoch 2.

**Data-constrained regimes.** When unique data runs out, repeating it is the only option. Empirical scaling studies (e.g. Muennighoff et al., 2023) found that up to about 4 epochs of repeated text is nearly as useful as fresh text for LLM pretraining, with rapidly diminishing returns after that. *(Background, not in the book.)* The book's tiny-corpus examples are far past that point, hence memorization; the remedy it points to is a larger corpus (60,000 Project Gutenberg books, book p. 151).

**Epochs as implicit regularization.** Limiting the number of epochs (early stopping) restricts how far the parameters can travel from their initialization; for linear least squares with gradient descent this is equivalent to a ridge penalty of strength ≈ $1/(\eta t)$. So "reduce the number of epochs" in the sidebar is a [[regularization]] lever, alongside [[dropout]] and [[weight-decay]].

## Connections
- Builds on: [[mini-batch]], [[stochastic-gradient-descent]], [[pytorch-dataset-and-dataloader]]
- Leads to: [[training-and-validation-loss]], [[learning-rate-schedule]], [[learning-rate-warmup]], [[cosine-decay]], [[overfitting]]
- Used by: [[training-loop]], [[llm-training-loop]], [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]]
- Related: [[hyperparameter]], [[learning-rate]], [[model-checkpointing]], [[random-seed-reproducibility]], [[gradient-descent]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2 (Listing 5.3, Fig. 5.11), book pp. 146–151.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7 (epochs sidebar), book pp. 195–199.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6, book pp. 231–232.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.6, book pp. 270–274; §A.7 (Listing A.9), book pp. 274–276.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1, book pp. 314–315; §5.4 Exercise 5.4, book p. 160.
