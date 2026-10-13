---
title: Training Loop (PyTorch)
type: concept
tags: [deep-learning, optimization, pytorch, training]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, llm-ch06-fine-tuning-for-classification, llm-ch05-pretraining-on-unlabeled-data, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Training Loop (PyTorch)

> **One-liner:** A training loop is the short routine you repeat thousands of times to teach a model: guess, measure the error, find out which way each weight should move, move it a little, repeat.

## 🟢 Intuition (Dummy)
Think of learning to throw darts. You throw (the model makes a guess), you see
how far you missed (the loss), you figure out how to adjust your arm (the
gradients), you adjust a bit (the update) — and you clear your head before the
next throw so the last correction doesn't get counted twice (resetting the
gradients). One round through your whole practice bucket of darts is an
**epoch**; each handful of darts you throw before adjusting is a **batch**.

Every PyTorch model in Raschka's book — a toy network with 2 inputs, a GPT
spam classifier, a full GPT pretraining run — is trained by the *same* five
or six lines in the middle of this loop. Once you can read those lines, you
can read any PyTorch training script.

## 🟡 Mechanics (Practitioner)

### The steps (Fig. 6.15, book p. 195)
1. For each training **[[epoch]]** — one complete pass over the training set.
2. For each **[[mini-batch|batch]]** — number of batches = training-set size ÷ batch size.
3. Reset the loss gradients (`optimizer.zero_grad()`). The figure's label says
   "from previous epoch," but the code resets them on every batch, i.e. it clears
   the gradients left by the previous batch.
4. Calculate the loss on the current batch ([[forward-pass]] + [[loss-function]]).
5. **Backward pass** to calculate the loss gradients (`loss.backward()`).
6. Update the model weights using the gradients (`optimizer.step()`).
7. *(optional)* Print training and validation losses.
8. *(optional, new in Ch. 6)* Calculate classification accuracy.

Steps 3–6 are "the usual steps used for training deep neural networks in
PyTorch."

### The minimal version (Listing A.9, §A.7, book pp. 274–275)
```python
import torch.nn.functional as F

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)                 # forward pass
        loss = F.cross_entropy(logits, labels)   # how wrong?
        optimizer.zero_grad()   # reset old gradients to 0
        loss.backward()         # compute gradients (autograd)
        optimizer.step()        # update the parameters
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")
    model.eval()
    # Insert optional model evaluation code
```
Line by line:
- `NeuralNetwork(2, 2)` — 2 inputs (two features), 2 outputs (two classes);
  layers 2→30→20→2 with ReLU, i.e. **752** trainable parameters
  ($90+620+42$; Exercise A.3).
- `torch.optim.SGD(model.parameters(), lr=0.5)` — the optimizer must be told
  *which* parameters to update; `lr` is the [[learning-rate]].
- `model.train()` / `model.eval()` — switch modes for layers that behave
  differently in training vs inference ([[dropout]], [[batch-normalization]]).
  This toy model has neither, so the calls are redundant here, but the book
  recommends keeping them ([[train-vs-eval-mode]]).
- `F.cross_entropy(logits, labels)` — takes raw [[logits]]; it applies
  [[softmax]] internally "for efficiency and numerical stability," so don't
  apply softmax yourself first ([[cross-entropy-loss]]).
- `optimizer.zero_grad()` — PyTorch **adds** new gradients to old ones; without
  this reset they accumulate (NOTE box, p. 276).
- `loss.backward()` — [[automatic-differentiation|autograd]] walks the
  [[computational-graph]] backward and fills every parameter's `.grad`.
- `optimizer.step()` — for SGD, "multiplying the gradients with the learning
  rate and adding the scaled negative gradient to the parameters."

**The update in symbols.**
$$\theta\leftarrow\theta-\eta\,\nabla_\theta\mathcal L(\theta),\qquad
\mathcal L=-\frac1B\sum_{i=1}^{B}\log\frac{\exp(z_{i,y_i})}{\sum_{k=1}^{K}\exp(z_{i,k})}$$
Symbol by symbol:
- $\theta$ — all trainable weights and biases; $\leftarrow$ — "is replaced by".
- $\eta$ (eta) — the learning rate (0.5 here).
- $\nabla_\theta\mathcal L$ — the [[gradient]] of the batch loss w.r.t. every parameter (what `.backward()` computes).
- $B$ — batch size (2); $K$ — number of classes (2).
- $z_{i,k}$ — the logit of example $i$ for class $k$; $y_i$ — the true class of example $i$.
- The fraction is the softmax probability of the true class; $-\log$ of it is the per-example cross-entropy; $\frac1B\sum$ averages over the batch.

(The book names these steps in words; the two formulas are the standard
definitions, written out here for reading practice.)

**The book's output** — 5 examples, `batch_size=2`, `drop_last=True` → 2
batches per epoch, 3 epochs → 6 updates (p. 275):

| epoch | batch 0 | batch 1 |
|---|---|---|
| 1 | 0.75 | 0.65 |
| 2 | 0.44 | 0.13 |
| 3 | 0.03 | 0.00 |

"The loss reaches 0 after three epochs, a sign that the model converged on
the training set." (The book's printout has a typo, "Trainl Loss," on one line.)

**After training** (pp. 276–278): `model.eval()` + `with torch.no_grad():`
for prediction; `torch.softmax(outputs, dim=1)` gives class probabilities
(first example: 99.91% class 0); `torch.argmax(..., dim=1)` gives labels
`[0, 0, 0, 1, 1]`; a `compute_accuracy` helper loops over a data loader
batch by batch and returns **1.0** on both train and test sets
([[classification-accuracy]], [[argmax]]).

**Hyperparameters and splits** (p. 275): learning rate and number of epochs
are [[hyperparameter|hyperparameters]] chosen by watching the loss. Tune them on
a **validation set** (used many times) and touch the **test set** only once,
so its estimate stays unbiased ([[train-validation-test-split]]).

### The same loop, scaled up: fine-tuning GPT-2 as a spam classifier (Listing 6.10, §6.7, book pp. 195–200)
```python
def train_classifier_simple(model, train_loader, val_loader, optimizer,
                            device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]   # examples, not tokens
            global_step += 1
            if global_step % eval_freq == 0:        # optional evaluation
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                ...
        train_accuracy = calc_accuracy_loader(train_loader, model, device,
                                              num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device,
                                            num_batches=eval_iter)
        ...
    return train_losses, val_losses, train_accs, val_accs, examples_seen
```
- The core four lines are **identical** to the toy loop. What changes is the
  bookkeeping: count `examples_seen` (not tokens, as in pretraining), and
  compute accuracy after each epoch instead of printing sample text.
- `calc_loss_batch` computes cross-entropy on the **last token's** logits only
  ([[last-token-pooling]]).
- `evaluate_model` wraps evaluation in `model.eval()` + `torch.no_grad()`,
  then switches back with `model.train()`.
- Settings (p. 197): [[adamw-optimizer|AdamW]], `lr=5e-5`,
  `weight_decay=0.1`, `num_epochs=5`, `eval_freq=50`, `eval_iter=5`, seed
  123. Runtime ≈ 6 min on an M3 MacBook Air (printed: 5.65 minutes), under
  half a minute on a V100/A100.

| after epoch | train acc. | val acc. |
|---|---|---|
| 1 | 70.00% | 72.50% |
| 2 | 82.50% | 85.00% |
| 3 | 90.00% | 90.00% |
| 4 | 100.00% | 97.50% |
| 5 | 100.00% | 97.50% |

Loss goes from 2.153 / 2.392 (train / val, step 0) to 0.083 / 0.074 (step 600).
Final full-dataset accuracy: **train 97.21%, val 97.32%, test 95.67%** (p. 200).
No real gap between training and validation curves → little
[[overfitting]] ([[training-and-validation-loss]]).

**Choosing the number of epochs** (sidebar, p. 199): no universal rule; 5
is a good start. Overfitting early → fewer epochs; validation loss still
falling → more.

### Variants in the book
- **Pretraining** (`train_model_simple`, Ch. 5): same loop, tracks tokens
  seen and prints generated text — see [[llm-training-loop]].
- **Bells and whistles** (Appendix D): [[learning-rate-warmup]],
  [[cosine-decay]], and [[gradient-clipping]] slotted between
  `loss.backward()` and `optimizer.step()` (applied after warmup, `max_norm=1.0`).
- **Multi-GPU** (§A.9): wrap the model in [[distributed-data-parallel|DDP]];
  the loop body is unchanged, gradients are averaged across GPUs.

## 🔴 Theory (Expert)
- **It's stochastic ERM.** The goal is [[empirical-risk-minimization]]:
  $\min_\theta\hat R(\theta)=\frac1N\sum_{n=1}^{N}\ell(f_\theta(\mathbf x_n),y_n)$.
  A uniformly shuffled mini-batch gives an **unbiased** estimate of
  $\nabla\hat R$ (by linearity of the gradient), so the loop is
  [[stochastic-gradient-descent]]: noisy steps that are right on average.
  `drop_last=True` with shuffling keeps batch sizes equal while still
  covering a random subset each epoch.
- **Why `zero_grad` is a design choice, not a quirk.** Because `.grad`
  accumulates, $k$ calls of `backward()` before one `step()` sum $k$
  mini-batch gradients — exactly [[gradient-accumulation]], a way to simulate
  a $k\times$ larger batch when memory is tight. Forgetting `zero_grad` does the
  same thing *by accident* — and keeps growing the step every iteration.
- **Order matters.** `zero_grad → forward → loss → backward → (clip) → step`.
  Clipping must sit between `backward` (so gradients exist) and `step` (before
  they are used). `zero_grad` may go anywhere outside that window (Listing A.9
  calls it after the loss; Listing 6.10 calls it first) — both are correct.
- **Loss → 0 on 5 examples means memorization.** A 752-parameter model can fit
  4–5 points perfectly; "converged on the training set" says nothing about
  generalization. That's what the validation/test split is for, and why
  validation accuracy can end up a bit above test accuracy after tuning on it
  (p. 200) — model selection leaks a little information
  ([[cross-validation]], [[overfitting-and-underfitting]]).
- **Surrogate objective.** The loop minimizes cross-entropy, not accuracy:
  accuracy's gradient is zero almost everywhere, so it gives the optimizer
  nothing to follow ([[surrogate-loss]]; §6.6, p. 193). Cross-entropy is the
  negative log-likelihood of a categorical model, so the loop is doing
  [[maximum-likelihood-estimation]].
- **Initial loss sanity check.** For $K$ classes, a random-but-calibrated
  model should start near $\ln K$ ($\ln2\approx0.693$). The Ch. 6 run starts
  at 2.153 because the freshly initialized classification head outputs
  uncalibrated logits (our inference).
- **Regularization knobs** that live in or around the loop: `weight_decay`
  in AdamW, `drop_rate` (dropout, active only in `train()` mode), and the
  number of epochs itself (early stopping) ([[regularization]],
  [[weight-decay]]).

## Connections
- Builds on: [[gradient]], [[automatic-differentiation]], [[backpropagation]], [[loss-function]], [[gradient-descent]], [[pytorch-dataset-and-dataloader]], [[pytorch-nn-module]]
- Leads to: [[llm-training-loop]], [[gradient-accumulation]], [[distributed-data-parallel]], [[model-checkpointing]]
- Used by: [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]]
- Related: [[stochastic-gradient-descent]], [[adamw-optimizer]], [[cross-entropy-loss]], [[epoch]], [[mini-batch]], [[train-vs-eval-mode]], [[learning-rate-schedule]], [[gradient-clipping]], [[empirical-risk-minimization]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7 "A typical training loop" (Listings A.9–A.10, Exercise A.3), book pp. 274–278 (PDF pp. 296–300).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7 "Fine-tuning the model on supervised data" (Fig. 6.15, Listings 6.10–6.11, Figs. 6.16–6.17), book pp. 195–200 (PDF pp. 217–222).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6 (accuracy not differentiable), book p. 193; §A.9 (training loop on GPU, DDP), book pp. 281–286; §D.3–D.4 (clipping placement), book pp. 317–320.
