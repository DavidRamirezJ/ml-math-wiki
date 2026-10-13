---
title: LLM Training Loop
type: concept
tags: [llm, deep-learning, optimization, pytorch, training]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# LLM Training Loop

> **One-liner:** The training loop is the repeat-forever routine that teaches an LLM: show it a batch of text, measure how wrong its next-word guesses were, work out which way to nudge each weight, nudge, repeat.

## 🟢 Intuition (Dummy)
Think of practising free throws. You shoot (a guess), see how far off you were
(the loss), figure out what to change in your arm (the gradients), adjust a
little (the update), and shoot again. You do this many times per session (one
**epoch** = one pass through all your practice material), and now and then a
coach checks you on shots you have *not* practised (the validation set) to see
whether you are really improving or just getting used to one spot.

An LLM's training loop is exactly that, with text. It reads a chunk, predicts each
next token, gets scored with [[cross-entropy-loss]], and nudges its millions of
weights to score better next time. The same loop, with tiny changes, is used for
[[pretraining]], [[classification-fine-tuning]] and [[instruction-fine-tuning]].

## 🟡 Mechanics (Practitioner)

### The eight steps (Raschka Fig. 5.11, book p. 146)
1. Loop over training **[[epoch|epochs]]** (one epoch = one full pass over the training set).
2. Loop over **[[mini-batch|batches]]** in each epoch. Number of batches ≈ training-set size ÷ batch size.
3. **Reset** the gradients from the previous batch.
4. **Compute the loss** on the current batch.
5. **Backward pass**: compute the loss gradients ([[backpropagation]]).
6. **Update** the weights with the gradients.
7. *(optional)* Print training and validation losses.
8. *(optional)* Generate a sample text to eyeball progress.

Steps 3–6 are the standard PyTorch recipe; 7–8 are just monitoring (book p. 146).

### The update in symbols
$$\theta\leftarrow\theta-\eta\,\nabla_\theta\,\mathcal L(\theta;\text{batch})$$
- $\theta$ = all trainable weights (`model.parameters()`), see [[model-parameters]].
- $\leftarrow$ = "is replaced by".
- $\eta$ (eta) = the [[learning-rate]], the step size (`lr`).
- $\nabla_\theta\mathcal L$ = the [[gradient]]: for every weight, how much the loss changes if that weight grows a little. `loss.backward()` computes it.
- $\mathcal L(\theta;\text{batch})$ = the cross-entropy loss on this batch.
- The minus sign = step *downhill*, against the gradient ([[gradient-descent]]).

This is the plain [[stochastic-gradient-descent]] form (used in appendix A with
`lr=0.5`, book p. 274). LLMs use **[[adamw-optimizer|AdamW]]**, which rescales each
weight's step with running averages of its gradients and applies
[[weight-decay]] separately:
$$\theta_t=\theta_{t-1}-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\,\theta_{t-1}\Big)$$
- $t$ = step number; $\hat m_t$ = running average of the gradient (its "direction"); $\hat v_t$ = running average of the squared gradient (its "size").
- $\epsilon$ = a tiny number to avoid dividing by zero.
- $\lambda$ (lambda) = weight decay (0.1 in the book), which gently shrinks every weight toward 0.

The book only describes AdamW in words: Adam with an improved weight-decay
approach, which penalizes large weights and helps generalization (book p. 148).
The formula is the standard one.

### The book's code: `train_model_simple` (Listing 5.3, book p. 147)
```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):                      # step 1
        model.train()                                    # dropout on
        for input_batch, target_batch in train_loader:   # step 2
            optimizer.zero_grad()                        # step 3
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # step 4
            loss.backward()                              # step 5
            optimizer.step()                             # step 6
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:             # step 7
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss); val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
        generate_and_print_sample(model, tokenizer, device, start_context)  # step 8
    return train_losses, val_losses, track_tokens_seen
```
- `optimizer.zero_grad()` matters because PyTorch **adds** new gradients onto old ones unless you clear them.
- `evaluate_model` switches to `model.eval()` (dropout off) and `torch.no_grad()`
  (no gradient bookkeeping), averages the loss over `eval_iter` batches, then
  switches back to `model.train()` (book pp. 147–148). See [[train-vs-eval-mode]].
- `tokens_seen` counts tokens processed: `numel()` of a $[2, 256]$ batch is 512.

### The book's pretraining run (book p. 149)
```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)        # context_length shortened to 256
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
train_model_simple(model, train_loader, val_loader, optimizer, device,
    num_epochs=10, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer)
```
Data: "The Verdict" (20,479 characters, 5,145 tokens), split 90/10, giving 9
training batches and 1 validation batch of shape $[2, 256]$ (book pp. 141–144).
About 5 minutes on a MacBook Air.

| Epoch (step) | Train loss | Val loss | Sample |
|---|---|---|---|
| 1 (0) | 9.781 | 9.933 | `Every effort moves you,,,,,,,,,,,,.` |
| 2 (10) | 6.661 | 7.048 | `…, and, and, and, …` |
| 9 (80) | 0.541 | 6.393 | fluent but copied text |
| 10 (85) | 0.391 | 6.452 | `…know," was one of the axioms he laid down…` |

Training loss falls fast; validation loss stalls near 6.4 after about epoch 2.
That gap is [[overfitting]]: the model memorizes the story word for word
(book pp. 150–151). Normally you train on a much bigger dataset for about **one
epoch** (book p. 151). See [[training-and-validation-loss]].

### Same loop, other jobs
| Use | What changes | Settings | Book |
|---|---|---|---|
| Classification ([[classification-fine-tuning]]) | loss on the last token's 2 logits; track accuracy; count examples instead of tokens (`train_classifier_simple`) | AdamW `lr=5e-5`, `weight_decay=0.1`, 5 epochs, `eval_freq=50` → val acc. 97.50% (on 5 eval batches); full-set accuracy 97.21% / 97.32% / 95.67% (train / val / test); 5.65 min on an M3 MacBook Air | §6.7, pp. 195–200 |
| Instruction tuning ([[instruction-fine-tuning]]) | nothing; reuse `train_model_simple` as is | GPT-2 medium, AdamW `lr=0.00005`, `weight_decay=0.1`, 2 epochs; loss 2.637 → 0.300 (val 0.657); 0.87 min on an A100 | §7.6, pp. 230–232 |
| Upgraded loop (`train_model`) | [[learning-rate-warmup]] + [[cosine-decay]] + [[gradient-clipping]] | peak lr 0.001 (AdamW default, `weight_decay=0.1`), `initial_lr=1e-5`, `min_lr=1e-5`, 27 warmup steps (20% of total, computed in §D.1, p. 315), 15 epochs, `eval_iter=1`, `max_norm=1.0` | App. D §D.4, pp. 319–321 |

### The upgraded loop (appendix D, book pp. 319–320)
Per batch: zero grads → step counter +1 → compute this step's learning rate →
write it into the optimizer → loss → `backward()` → clip gradients (only after
warmup) → `optimizer.step()`. The [[learning-rate-schedule]]:
$$\eta_t=\begin{cases}\eta_0+t\,\dfrac{\eta_{\max}-\eta_0}{W} & t<W\ \text{(warmup)}\\[2mm]
\eta_{\min}+(\eta_{\max}-\eta_{\min})\cdot\tfrac12\big(1+\cos(\pi\,p_t)\big),\ \ p_t=\dfrac{t-W}{T-W} & t\ge W\ \text{(cosine)}\end{cases}$$
- $\eta_t$ = learning rate at step $t$; $\eta_0$ = start value; $\eta_{\max}$ = peak; $\eta_{\min}$ = floor.
- $W$ = number of warmup steps; $T$ = total steps (batches per epoch × epochs).
- $p_t$ = "progress" through the post-warmup part, from 0 to almost 1.
- $\tfrac12(1+\cos\pi p_t)$ slides smoothly from 1 down to 0, so the rate glides from peak to floor.

Clipping rescales the full gradient vector $g$ when it is too long:
$g\leftarrow g\cdot\min\!\big(1,\ c/\lVert g\rVert_2\big)$ with $c=1.0$ (see [[norm]]).
Result on "The Verdict": losses 10.934 → **0.035** train, **6.938** validation,
again overfitting on the tiny dataset (book p. 321).

## 🔴 Theory (Expert)
- **What the loop optimizes.** It runs stochastic gradient descent on the
  [[empirical-risk-minimization|empirical risk]]
  $\hat R(\theta)=\frac1N\sum_n\ell_n(\theta)$, the average token-level
  cross-entropy. Each mini-batch gradient is an unbiased estimate of
  $\nabla\hat R$ (when batches are sampled uniformly), with variance shrinking
  like $1/B$ for batch size $B$. The book's batch size 2 is tiny; it notes real
  LLMs use 1,024 or more (book p. 143).
- **Pretraining is maximum likelihood.** Minimizing average cross-entropy is
  [[maximum-likelihood-estimation]] of the autoregressive model
  $\prod_t p_\theta(x_{t+1}\mid x_{\le t})$. The chapter 5 summary: pretraining =
  changing the weights to minimize the training loss, with conventional
  cross-entropy and AdamW (book p. 168).
- **Why validation loss is noisy here.** `eval_iter=5` estimates the loss from
  only 5 batches (and the validation loader has just 1 batch of 512 tokens).
  Losses printed during training are estimates, not exact dataset averages.
- **Decoupled weight decay.** In Adam with L2 regularization, the penalty
  gradient $\lambda\theta$ is divided by $\sqrt{\hat v_t}$, so weights with large
  gradient history are barely regularized. AdamW applies $\lambda\theta$ outside the
  adaptive scaling, so decay is uniform. This is the "improved weight decay"
  the book refers to (book p. 148). See [[regularization]].
- **Why warmup and clipping.** At initialization Adam's second-moment estimates
  are poor, and big early updates can push a transformer into unstable regions;
  linear warmup keeps early steps small. Clipping by global norm keeps the step
  direction but caps its length, guarding against
  [[vanishing-and-exploding-gradients|exploding gradients]]. The book applies
  clipping only after warmup (book p. 320). Cosine decay lets the model settle
  into a minimum with small final steps.
- **Resuming training.** To continue later you must save the optimizer state
  too (`optimizer.state_dict()`), because AdamW keeps per-weight moment estimates
  (Exercise 5.4, appendix C p. 305). See [[model-checkpointing]].
- **Scaling out.** The same loop runs on many GPUs with
  [[distributed-data-parallel]] (appendix A.9): each GPU computes the gradient on its
  own shard, gradients are averaged, and every replica applies the same update.
  [[gradient-accumulation]] fakes a big batch on small hardware.

## Connections
- Builds on: [[cross-entropy-loss]], [[gradient-descent]], [[stochastic-gradient-descent]], [[backpropagation]], [[gradient]], [[mini-batch]], [[epoch]], [[pytorch-dataset-and-dataloader]]
- Leads to: [[training-and-validation-loss]], [[adamw-optimizer]], [[learning-rate-schedule]], [[learning-rate-warmup]], [[cosine-decay]], [[gradient-clipping]], [[model-checkpointing]]
- Used by: [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]], [[training-loop]]
- Related: [[overfitting]], [[weight-decay]], [[train-vs-eval-mode]], [[dropout]], [[distributed-data-parallel]], [[llm-building-roadmap]], [[empirical-risk-minimization]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7, book pp. 14–15 (stage 2: pretraining, training loop, model evaluation)
- Same, §5.1.3, book pp. 140–145 (dataset, 90/10 split, data loaders, `calc_loss_batch`, `calc_loss_loader`)
- Same, §5.2, book pp. 146–151 (Fig. 5.11 eight steps, Listing 5.3 `train_model_simple`, `evaluate_model`, AdamW sidebar p. 148, run and losses p. 149, Fig. 5.12 overfitting p. 150); §5.5 summary, book p. 168
- Same, §6.7, book pp. 195–200 (`train_classifier_simple`, settings and results)
- Same, §7.6, book pp. 229–233 (Listing 7.8, instruction-tuning run)
- Same, appendix A §A.7, book pp. 274–276 (basic PyTorch loop with SGD, `lr=0.5`); §A.9, book pp. 281–283 (multi-GPU)
- Same, appendix D §D.4, book pp. 319–321 (`train_model` with warmup, cosine decay, clipping)
- Same, appendix C, Exercise 5.4, book p. 305 (resume with optimizer state)
