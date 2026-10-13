---
title: Training vs Evaluation Mode (and torch.no_grad)
type: concept
tags: [pytorch, deep-learning, training, inference, llm]
difficulty: beginner
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Training vs Evaluation Mode (and torch.no_grad)

> **One-liner:** A model behaves a bit differently while it is learning than while it is being used, and two PyTorch switches, `model.train()`/`model.eval()` and `torch.no_grad()`, tell it which situation it is in.

## 🟢 Intuition (Dummy)
Picture a basketball player. In **practice** the coach makes them play with one hand tied, at
random, so they don't lean on any single habit. In the **real game** nobody ties their hand; they
play with everything they've got. That random hand-tying is [[dropout]], and it only belongs in
practice.

- `model.train()` = "this is practice": random training-only tricks (dropout, batch-norm updates) are on.
- `model.eval()` = "this is the real game": those tricks are off, so the same input always gives
  the same output.

There is a second, separate switch. During practice a video analyst records every move so the
coach can work out, afterwards, what went wrong (that recording is the
[[computational-graph]] used by [[backpropagation]]). In the real game you don't need the replay
tape. `torch.no_grad()` says "stop recording". This saves memory and time.

The two switches do **different jobs**, and when you just want predictions you usually use both.

## 🟡 Mechanics (Practitioner)

### Switch 1: `model.train()` / `model.eval()`
These set the model to training mode and evaluation mode. You need them for components that behave
differently in training and inference, for example **dropout** or **batch normalization** layers
(book pp. 275–276).

**Dropout in each mode.** With dropout rate $p$, in training mode each value $x_i$ becomes

$$\tilde x_i = \frac{m_i}{1-p}\,x_i, \qquad m_i \sim \text{Bernoulli}(1-p)$$

Reading it symbol by symbol:
- $x_i$ = one number coming out of a layer (e.g. one attention weight);
- $m_i$ = a random "keep" switch: it is 1 with probability $1-p$ and 0 with probability $p$
  ("$\sim$ Bernoulli$(1-p)$" means "is drawn like a biased coin flip that lands 1 with probability $1-p$");
- $\frac{1}{1-p}$ = the scale-up for survivors;
- $\tilde x_i$ = the value after dropout.

In evaluation mode dropout is simply the identity: $\tilde x_i = x_i$.

**The book's worked numbers (§3.5.2).** `torch.nn.Dropout(0.5)` on a $6\times6$ matrix of ones
(seed 123) zeroes about half the entries and turns the rest into **2.** — the survivors are scaled
up by $1/0.5 = 2$ to keep the overall balance (book pp. 78–79). The first printed row is
`[2., 2., 0., 2., 2., 0.]`. In eval mode the same call would return the ones unchanged. GPT
training later uses a lower rate, such as 0.1 or 0.2 (book p. 78).

**Where the book flips the switch.**
| Place | Code | Why (book) | Page |
|---|---|---|---|
| Ch. 4 first generation | `model.eval()` before `generate_text_simple` | disables random components like dropout, used only in training | p. 126 |
| Ch. 5 `evaluate_model` | `model.eval()` … `model.train()` | "Dropout is disabled during evaluation for stable, reproducible results" | p. 148 |
| Ch. 5 `generate_and_print_sample` | `model.eval()` … `model.train()` | print a sample mid-training, then go back to training | p. 148 |
| Ch. 5 after loading weights | `model.eval()` | inference; we don't want to drop learned information | p. 160 |
| Ch. 5 resume training | `model.train()` | dropout active again for more training | p. 160 |
| Ch. 6 `classify_review` | `model.eval()` | classify new texts | p. 201 |
| App. A training loop | `model.train()` per epoch, `model.eval()` after | best practice | pp. 274–276 |

**Redundant but good practice.** The appendix A `NeuralNetwork` has no dropout or batch-norm
layers, so the calls are redundant there. The book still includes them, to avoid surprises when you
change the architecture or reuse the code for another model (book p. 276).

### Switch 2: `torch.no_grad()`
During training PyTorch builds a computational graph so it can run `loss.backward()`. If you only
want predictions, building that graph "performs unnecessary computations and consumes additional
memory". So for **inference** the best practice is the `torch.no_grad()` context manager, which
tells PyTorch not to track gradients (book p. 269).

**The book's worked numbers (§A.5).** For the untrained `NeuralNetwork(50, 3)` with seed 123 and
`X = torch.rand((1, 50))`:

```python
out = model(X)
# tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)

with torch.no_grad():
    out = model(X)
# tensor([[-0.1262,  0.1080, -0.1792]])
```
(book pp. 268–269)

Same numbers, but the second result has no `grad_fn`: no graph was recorded. (`Addmm` = matrix
multiply then add, the last [[linear-layer]]'s $XW^\top + \mathbf b$.) The book says skipping the
graph "can result in significant savings in memory and computation" (book p. 269).

The model returns raw [[logits]]; to get class probabilities you call [[softmax]] explicitly, also
under `no_grad`: `torch.softmax(model(X), dim=1)` gives `tensor([[0.3113, 0.3934, 0.2952]])`,
roughly equal values that sum to 1, as expected for an untrained model (book p. 269).

### Using both together
The standard inference pattern from the book:

```python
model.eval()
with torch.no_grad():
    outputs = model(X_train)
```
(book p. 276)

and in Chapter 6:

```python
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    ...
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]   # last-token logits, shape (1, 2)
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"
```
(book p. 201)

The text-generation functions also wrap the forward pass in `torch.no_grad()` (Ch. 4 `generate_text_simple`, book p. 124; Ch. 5 `generate`, book pp. 157–158).

| | `model.eval()` | `torch.no_grad()` |
|---|---|---|
| Changes the **output values**? | Yes, if dropout/batch-norm exist | No |
| Saves memory / compute? | No | Yes (no graph stored) |
| Needed for `loss.backward()` later? | — | Must be **off** |

## 🔴 Theory (Expert)
- **Why scale by $1/(1-p)$: matching expectations.** For one unit,
  $\mathbb E[\tilde x_i] = \Pr(m_i{=}1)\cdot\frac{x_i}{1-p} = (1-p)\frac{x_i}{1-p} = x_i$.
  So the training-mode output equals the eval-mode output *on average*
  ([[expected-value-and-mean]]). This "inverted dropout" lets eval mode be the plain identity.
  The original Srivastava et al. formulation instead scaled the weights by $1-p$ at test time; the
  two are equivalent in expectation.
- **Eval mode as an approximation to an ensemble.** Each dropout mask defines a thinned
  sub-network; training samples sub-networks at random. Eval mode replaces the average prediction
  over all $2^{n}$ masks with a single deterministic pass. For a single linear layer this is exact
  in expectation; through nonlinearities it is an approximation ("weight-scaling inference rule").
  Keeping dropout *on* at test time and averaging many passes ("MC dropout") is a different
  technique that estimates the ensemble directly ([[monte-carlo-estimation]]).
- **Batch normalization is the other mode-dependent layer.** In train mode it normalizes with the
  current mini-batch's mean and variance and updates running averages; in eval mode it uses the
  stored running averages, so a single example is processed the same way regardless of its batch
  ([[batch-normalization]]). [[layer-normalization]] (used by GPT) normalizes each token on its own
  and behaves the same in both modes.
- **What `no_grad` changes in autodiff terms.** Reverse-mode [[automatic-differentiation]] must
  keep every intermediate activation of the forward pass to evaluate the local Jacobians in the
  backward pass. That memory scales with depth × batch × sequence length. Under `no_grad` the
  intermediate tensors can be freed as soon as the next layer has used them, so inference memory is
  dominated by the weights. The math of the forward function is unchanged, which is why the A.5
  outputs match to every digit.
- **Determinism.** Eval mode removes dropout randomness, but sampling-based decoding
  ([[temperature-scaling]], [[top-k-sampling]], [[multinomial-sampling]]) still adds randomness at
  the *output*. Only greedy decoding with eval mode is fully deterministic ([[greedy-decoding]]).

## Connections
- Builds on: [[dropout]], [[computational-graph]], [[automatic-differentiation]], [[backpropagation]], [[pytorch-nn-module]]
- Leads to: [[llm-evaluation]], [[training-and-validation-loss]], [[autoregressive-text-generation]], [[model-checkpointing]]
- Related: [[batch-normalization]], [[layer-normalization]], [[regularization]], [[overfitting]], [[llm-training-loop]], [[forward-pass]]
- People: [[nitish-srivastava]], [[geoffrey-hinton]]

> **From [[llm-appD-training-loop-bells-and-whistles]]:** Appendix D puts the freshly built model in `model.eval()` mode during setup (book p. 313), and the new `train_model` calls `model.train()` at the start of every epoch (book p. 319), the same pattern as Ch. 5.

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5.2, book pp. 78–79 (dropout rate 0.5, survivors scaled by 1/0.5 = 2, printed 6×6 example).
- Same PDF, §4.7, book pp. 124, 126 (`no_grad` in `generate_text_simple`; `model.eval()` before first generation).
- Same PDF, §5.2, book p. 148 (`evaluate_model`, `generate_and_print_sample`); §5.3, book pp. 151, 157–158; §5.4, book p. 160.
- Same PDF, §6.8, book p. 201 (Listing 6.12 `classify_review`).
- Same PDF, §A.5, book pp. 268–269 (`grad_fn=<AddmmBackward0>`, `torch.no_grad()` best practice).
- Same PDF, §A.7, book pp. 274–276 (`model.train()`/`model.eval()`, dropout and batch norm, redundant but best practice).
- Expectation argument, ensemble view, MC dropout, batch-norm running stats: standard background (Srivastava et al. 2014, cited in the book's appendix B, p. 292), not derived in the book.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D intro, book p. 313; §D.4, book p. 319 (`model.eval()` / `model.train()`).

## From llm-ch07-fine-tuning-to-follow-instructions
- §7.5 loads GPT-2 medium (355M) and calls `model.eval();` right after `load_weights_into_gpt(model, params)` (book p. 227). This happens before the [[baseline-evaluation]] on `val_data[0]`. The config sets `drop_rate` to 0.0 (book p. 227), so dropout does nothing here anyway. Calling `eval()` is still the habit the book follows. The training loop from Ch. 5, which §7.6 reuses, switches between `model.train()` and `model.eval()` for each evaluation step. See [[llm-ch07-fine-tuning-to-follow-instructions]].
- Source: `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5, book pp. 226–228; §7.6, book pp. 230–231.
