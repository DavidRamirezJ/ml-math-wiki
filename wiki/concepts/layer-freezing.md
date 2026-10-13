---
title: Layer Freezing and Partial Fine-Tuning
type: concept
tags: [deep-learning, fine-tuning, transfer-learning, optimization, pytorch, llm]
difficulty: intermediate
sources: [llm-ch06-fine-tuning-for-classification, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Layer Freezing and Partial Fine-Tuning

> **One-liner:** Freezing means you lock most of a pretrained network so it can't change, and you only let a few layers learn the new task.

## 🟢 Intuition (Dummy)
Picture an experienced translator who's moving into legal documents. You don't send
them back to learn the alphabet and grammar. Those skills are already solid. You just
teach them the legal words and phrasing on top.

A pretrained [[large-language-model]] is in the same position. Its **lower layers**
(the ones near the input) have already learned general stuff about language, like
grammar, word meanings and basic structure. Pretty much every task needs that. The
**last layers** (near the output) hold the more specific, task-flavoured patterns. So
when you [[fine-tuning|fine-tune]] for something new, like spotting spam, you can
"freeze" the lower layers (no more changes allowed) and only train the top few.

You get two wins from this:
- **It's cheaper.** Frozen layers don't need [[gradient|gradients]] or optimizer memory.
- **It's safer.** The general knowledge the model picked up in [[pretraining]] can't get
  wiped out by a small task-specific dataset.

The book puts it like this: lower layers "capture basic language structures and
semantics applicable across a wide range of tasks", while the last layers are "more
specific to nuanced linguistic patterns and task-specific features" (book p. 185).

## 🟡 Mechanics (Practitioner)

### The update rule with frozen parameters
Split all the model's parameters $\theta$ into two groups, $\theta = (\theta_{\text{frozen}}, \theta_{\text{train}})$.
[[gradient-descent|Gradient descent]] then only touches the trainable group:

$$\theta_{\text{train}} \leftarrow \theta_{\text{train}} - \eta\,\nabla_{\theta_{\text{train}}}\mathcal{L}(\theta_{\text{frozen}}, \theta_{\text{train}}), \qquad \theta_{\text{frozen}} \ \text{stays the same}$$

Reading it symbol by symbol:
- $\theta$ (theta) = every number the model can learn (all weights and biases), see [[model-parameters]].
- $\theta_{\text{frozen}}$ = the parameters we've locked. In the update they act like constants.
- $\theta_{\text{train}}$ = the parameters we still let learn.
- $\leftarrow$ = "gets replaced by" (it's an assignment, not an equation).
- $\eta$ (eta) = the [[learning-rate]], meaning how big each step is.
- $\nabla_{\theta_{\text{train}}}\mathcal{L}$ = the [[gradient]] of the loss $\mathcal{L}$, but only with respect to
  the trainable parameters. "Which way should *these* knobs turn to lower the loss?"
- $\mathcal{L}(\theta_{\text{frozen}}, \theta_{\text{train}})$ = the loss still depends on *both* groups. The frozen
  weights still do their job in the forward pass. They just never get updated.

### How you do it in PyTorch
Each tensor in a [[pytorch-nn-module|`nn.Module`]] has a `requires_grad` flag. Set it to
`False` and [[automatic-differentiation|autograd]] stops computing a gradient for that tensor, so the
optimizer has nothing to update it with (book p. 186).

**Chapter 6 recipe (spam classifier, GPT-2 small 124M), book pp. 186–187:**

```python
# 1) Freeze everything
for param in model.parameters():
    param.requires_grad = False

# 2) Swap in a new classification head (new layers come with requires_grad=True)
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"],  # 768
                                 out_features=num_classes)

# 3) Unfreeze the last transformer block and the final LayerNorm
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```

The order matters here. You freeze first, *then* add the new [[classification-head]].
A freshly created layer has `requires_grad=True` by default, so right after step 2 it's
the only trainable layer (book p. 186). Step 3 then opens up a bit more of the model.

**What ends up trainable** (book p. 187, Fig. 6.10):

| Part of GPT-2 small | Trainable? |
|---|---|
| Token + positional [[embedding-layer\|embeddings]] | frozen |
| [[transformer-block\|Transformer blocks]] 1–11 | frozen |
| Transformer block 12 (`trf_blocks[-1]`) | **trainable** |
| Final [[layer-normalization\|LayerNorm]] (`final_norm`) | **trainable** |
| New output head 768 → 2 | **trainable** |

**How many parameters is that?** (Our arithmetic from the printed architecture on book
p. 185. The book doesn't print this total for chapter 6.)
- One transformer block: attention $4 \times (768\cdot768 + 768) = 2{,}362{,}368$; feed-forward
  $(768\cdot3072 + 3072) + (3072\cdot768 + 768) = 4{,}722{,}432$; two LayerNorms $2 \times 1{,}536 = 3{,}072$.
  Block total: **7,087,872**.
- Final LayerNorm: $768 + 768 = 1{,}536$. New head: $768\cdot2 + 2 = 1{,}538$.
- **Trainable total ≈ 7,090,946**, which is about **5.7%** of the 124,441,346 parameters
  (the full count after the head swap, book p. 332).

**Does freezing hurt accuracy?** Only a little. Exercise 6.2 drops the freezing loop
and fine-tunes the whole model. Test accuracy goes from **95.67%** to **96.67%**, the
"1% improved" result in Appendix C (book p. 306). Going the other way, the author
notes that training *only* the new head is technically enough, but unfreezing more
layers "noticeably improves the predictive performance" (book p. 186).

### Freezing + LoRA (Appendix E)
[[low-rank-adaptation|LoRA]] pushes freezing all the way. *Every* original weight gets
frozen, and only small new matrices $A$ and $B$ learn (book pp. 331–332):

```python
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")   # 124,441,346
for param in model.parameters():
    param.requires_grad = False
# ... count again -> 0
replace_linear_with_lora(model, rank=16, alpha=16)
# Total trainable LoRA parameters: 2,666,528
```

Same rule as before: **freeze first, then add the new layers**, so only $A$ and $B$
come out trainable. That's about 2.7M trainable weights instead of 124M,
124,441,346 / 2,666,528 ≈ 46.7 times fewer, which the book rounds to "almost 50×"
(book pp. 332, 336). The printed architecture shows every `Linear` (Q/K/V/out
projections, both feed-forward layers, and the 768 → 2 `out_head`) wrapped as a
`LinearWithLoRA` whose original `linear` part is "set to nontrainable" (book pp. 332–333).
LoRA training then still passes all of `model.parameters()` to AdamW
(`lr=8e-4, weight_decay=0.1`, book p. 334).

### Freezing vs. `torch.no_grad()`
They're easy to mix up:
- `requires_grad=False` is about **which parameters learn**. It's a permanent property of the tensor.
- `with torch.no_grad():` is about **whether this block of code builds a gradient graph at all**.
  You use it for evaluation and inference, like in `calc_accuracy_loader` (book p. 192).

## 🔴 Theory (Expert)
- **Optimizing on a coordinate subspace.** Freezing limits the optimization to the
  affine slice $\{\theta : \theta_{\text{frozen}} = \theta^{(0)}_{\text{frozen}}\}$ of parameter space. Here
  $\theta^{(0)}$ means the pretrained values. The partial gradient
  $\nabla_{\theta_{\text{train}}}\mathcal{L}$ is just the projection of the full gradient onto the
  trainable coordinates. So partial fine-tuning is plain [[gradient-descent]] on a
  lower-dimensional problem. (Compare LoRA, which instead restricts the *update* to a
  low-rank set, $\Delta W = \tfrac{\alpha}{r}AB$. See [[low-rank-approximation]].)
- **Backprop still has to go through what sits above the trainable layers.** Gradients
  for block 12 need $\partial \mathcal{L}/\partial \mathbf{h}$ from the head and final norm. Those are on top, so
  that's cheap. Blocks 1–11 are *below* every trainable parameter, though, so
  [[backpropagation]] can stop at the input of block 12. That's why freezing the lower
  layers saves backward-pass compute and not just optimizer memory. If you froze the
  top and trained the bottom instead, you'd still pay for the full backward pass.
- **Optimizer memory.** [[adamw-optimizer|AdamW]] keeps two moment tensors for every parameter it updates.
  Frozen parameters never get a `.grad`, so (standard PyTorch behaviour, not stated in
  the book) the optimizer never creates state for them. That's why the book can pass
  all of `model.parameters()` to the optimizer without any trouble (book p. 334).
- **The extreme case: train only the head.** If everything below the head is frozen,
  the final hidden vector $\mathbf{h}$ becomes a *fixed* feature map. Training the head
  with cross-entropy is then exactly multinomial [[logistic-regression]] on those
  features, which is a convex problem with a well-behaved optimum. This is the
  "linear probe" or [[feature-extraction]] style of [[transfer-learning]]. Unfreezing
  more layers makes the problem non-convex again, but lets the features themselves
  adapt. That's the accuracy gain the book saw (book p. 186; Appendix B experiments).
- **Why lower layers transfer.** The book's claim (book p. 185) matches the general
  deep-learning picture: early layers learn general-purpose features and later layers
  specialise. It's a tendency, not a theorem. Which layers are best to unfreeze
  depends on the task and is found by experiment, which is what Appendix B does.
- **Statistical caveat.** The +1 percentage point from Exercise 6.2 is 290/300 vs.
  287/300 correct test examples, so 3 examples. The binomial standard error at
  $p \approx 0.96$, $N = 300$ is about 1.1 points (see [[classification-accuracy]]). So
  that gain is within noise. Full fine-tuning helps "a bit", not decisively.
- **Regularization angle.** Keeping $\theta_{\text{frozen}}$ pinned to the pretrained values acts
  like a very strong prior on those parameters. It's the hard-constraint cousin of
  penalising $\lVert\theta - \theta^{(0)}\rVert^2$, see [[regularization]]. That helps against
  [[overfitting]] when the fine-tuning set is small (1,045 training texts here).

## Connections
- Builds on: [[model-parameters]], [[gradient]], [[gradient-descent]], [[backpropagation]], [[automatic-differentiation]], [[pytorch-nn-module]], [[pretrained-weights]]
- Leads to: [[classification-fine-tuning]], [[parameter-efficient-fine-tuning]], [[low-rank-adaptation]]
- Related: [[transfer-learning]], [[feature-extraction]], [[classification-head]], [[last-token-pooling]], [[regularization]], [[overfitting]], [[adamw-optimizer]], [[train-vs-eval-mode]]
- Used by: [[fine-tuning]], [[instruction-fine-tuning]] (LoRA variant, Exercise 7.4)
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 185–187 (sidebar "Fine-tuning selected layers vs. all layers", freezing loop, Listing 6.7, unfreezing `trf_blocks[-1]` and `final_norm`, Fig. 6.10)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book pp. 331–336 (freezing before LoRA; 124,441,346 → 0 → 2,666,528 trainable parameters)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise 6.2, book p. 306 (full fine-tuning: 96.67% vs. 95.67% test accuracy)
- Derived (not in the book): per-block parameter count 7,087,872 and trainable total ≈ 7.09M; the binomial standard-error remark.
