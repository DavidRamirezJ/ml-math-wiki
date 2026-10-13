---
title: Parameter-Efficient Fine-Tuning (PEFT)
type: concept
tags: [llm, fine-tuning, transfer-learning, optimization]
difficulty: intermediate
sources: [llm-appE-parameter-efficient-fine-tuning-with-lora, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Parameter-Efficient Fine-Tuning (PEFT)

> **One-liner:** PEFT means adapting a big pretrained model to a new task by training only a tiny slice of its numbers, leaving the rest frozen.

## 🟢 Intuition (Dummy)
A pretrained LLM already "knows" a lot of language. To teach it one new job, like spotting spam
SMS messages, you don't need to rewrite its whole brain. That is slow, needs lots of GPU memory,
and gives you a full new copy of the model for every job.

PEFT says: keep the big model frozen, like a finished book, and add or train only a small set of
"sticky notes". Training is lighter, and you save only the sticky notes per task. The book's
example: train **2.7 million** numbers instead of **124 million**, and still classify spam with
about 97–98% test accuracy (Appendix E).

The book introduces PEFT through its most popular member, [[low-rank-adaptation|LoRA]]
(book p. 322).

## 🟡 Mechanics (Practitioner)

### The general recipe
Split the parameters into a frozen part and a small trainable part:

$$\theta = \theta_0 + \Delta\theta(\phi),\qquad \phi^\star = \arg\min_{\phi}\ \mathcal L\big(\theta_0 + \Delta\theta(\phi)\big),\qquad |\phi| \ll |\theta_0|$$

Symbol by symbol:
- $\theta$ = all the weights the model actually uses.
- $\theta_0$ = the frozen [[pretrained-weights]] (never changed).
- $\phi$ ("phi") = the small set of **trainable** parameters.
- $\Delta\theta(\phi)$ = the change to the weights that $\phi$ produces.
- $\arg\min_\phi$ = "the value of $\phi$ that makes the following smallest".
- $\mathcal L$ = the task [[loss-function]] (for spam classification, [[cross-entropy-loss]]).
- $|\phi| \ll |\theta_0|$ = "the number of trainable values is much, much smaller than the number of
  frozen ones".

[[gradient-descent|Gradient descent]] only updates $\phi$: $\phi \leftarrow \phi - \eta\,\nabla_\phi\mathcal L$
($\eta$ = [[learning-rate]], $\nabla_\phi\mathcal L$ = the [[gradient]] with respect to $\phi$ only).

### LoRA as the instance the book builds
For every [[linear-layer]] with frozen weight $W$, LoRA sets $\Delta W = \frac{\alpha}{r}AB$ with
$A\in\mathbb R^{d_{in}\times r}$, $B\in\mathbb R^{r\times d_{out}}$, so $\phi = \{A, B\}$ for all layers.

In PyTorch, "frozen" just means `requires_grad = False`:

```python
for param in model.parameters():
    param.requires_grad = False           # freeze theta_0
replace_linear_with_lora(model, rank=16, alpha=16)   # add phi
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Book numbers for GPT-2 small (book p. 332):
- trainable before freezing: **124,441,346**
- after freezing: **0**
- after adding LoRA ($r=\alpha=16$): **2,666,528**

Trainable fraction: $2{,}666{,}528 / 124{,}441{,}346 \approx 0.0214$, about **2.1%**
("almost 50×" fewer).

### Why bother? (book pp. 322, 324, 335)
1. **Less compute and memory** for fine-tuning large models. Only trainable parameters need
   gradients and [[adamw-optimizer|AdamW]] optimizer state.
2. **Cheap storage and swapping.** The base model stays unchanged. You save only the small
   adapter per customer or application, instead of "multiple complete versions" of the LLM.
3. **Speed depends on size.** In the book's small spam example, LoRA training was slightly slower
   (12.10 min) because of the extra forward-pass work. For larger models it is typically faster. In
   Exercise 7.4 (GPT-2 medium, instruction tuning on an L4 GPU) it was ≈ 28% faster: 1.30 vs 1.80 min
   (book p. 310).

### A cousin in chapter 6: partial fine-tuning
Chapter 6 also trains only part of the model: the new [[classification-head]], the last
[[transformer-block]], and the final [[layer-normalization|LayerNorm]], with all other layers frozen
([[layer-freezing]]). That is "parameter-efficient" in spirit, but it changes existing weights
directly. LoRA instead adds a separate low-rank update to *every* Linear layer and leaves all
original weights untouched.

## 🔴 Theory (Expert)
- **Why so few parameters can be enough** (background, not in the book). Fine-tuning works in a
  very low-dimensional "intrinsic" subspace of parameter space (Aghajanyan et al. 2020). Random
  projections of the update into a few hundred or thousand dimensions already recover most of the
  full fine-tuning performance. LoRA's low-rank structure ([[basis-and-rank]],
  [[low-rank-approximation]]) is one structured way to pick such a subspace. The book states the
  idea informally: limit adjustments to "a smaller dimensional subspace" that captures "the most
  influential directions" (book p. 322).
- **Taxonomy** (general knowledge, beyond the book):
  - *selective*: train a chosen subset of existing weights (last layers, biases only: BitFit);
  - *additive*: insert new small modules (adapters) or learned soft prompts (prompt / prefix tuning);
  - *reparametrized*: express the update in a compact form (LoRA, DoRA; QLoRA = LoRA on a
    4-bit-quantized frozen base).
- **Regularization effect.** Restricting the hypothesis class to $\theta_0 + \{\Delta\theta(\phi)\}$
  acts like strong [[regularization]] toward the pretrained solution. This helps on small datasets
  but can underfit tasks that need large changes. The book's train 100% / val 96.64% shows some
  [[overfitting]] remains even so.
- **Memory accounting.** With AdamW each trainable parameter costs the weight + gradient + two
  moments. Frozen parameters cost only the weight (activations for backprop are still needed). So the
  memory saving scales with $|\theta_0|/|\phi|$ for optimizer state, not for activations
  ([[parameter-count-and-memory]]).
- Not the same as [[galore]]: GaLore trains *all* weights and compresses only the gradients /
  optimizer state with a low-rank projection.

## Connections
- Builds on: [[fine-tuning]], [[transfer-learning]], [[pretrained-weights]], [[layer-freezing]], [[gradient-descent]], [[model-parameters]]
- Leads to: [[low-rank-adaptation]], [[lora-rank-and-alpha]]
- Used by: [[classification-fine-tuning]], [[instruction-fine-tuning]], [[custom-llm]]
- Related: [[galore]], [[parameter-count-and-memory]], [[basis-and-rank]], [[regularization]]
- People: [[edward-j-hu]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix E intro and §E.1, book pp. 322–324 (PDF pp. 344–346): PEFT named, LoRA as the most widely used PEFT method, storage benefit.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book pp. 331–336 (PDF pp. 353–358): freezing, parameter counts (p. 332), timing trade-off (p. 335), accuracies (p. 336).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise 7.4 solution, book p. 310.
- Intrinsic-dimension result, the PEFT taxonomy, and the memory accounting are background / general knowledge, not from the book.
