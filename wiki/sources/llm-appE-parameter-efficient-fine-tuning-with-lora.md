---
title: "LLM App. E — Parameter-efficient fine-tuning with LoRA"
type: source
tags: [llm, fine-tuning, lora, peft, linear-algebra, low-rank, source]
created: 2026-10-13
updated: 2026-10-13
---

# LLM App. E — Parameter-efficient fine-tuning with LoRA

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix E, **book pp. 322–336** (PDF pages 344–358)
- **Type:** book appendix. Source is *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025)
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13. The staging notes checked every listing, number and formula line by line. The display formulas in E.1 are images, so they were checked on rendered pages.
- *Page rule:* PDF page = book page + 22.
- **Related:** Appendix B has **no reference section for Appendix E**. Its last group is Appendix A (book pp. 298–299). The LoRA paper is cited inline in E.1 (book p. 323). Appendix C has one exercise that uses this appendix: **Exercise 7.4** (book p. 310).

This appendix is the "cheap fine-tuning" chapter. Chapter 6 fine-tuned GPT-2 to spot spam by
changing real model weights. Here we freeze *all* 124 million weights. Then we bolt a pair of
tiny matrices onto every linear layer and train only those. That is about 2.7 million
numbers. The result is almost the same accuracy, and it comes from one idea in linear algebra:
a weight *update* can be well approximated by a **low-rank** matrix.

## Section coverage (all of App. E accounted for)
| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Appendix intro (LoRA as a PEFT method; builds on Ch. 6 spam classification, also usable for Ch. 7 instruction fine-tuning) | 322 | ✅ [[low-rank-adaptation]], [[parameter-efficient-fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]] |
| E.1 | Introduction to LoRA ($W + \Delta W$, $\Delta W \approx AB$, Fig. E.1, distributive law, storage benefits) | 322–324 | ✅ [[low-rank-adaptation]], [[low-rank-approximation]], [[basis-and-rank]], [[matrix-multiplication]], [[backpropagation]], [[vector-space]], [[linear-layer]], [[edward-j-hu]] |
| E.2 | Preparing the dataset (Listings E.1–E.3: SMS spam CSVs, `SpamDataset`, `DataLoader`) | 324–325 | ✅ [[class-imbalance]], [[label-encoding]], [[train-validation-test-split]], [[tokenization]], [[byte-pair-encoding]], [[padding-and-truncation]], [[pytorch-dataset-and-dataloader]], [[mini-batch]], [[random-seed-reproducibility]] |
| E.3 | Initializing the model (Listing E.4: load GPT-2 124M, sanity-check generation, swap in a 2-class head, baseline accuracy) | 326–328 | ✅ [[pretrained-weights]], [[gpt-2]], [[gpt-model-configuration]], [[train-vs-eval-mode]], [[greedy-decoding]], [[classification-head]], [[classification-accuracy]], [[baseline-evaluation]] |
| E.4 | Parameter-efficient fine-tuning with LoRA (Listings E.5–E.7, Figs. E.2–E.5: `LoRALayer`, `LinearWithLoRA`, freezing, training, results) | 328–336 | ✅ [[low-rank-adaptation]], [[lora-rank-and-alpha]], [[weight-initialization]], [[layer-freezing]], [[parameter-count-and-memory]], [[adamw-optimizer]], [[weight-decay]], [[training-and-validation-loss]], [[overfitting]] |
| App. B | References for App. E | — | — (none exist. The only citation is Hu et al. inline on p. 323, see below) |
| App. C | Exercise 7.4 (instruction fine-tuning with LoRA) | 310 | ✅ see "Exercises" below |

> ✅ **Every section of App. E maps to at least one wiki page.** None of E.1–E.4 has numbered
> subsections (no E.1.1 etc.). The appendix has no end-of-chapter summary and no exercise boxes
> of its own.

## TL;DR (5 bullets)
- **The problem:** full fine-tuning learns an update $\Delta W$ with the *same shape* as each weight matrix $W$, so $W_{\text{updated}} = W + \Delta W$ touches every parameter. [[parameter-efficient-fine-tuning|PEFT]] methods update only a small subset.
- **The LoRA trick** (Hu et al., 2021, [[edward-j-hu]]): approximate the update by a product of two thin matrices, $\Delta W \approx AB$, with $A \in \mathbb{R}^{d_{in}\times r}$ and $B \in \mathbb{R}^{r\times d_{out}}$. The product has [[basis-and-rank|rank]] at most $r$, so this is a [[low-rank-approximation]]. The update is limited to a small subspace of the weight space.
- **Keep the paths separate:** because matrix multiplication is distributive, $x(W + AB) = xW + xAB$. The frozen layer and the LoRA path run in parallel and their outputs are added. You never build the big matrix $AB$, and you can save one small adapter per customer or task instead of a full copy of the LLM.
- **In code:** `LoRALayer` computes $\frac{\alpha}{r}\,xAB$. $A$ uses Kaiming-uniform init and $B$ starts at **zeros**, so LoRA is a no-op at the start. `LinearWithLoRA` wraps every `nn.Linear` (Q/K/V, `out_proj`, both feed-forward layers, `out_head`). See [[lora-rank-and-alpha]] for the two knobs.
- **Result on spam classification:** with rank = alpha = 16, trainable parameters drop from **124,441,346 to 2,666,528** (about 47×, the book says "almost 50×"). Final accuracy is **100% train / 96.64% val / 98.00% test**, close to Chapter 6's full fine-tuning.

## Key concepts introduced
**The idea (§E.1)**
- [[low-rank-adaptation]] — LoRA: freeze $W$, learn $\Delta W \approx AB$ with a small inner dimension $r$
- [[parameter-efficient-fine-tuning]] — the family of methods that train only a few parameters
- [[low-rank-approximation]] — replace a big matrix by a product of thin ones. Truncated SVD is the classical best version (Eckart–Young)
- [[basis-and-rank]] — $\operatorname{rank}(AB) \le r$. This is *why* the update lives in an $r$-dimensional subspace
- [[matrix-multiplication]] — the distributive law $x(W+\Delta W) = xW + x\Delta W$ and associativity $x(AB) = (xA)B$
- [[vector-space]] — "a smaller-dimensional subspace of the total weight-parameter space"
- [[backpropagation]], [[gradient]], [[loss-function]] — how $\Delta W$ (or $A$, $B$) is learned
- [[fine-tuning]], [[pretrained-weights]] — the frozen starting point

**Setup reused from Ch. 6 (§E.2–E.3)**
- [[classification-fine-tuning]], [[text-classification]] — spam vs. ham on the UCI SMS Spam Collection
- [[class-imbalance]] — undersample "ham" to the size of "spam"
- [[label-encoding]] — ham → 0, spam → 1
- [[train-validation-test-split]] — 70 / 10 / 20 %
- [[tokenization]], [[byte-pair-encoding]], [[padding-and-truncation]], [[end-of-text-token]] — GPT-2 tiktoken encoding, pad to the training max length
- [[pytorch-dataset-and-dataloader]], [[mini-batch]], [[random-seed-reproducibility]] — batch size 8, shuffle, `drop_last`, seed 123
- [[gpt-2]], [[gpt-model-configuration]] — the 124M model and its config dictionary
- [[train-vs-eval-mode]], [[greedy-decoding]] — sanity-check text generation
- [[classification-head]], [[last-token-pooling]], [[logits]], [[argmax]] — the new 768 → 2 output layer
- [[classification-accuracy]], [[baseline-evaluation]] — about 50% before training (chance for 2 balanced classes)

**LoRA in code (§E.4)**
- [[lora-rank-and-alpha]] — rank $r$ (capacity vs. efficiency) and scale $\alpha$ (strength of the update, applied as $\alpha/r$)
- [[weight-initialization]] — Kaiming-uniform $A$, zero $B$ ⇒ $AB = 0$ at the start
- [[linear-layer]], [[linear-mapping]], [[affine-space]] — a LoRA layer computes $x(W + \tfrac{\alpha}{r}AB) + b$
- [[pytorch-nn-module]] — `nn.Parameter`, `named_children`, recursive layer replacement
- [[layer-freezing]] — `requires_grad = False` on all original weights, done *before* adding LoRA
- [[parameter-count-and-memory]] — $r(d_{in}+d_{out})$ instead of $d_{in}d_{out}$ per layer
- [[transformer-block]], [[multi-head-attention]], [[feed-forward-network]] — where the adapted linear layers sit (Fig. E.4)
- [[adamw-optimizer]], [[weight-decay]], [[learning-rate]] — lr = 8e-4, weight_decay = 0.1
- [[training-loop]], [[epoch]], [[training-and-validation-loss]], [[cross-entropy-loss]] — 5 epochs with `train_classifier_simple`
- [[overfitting]], [[overfitting-and-underfitting]] — the small gap between 100% train and ~97% val/test accuracy
- [[gpu-computing]] — ~15 min on an M3 MacBook Air vs. under 30 s on a V100/A100

## Notable results / numbers
- **Core formulas (E.1, pp. 322–324):**
  $$W_{\text{updated}} = W + \Delta W, \qquad \Delta W \approx AB, \qquad W_{\text{updated}} = W + AB$$
  $$x(W+\Delta W) = xW + x\Delta W, \qquad x(W+AB) = xW + xAB$$
- **Forward pass in code (Listings E.5–E.6, pp. 329–330):**
  $$y = \underbrace{xW + b}_{\text{frozen Linear}} + \underbrace{\tfrac{\alpha}{r}\,xAB}_{\text{LoRALayer}} = x\Big(W + \tfrac{\alpha}{r}AB\Big) + b$$
  Shapes: $A$ is `(in_dim, rank)`, $B$ is `(rank, out_dim)`.
- **Zero init = no-op:** $B = 0 \Rightarrow AB = 0$. So the accuracies before LoRA training are **46.25% / 45.00% / 48.75%** (train/val/test, 10 batches each). These are identical to E.3 and Ch. 6 (p. 334). *Note (not in the book):* on step 1 the gradient with respect to $A$ is also zero, because it is multiplied by $B$. The gradient with respect to $B$ is non-zero, so learning still starts.
- **Parameter count (p. 332):** 124,441,346 trainable → 0 after freezing → **2,666,528** after `replace_linear_with_lora(model, rank=16, alpha=16)`. The breakdown (derived in the notes, it matches the book exactly) is per block $4\cdot16\cdot1536 + 2\cdot16\cdot3840 = 221{,}184$. Then $\times 12 = 2{,}654{,}208$, plus `out_head` $16\cdot770 = 12{,}320$.
- **Defaults (p. 332):** rank = 16 and alpha = 16 are good defaults. Alpha is usually **half, double, or equal to** the rank.
- **Training (Listing E.7, p. 334):** AdamW, lr = 8e-4, weight_decay = 0.1, 5 epochs, eval_freq = 50, eval_iter = 5, seed 123. The loss falls from 3.820/3.462 (train/val) at step 0 to 0.019/0.252 at step 600. It took **12.10 minutes** on the author's machine.
- **Speed trade-off (p. 335):** here LoRA was *slower* than Ch. 6, because the extra forward-pass compute dominates on a small model. On large models, where backpropagation dominates, LoRA usually makes training *faster*. Exercise 7.4 shows this: it was 28% faster.
- **Final accuracy (p. 336):** **100.00% train, 96.64% val, 98.00% test**. This is very close to full fine-tuning while training only ~2.7M of 124M weights.
- ⚠️ **Book inconsistencies found during validation:**
  - p. 336: the printed output says test = 98.00%, but the paragraph says 97.33%.
  - Fig. E.5 caption says "six epochs", but the code uses `num_epochs = 5`.
  - Fig. E.3 caption says "weight update matrix (W)". It should say ΔW.
  - Listing E.1 imports from `ch06`, but Listing E.2 imports from `chapter06`.
  - p. 327 has the typo "loaded corrected".

## Exercises
Appendix E has no exercises of its own. One exercise in Appendix C uses it:
- **Exercise 7.4 — Instruction fine-tuning with LoRA** (Ch. 7, solution on book p. 310).
  *Task:* repeat Chapter 7's instruction fine-tuning, but with LoRA.
  *Solution idea:* run `from appendix_E import LoRALayer, LinearWithLoRA, replace_linear_with_lora`. After loading the model (§7.5), freeze all parameters, then call `replace_linear_with_lora(model, rank=16, alpha=16)` and move the model to the device. Count trainable parameters before and after to confirm that only the LoRA matrices train.
  *Result:* on an Nvidia L4 GPU it took **1.30 min vs. 1.80 min** for full fine-tuning (about **28% faster**, $1 - 1.30/1.80 = 27.8\%$). The Ollama Llama 3 judge gave the same quality score, **about 50**. This is the "bigger job, LoRA wins on speed" case from p. 335. See [[instruction-fine-tuning]], [[llm-as-a-judge]], [[local-llm-inference-ollama]].

## References cited
- **Hu et al. (2021)**, "LoRA: Low-Rank Adaptation of Large Language Models", https://arxiv.org/abs/2106.09685. This paper proposed LoRA (book p. 323). The first author is [[edward-j-hu]]. His name and the year come from the arXiv record, not from the book.
- **UCI Machine Learning Repository — SMS Spam Collection** (dataset 228), https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip (book p. 324).
- **OpenAI GPT-2** weights (124M/355M/774M/1558M configs listed on p. 327). The book gives no paper citation in this appendix.
- **Background only. The appendix uses these but does not cite them:** Kaiming/He initialization (`kaiming_uniform_`, He et al. 2015), the [[adamw-optimizer]] (Loshchilov & Hutter), and Eckart–Young / truncated SVD as the classical [[low-rank-approximation]].
- Appendix B has **no** Appendix E section. The nearest related entry is GaLore (Zhao et al., 2024, Ch. 5 refs, p. 294; [[galore]], [[jiawei-zhao]]). It is another *low-rank* memory-saving idea, but it projects the gradients instead of the weight update.

## What this changes in the wiki
- **New concept pages:** [[low-rank-adaptation]], [[parameter-efficient-fine-tuning]] and [[lora-rank-and-alpha]], plus the person page [[edward-j-hu]].
- **Bridge from MML to LLMs:** LoRA is the clearest place in the LLM book where the linear algebra from MML pays off directly. [[basis-and-rank]] (MML §2.6) and [[low-rank-approximation]] (MML §4.6, Eckart–Young via SVD) now each get an "Used by: LoRA" connection and a worked parameter-count example: $d^2$ vs. $2dr$, and for GPT-2 2.67M vs. 124M.
- **Refinement, not a contradiction:** MML's low-rank approximation is a *best-fit* truncation of a known matrix (SVD). LoRA never computes $\Delta W$ or its SVD. It *learns* the low-rank factors $A$ and $B$ directly by gradient descent. The rank limit is a modeling assumption (the idea that fine-tuning updates have low "intrinsic rank"), not an optimal approximation. Both pages note this difference.
- **Ties together Ch. 6 and Ch. 7:** [[classification-fine-tuning]] and [[instruction-fine-tuning]] can now link to [[parameter-efficient-fine-tuning]] as the cheaper alternative to full or last-layer fine-tuning. [[layer-freezing]] gets its most extreme case: freeze everything, then add new trainable adapters.
- **Book errata** (97.33% vs 98.00%, "six epochs", "(W)" for ΔW, `ch06`/`chapter06`) are recorded above. No other wiki page contradicts this appendix.
