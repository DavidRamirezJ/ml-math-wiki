---
title: "LLM Ch.6 — Fine-tuning for Classification"
type: source
tags: [llm, fine-tuning, classification, source]
sources: []
created: 2026-10-13
updated: 2026-10-13
---

# LLM Ch.6 — Fine-tuning for Classification

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 6, **book pp. 169–203** (PDF pp. 191–225). Related back matter: Appendix B references for Ch. 6, **book pp. 295–296**. Appendix C solutions for Exercises 6.1–6.3, **book pp. 306–307**.
- **Type:** book chapter. *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025).
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13. Every section was checked line by line against the extracted text, and the key figures (6.1, 6.5, 6.13, 6.14) were checked against rendered pages.
- *Page rule:* PDF page = book page + 22.

This chapter takes the pretrained GPT-2 model from Ch. 5 and turns it into a **spam classifier**. The idea is simple: keep the pretrained "brain", swap the output layer for a tiny 2-way layer, and train only the top of the network on labeled SMS messages. After 5 epochs (about 6 minutes on a laptop) it reaches **95.67% test accuracy**.

## Section coverage (all of Ch. 6 accounted for)

None of the sections in Ch. 6 has numbered subsections (there is no 6.x.y heading), so every row below is a full section.

| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Chapter intro ("This chapter covers", Fig. 6.1) | 169–170 | [[fine-tuning]], [[classification-fine-tuning]], [[foundation-model]], [[llm-building-roadmap]] |
| 6.1 | Different categories of fine-tuning | 170–171 | [[classification-fine-tuning]], [[instruction-fine-tuning]], [[text-classification]], [[supervised-learning]] |
| 6.2 | Preparing the dataset | 172–175 | [[text-classification]], [[class-imbalance]], [[label-encoding]], [[train-validation-test-split]], [[random-seed-reproducibility]] |
| 6.3 | Creating data loaders | 175–181 | [[padding-and-truncation]], [[end-of-text-token]], [[special-tokens]], [[pytorch-dataset-and-dataloader]], [[mini-batch]], [[context-length]] |
| 6.4 | Initializing a model with pretrained weights | 181–183 | [[pretrained-weights]], [[transfer-learning]], [[gpt-2]], [[gpt-model-configuration]], [[greedy-decoding]], [[zero-shot-learning]] |
| 6.5 | Adding a classification head | 183–190 | [[classification-head]], [[layer-freezing]], [[last-token-pooling]], [[causal-attention-mask]], [[linear-layer]], [[logits]], [[transfer-learning]] |
| 6.6 | Calculating the classification loss and accuracy | 190–194 | [[classification-accuracy]], [[surrogate-loss]], [[softmax]], [[argmax]], [[monotonic-function]], [[cross-entropy-loss]] |
| 6.7 | Fine-tuning the model on supervised data (+ sidebar "Choosing the number of epochs") | 195–200 | [[training-loop]], [[epoch]], [[adamw-optimizer]], [[weight-decay]], [[training-and-validation-loss]], [[overfitting-and-underfitting]], [[regularization]], [[dropout]] |
| 6.8 | Using the LLM as a spam classifier | 200–202 | [[train-vs-eval-mode]], [[model-checkpointing]], [[last-token-pooling]], [[padding-and-truncation]] |
| — | Summary | 202–203 | [[classification-fine-tuning]], [[classification-accuracy]], [[cross-entropy-loss]] |
| — | Exercises 6.1–6.3 (in §6.3, §6.5) | 179, 186, 190 | see *Exercises* below |
| App. B | References, Ch. 6 | 295–296 | see *References cited* below |
| App. C | Solutions 6.1–6.3 | 306–307 | see *Exercises* below |

Sidebars: "Choosing the right approach" (p. 171, §6.1), "Output layer nodes" (p. 183, §6.5), "Fine-tuning selected layers vs. all layers" (p. 185, §6.5), and "Choosing the number of epochs" (p. 199, §6.7). Each one is covered in the pages linked for its section.

## TL;DR (5 bullets)
- **Two kinds of fine-tuning.** *Instruction* fine-tuning makes a generalist that follows prompts. *Classification* fine-tuning makes a specialist that can only output a fixed set of labels. The specialist needs less data and less compute (§6.1).
- **Data first.** The book uses the UCI SMS Spam Collection (4,825 ham, 747 spam). It **undersamples** the data to 747/747, encodes the labels as ham→0 and spam→1, splits 70/10/20, and pads every message to 120 tokens with token ID 50256 (§6.2–6.3).
- **Swap the head, freeze the body.** The 768→50,257 language-model output layer is replaced by a 768→2 [[classification-head]]. The book freezes everything, then unfreezes only the **last transformer block + final LayerNorm + new head** (§6.5).
- **Read the last token.** Because of the [[causal-attention-mask]], only the last token has seen the whole message, so its 2 logits are used for prediction and loss ([[last-token-pooling]], §6.5).
- **Train on cross-entropy, judge on accuracy.** Accuracy is not differentiable, so [[cross-entropy-loss]] serves as a [[surrogate-loss]]. Accuracy goes from about 46% (random) to **97.21% train / 97.32% val / 95.67% test** after 5 epochs of AdamW (§6.6–6.7).

## Key concepts introduced
**Owned by this chapter (new pages):**
- [[text-classification]]: assign one label from a fixed set to a piece of text (spam / not spam).
- [[class-imbalance]]: one class is much more common than the other (86.6% ham); fixed here by undersampling.
- [[label-encoding]]: map string labels to integers, $\{\text{ham},\text{spam}\}\to\{0,1\}$.
- [[overfitting-and-underfitting]]: read from the gap between the training and validation curves (Figs. 6.16–6.17).
- [[transfer-learning]]: start training from pretrained parameters, $\theta_0=\theta_{\text{pretrained}}$, and not from random weights.
- [[classification-head]]: the new affine output layer $\mathbf z = \mathbf h W + \mathbf b$, $\mathbb R^{768}\to\mathbb R^{2}$.
- [[layer-freezing]]: `requires_grad = False`, so the optimizer only updates a subset of the parameters.
- [[last-token-pooling]]: use the final position's output, `outputs[:, -1, :]`, as the summary of the sequence.
- [[classification-accuracy]]: $\frac1N\sum_i \mathbb 1[\hat y_i = y_i]$.
- [[surrogate-loss]]: a differentiable stand-in (cross-entropy) for a non-differentiable goal (accuracy).

**Reused / extended from earlier chapters:**
[[fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[foundation-model]], [[supervised-learning]], [[pretrained-weights]], [[gpt-2]], [[gpt-model-configuration]], [[train-validation-test-split]], [[random-seed-reproducibility]], [[padding-and-truncation]], [[end-of-text-token]], [[special-tokens]], [[token-id]], [[pytorch-dataset-and-dataloader]], [[mini-batch]], [[context-length]], [[causal-attention-mask]], [[linear-layer]], [[logits]], [[softmax]], [[argmax]], [[monotonic-function]], [[cross-entropy-loss]], [[training-loop]], [[epoch]], [[adamw-optimizer]], [[weight-decay]], [[training-and-validation-loss]], [[regularization]], [[dropout]], [[train-vs-eval-mode]], [[model-checkpointing]], [[greedy-decoding]], [[zero-shot-learning]], [[bert]].

**Math links (MML pages):** [[linear-mapping]], [[matrix]], [[affine-space]], [[loss-function]], [[maximum-likelihood-estimation]], [[empirical-risk-minimization]], [[gradient]], [[gradient-descent]], [[stochastic-gradient-descent]], [[backpropagation]], [[automatic-differentiation]], [[probability-distribution]], [[expected-value-and-mean]], [[cross-validation]], [[overfitting]].

## Notable results / numbers
**Dataset (§6.2–6.3)**
- Raw: 5,572 messages, of which **4,825 ham** and **747 spam** (p. 174). A classifier that always says "ham" would already score $4825/5572\approx 86.6\%$ (derived), which is why the book balances the data.
- Balanced: 747 + 747 = 1,494. Split 70/10/20 gives **1,045 / 149 / 300** (derived from $\lfloor N f\rfloor$).
- Longest training message = **120 tokens**. The model supports up to **1,024** (p. 178). Pad ID = **50256** (`<|endoftext|>`) (p. 176).
- Batch size 8 gives **130 / 19 / 38** batches. The training loader uses `drop_last=True`, so $\lfloor1045/8\rfloor=130$. The others use $\lceil\cdot\rceil$ (p. 181).

**Model (§6.4–6.5)**
- GPT-2 small (124M): `emb_dim` 768, 12 layers, 12 heads, vocab 50,257, `drop_rate` 0.0, `qkv_bias` True. Medium/large/XL configs are also listed (1024/24/16, 1280/36/20, 1600/48/25). All four have a head dimension of 64 (p. 182).
- Old head: 768 × 50,257 = **38,597,376** params, no bias. New head: 768 × 2 + 2 = **1,538** params (derived).
- The pretrained model **cannot** classify spam from a prompt: it just echoes the instruction (p. 183). This is why it needs fine-tuning.
- "Do you have time" → token IDs `[5211, 345, 423, 640]` → output shape `[1, 4, 2]`; last-token logits `[-3.5983, 3.9902]` (p. 188).
- Fig. 6.12 last-row attention weights: 0.27 / 0.24 / 0.24 / 0.25. Only "time" attends to every token (p. 190).

**Before training (§6.6)**
- Accuracy (10 batches): **46.25% / 45.00% / 48.75%** (train / val / test). That is about the 50% chance level (p. 193).
- Loss (5 batches): **2.453 / 2.583 / 2.322**. That is far above the uniform-guess loss $\ln 2\approx0.693$ (derived), which means the random head is *confidently wrong*.

**Training (§6.7)**
- AdamW, `lr = 5e-5`, `weight_decay = 0.1`, 5 epochs, `eval_freq = 50`, `eval_iter = 5`, seed 123 (p. 197).
- Runtime: about 6 min on an M3 MacBook Air, under 30 s on a V100/A100. The book's run printed **5.65 min** (p. 197).
- Loss falls from 2.153/2.392 (step 0) to **0.083/0.074** (step 600). Per-epoch accuracy (5 batches): 70→82.5→90→100→100% train, 72.5→85→90→97.5→97.5% val (p. 197).
- Full-dataset accuracy: **97.21% train, 97.32% val, 95.67% test** (p. 200). Validation is usually a bit above test because hyperparameters were tuned on validation. To shrink the gap, raise `drop_rate` or `weight_decay` (p. 200).

**Inference (§6.8)**
- "You are a winner … $1000 cash or a $2000 award." → **spam**. "Hey, just wanted to check if we're still on for dinner tonight?" → **not spam** (p. 202).
- The model is saved as `review_classifier.pth` with `torch.save(model.state_dict(), …)` (p. 202).

## Exercises
| # | Task (book page) | Solution idea and result (App. C) |
|---|---|---|
| 6.1 | *Increasing the context length*: pad inputs to the full 1,024 tokens and see what happens (p. 179) | Use `max_length=1024` for all three `SpamDataset`s. Test accuracy **drops to 78.33%** (from 95.67%): the many extra pad tokens hurt (p. 306). |
| 6.2 | *Fine-tuning the whole model*: train every layer, not only the last block + head (p. 186) | Delete the `requires_grad = False` freeze loop. Test accuracy **rises to 96.67%** (+1 point), at a much higher compute cost (p. 306). |
| 6.3 | *First vs. last token*: train on the first output token, not the last (p. 190) | Replace `[:, -1, :]` with `[:, 0, :]` everywhere. Test accuracy **falls to 75.00%**. Under the causal mask the first token sees only itself, so it knows almost nothing about the message (p. 307). |

These three results together form a neat little ablation study: **last token > first token**, **fit-to-data padding > max padding**, and **full fine-tuning ≈ partial fine-tuning** (only +1 point for far more trainable parameters).

## References cited (Appendix B, pp. 295–296)
1. Raschka, "Using and Finetuning Pretrained Transformers" (mng.bz/VxJG): the types of fine-tuning.
2. Raschka, "Finetuning Large Language Models" (mng.bz/x28X): the types of fine-tuning.
3. Book code, "Additional spam classification experiments" (mng.bz/AdJx): more experiments, including first-token vs last-token fine-tuning.
4. Raschka, "Losses Learned—Optimizing Negative Log-Likelihood and Cross-Entropy in PyTorch" (mng.bz/ZEJA): a binary task can use **one** output node instead of two.
5. Raschka, "Finetuning Large Language Models" (mng.bz/RZJv): which layers to tune. Tuning the last block + head gives a big gain.
6. "Imbalanced-Learn User Guide" (mng.bz/2KNa): methods for [[class-imbalance]].
7. Email Spam Classification Dataset (mng.bz/1GEq): a larger email-spam CSV in the same format.
8. [[jacob-devlin]] et al. (2018), "BERT", arXiv 1810.04805: encoder models ([[bert]]) as strong classifiers.
9. Liu et al. (2019), "RoBERTa", arXiv 1907.11692: a second encoder-based alternative.
10. Book code, "Additional Experiments Classifying the Sentiment of 50k IMDb Movie Reviews" (mng.bz/PZJR): a model comparison on IMDb sentiment.
11. Li et al. (2023), "Label Supervised LLaMA Finetuning", arXiv 2310.01208: **removing the causal mask** during classification fine-tuning can help.
12. BehnamGhader et al. (2024), "LLM2Vec", arXiv 2404.05961: decoder LLMs become strong text encoders once the causal mask is removed.

> Note: the §6.5 sidebar (p. 183) prints the "Losses Learned" link as `mng.bz/NRZ2`, but Appendix B (p. 295) prints `mng.bz/ZEJA`. Appendix B also uses the title "Finetuning Large Language Models" for two different links (x28X and RZJv).

## Errata / quirks found during validation
- **Fig. 6.14 vs. the text (pp. 191–192).** The figure shows "You won the lottery" with logits `[3.5983, −3.9902]` → label 0, but the printed tensor on p. 192 is `[[−3.5983, 3.9902]]` → label 1. The signs are swapped. The figure is only illustrative, and both of its predictions are wrong on purpose, because the model is untrained.
- **Summary (p. 202)** says the old output layer had "50,256" nodes. The vocabulary size is **50,257**: IDs run 0–50256, so 50,256 is the largest ID, not the count. The wiki uses 50,257.
- **Fig. 6.12 (p. 190)** calls its values "attention scores", but the rows sum to 1. In Ch. 3 terms they are normalized [[attention-weights]].
- **Fig. 6.16 (p. 198)**: the caption says "dashed" for validation, but the code uses dash-dot (`"-."`).
- **`classify_review` (p. 201)** has the default `max_length=None`, but `None` would crash inside `min(...)`. Always pass `train_dataset.max_length`.

## What this changes in the wiki
- **Adds 10 new concept pages** (listed above). This makes classification fine-tuning the first complete "pretrained model → useful task" pipeline in the wiki.
- **Connects the LLM book to the MML math.** The classification head is a [[linear-mapping]] / [[affine-space]] map. Cross-entropy is [[maximum-likelihood-estimation]] of a categorical [[probability-distribution]]. Accuracy is the complement of the mean 0-1 loss in [[empirical-risk-minimization]]. Freezing restricts [[gradient-descent]] to a subset of the parameters. The train/val/test methodology refines [[cross-validation]] and [[overfitting]].
- **New page vs. older page.** [[overfitting-and-underfitting]] (owned here) is the full three-tier treatment. The older [[overfitting]] page from MML Ch. 8 should link to it. They do not contradict each other: both define overfitting as a training/validation (or test) gap.
- **Sets up Ch. 7.** [[instruction-fine-tuning]] is introduced here only by contrast (Fig. 6.2, p. 170). Ch. 7 fills it in.
- **No contradictions** with existing pages. The book quirks above (50,256 vs 50,257, the sign flip in Fig. 6.14, "scores" vs "weights") are recorded here so the concept pages can cite them instead of silently overwriting anything.
