---
title: Fine-Tuning
type: concept
tags: [llm, training, transfer-learning, supervised-learning, deep-learning]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Fine-Tuning

> **One-liner:** Fine-tuning takes a model that already "knows language" from pretraining and trains it a bit more on a small, labeled dataset so it gets good at one particular job.

## 🟢 Intuition (Dummy)
Think of a medical school graduate. Years of general study (that's
[[pretraining]]) gave them a broad base. A short residency in cardiology (that's
fine-tuning) turns them into a heart specialist — far faster than starting from
zero, because they're not relearning anatomy, just sharpening one skill.

An LLM works the same way. After pretraining, the [[foundation-model]] can finish your
sentences, but it's clumsy at following a request like "Fix the grammar in this text"
(book p. 205). Fine-tuning means **further training the pretrained model on a narrower
dataset**, specific to a task or domain (book p. 6) — and, crucially, this dataset is
**labeled** (book p. 7). Because the model starts from good weights instead of random
ones, a small dataset and modest compute are enough (book p. 12).

The book's two flavours:
- **Classification fine-tuning** — teach it to pick from a fixed set of labels, e.g. "spam" / "not spam."
- **Instruction fine-tuning** — teach it to answer requests, like a chat assistant.

## 🟡 Mechanics (Practitioner)

### The two categories (§1.3, book p. 7; §6.1, book pp. 170–171)

| | [[instruction-fine-tuning]] | [[classification-fine-tuning]] |
|---|---|---|
| Labeled data | instruction–response pairs | texts + class labels |
| Example | "Convert 45 kilometers to meters." → "45 kilometers is 45000 meters." | "You are a winner you have been specially selected to receive \$1000 cash…" → **Spam** |
| Result | generalist: personal assistant / chat model | specialist: classifier |
| Data / compute | larger datasets, more compute | less data, less compute |
| Limitation | — | can only output classes seen in training |

In the book's roadmap (Figure 1.9, book p. 14) these are steps **8** (dataset with
class labels → classifier, Ch. 6) and **9** (instruction dataset → personal assistant,
Ch. 7), both branching off the foundation model.

### The common recipe
1. **Load a pretrained model** — e.g. OpenAI's GPT-2 weights via
   `download_and_load_gpt2` and `load_weights_into_gpt` (§5.5). It's the starting
   point: "we must first load a pretrained GPT model that we want to fine-tune"
   (book p. 226).
2. **Adapt the head if needed.**
3. **Keep training** with gradient descent on the new labeled data:
   $$\theta\leftarrow\theta-\eta\,\nabla_\theta\mathcal L_{\text{task}}(\theta),\qquad \theta_0=\theta_{\text{pretrained}}$$
   - $\theta$ = the model's weights; $\theta_0$ = where training **starts** (the pretrained weights — the whole point);
   - $\eta$ = the [[learning-rate]] (step size);
   - $\nabla_\theta\mathcal L_{\text{task}}$ = the [[gradient]] of the task loss, i.e. "which direction makes the loss go up";
   - the minus sign = step *downhill*.
   (Plain [[gradient-descent]] form for intuition; the book uses AdamW.)

### Classification fine-tuning in the book (Ch. 6, GPT-2 small 124M)
```python
for param in model.parameters():          # 1. freeze everything
    param.requires_grad = False
torch.manual_seed(123)
num_classes = 2                           # 2. new head: 768 -> 2 instead of 768 -> 50,257
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
for param in model.trf_blocks[-1].parameters():   # 3. unfreeze last block
    param.requires_grad = True
for param in model.final_norm.parameters():       #    and final LayerNorm
    param.requires_grad = True
```
(book pp. 186–187). Why only the top? Lower layers capture basic language structure
useful across tasks; the last layers are more task-specific, so tuning them is often
enough and cheaper (sidebar, book p. 185) — see [[layer-freezing]].
The prediction is read from the **last token's** logits, e.g.
`tensor([[-3.5983, 3.9902]])` (book p. 188) — see [[last-token-pooling]].

Loss: still [[cross-entropy-loss]], but over $K=2$ classes:
$$\mathcal L=-\frac1N\sum_{i=1}^{N}\log\frac{e^{z^{(i)}_{y_i}}}{\sum_{k=1}^{K}e^{z^{(i)}_k}}$$
- $N$ = number of examples; $y_i$ = the true class of example $i$ (0 = not spam, 1 = spam);
- $z^{(i)}_k$ = the logit for class $k$ at the last token of example $i$;
- the fraction = the [[softmax]] probability of the correct class.

Settings: AdamW, `lr=5e-5`, `weight_decay=0.1`, **5 epochs**, about **5.65 minutes**
(book p. 197). Accuracy jumps from **46.25% train / 45.00% val / 48.75% test** (new random head,
coin-flip level, book p. 193; same numbers in appendix E, p. 328) to **97.21% train / 97.32% val / 95.67% test** (book p. 200).

### Instruction fine-tuning in the book (Ch. 7, GPT-2 medium 355M)
- 124M is too small ("capacity") for good instruction following, so the book switches
  to **355M** (book pp. 226–227) — see [[model-capacity]].
- Objective **does not change** from pretraining: next-token cross-entropy on the
  formatted (instruction + response) text, using the Ch. 5 `train_model_simple`
  (book pp. 229–231).
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2
```
- Loss before: **3.826 train / 3.762 val** (book p. 230) → after two epochs
  **0.300 / 0.657**; run time 0.87 min on an A100, 15.78 min on an M3 MacBook Air CPU
  (book pp. 231–232).
- Before fine-tuning, asked to convert "The chef cooks the meal every day." to passive
  voice, the base model just echoes the input; after, it answers *"The meal is cooked
  every day by the chef."* (book pp. 228–232).

### Parameter-efficient fine-tuning (appendix E)
[[low-rank-adaptation]] (LoRA) freezes the pretrained $W$ and learns a low-rank update:
$$W_{\text{updated}}=W+AB$$
- $W$ = the frozen pretrained weight matrix ($d\times d$);
- $A$ ($d\times r$) and $B$ ($r\times d$) = two small trainable matrices; their product $AB$ stands in for the full update $\Delta W$;
- $r$ = the rank, a small [[hyperparameter]].
With `rank=16, alpha=16`, trainable parameters drop from **124,441,346** to
**2,666,528** (book p. 332), and the spam classifier still reaches 100.00% / 96.64% / 98.00%
train/val/test accuracy (book p. 336). See [[parameter-efficient-fine-tuning]].

## 🔴 Theory (Expert)
- **Transfer learning as initialisation.** Fine-tuning is the same optimisation
  problem (ERM on a new dataset), started at $\theta_0=\theta_{\text{pre}}$ instead of
  a random point. With a small dataset, the pretrained start acts as an implicit
  prior / regulariser: the solution stays near $\theta_{\text{pre}}$, inheriting its
  representations ([[transfer-learning]], [[regularization]]).
- **Generative vs. discriminative fine-tuning.** Instruction tuning keeps the
  generative factorisation $\prod_t p_\theta(x_t\mid x_{<t})$ and simply changes the
  data distribution (plus [[instruction-masking]] / [[ignore-index-loss-masking]] of
  padding). Classification tuning swaps the output space from $\mathbb R^{50257}$ to
  $\mathbb R^{2}$, turning the LM into a discriminative model $p(y\mid\mathbf x)$ — hence
  its closed label set (book p. 171).
- **Which tokens / which layers.** With causal attention only the last token has seen
  the whole input, so it's the natural summary for classification (§6.5). Appendix B
  cites experiments where tuning the last block plus head beats head-only, and Li et
  al. (2023) on removing the causal mask during classification tuning (book p. 296).
- **Low-rank hypothesis.** LoRA restricts $\Delta W$ to rank $\le r$, i.e. to a
  low-dimensional subspace capturing the "most influential directions" of the update
  (book p. 322). By the distributive law $x(W+AB)=xW+xAB$ (book pp. 323–324), the
  update stays a separate module — one frozen base, many cheap adapters. Link:
  [[low-rank-approximation]], [[basis-and-rank]].
- **What fine-tuning changes.** Gekhman et al. (2024), cited in appendix B (book
  pp. 297–298): LLMs learn factual knowledge mainly in pretraining; fine-tuning mostly
  improves how they *use* it, and forcing new facts in fine-tuning is slow and raises
  hallucination. Shi (2024): *not* masking instructions can help with long
  instructions / short outputs (book p. 297).
- **After instruction tuning.** An optional [[preference-fine-tuning]] step (e.g.
  [[direct-preference-optimization]]) aligns outputs with human preferences
  (book p. 247).

## Connections
- Builds on: [[pretraining]], [[foundation-model]], [[pretrained-weights]], [[supervised-learning]], [[gradient-descent]], [[backpropagation]], [[cross-entropy-loss]], [[empirical-risk-minimization]]
- Leads to: [[classification-fine-tuning]], [[instruction-fine-tuning]], [[parameter-efficient-fine-tuning]], [[low-rank-adaptation]], [[preference-fine-tuning]], [[direct-preference-optimization]]
- Techniques: [[layer-freezing]], [[classification-head]], [[last-token-pooling]], [[instruction-dataset]], [[prompt-template]], [[instruction-masking]], [[adamw-optimizer]]
- Related: [[transfer-learning]], [[custom-llm]], [[llm-applications]], [[overfitting]], [[model-capacity]], [[classification-accuracy]], [[llm-as-a-judge]]
- People: [[long-ouyang]], [[edward-j-hu]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 5–7 (definition, two categories); §1.5, book p. 12; §1.7, book pp. 14–16 (Figure 1.9, summary).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book p. 17.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 161, 167 (reusing pretrained weights).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.1, book pp. 169–171; §6.5, book pp. 183–190; §6.7, book pp. 195–200; §6.8, book pp. 202–203 (Ch. 6 summary).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.1, book pp. 204–206; §7.5, book pp. 226–229; §7.6, book pp. 229–233; §7.9, book pp. 247–249.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.1, book pp. 322–324; §E.3–E.4, book pp. 326–336.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 295–298 (Ch. 6/7 references).
