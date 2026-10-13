---
title: Transfer Learning
type: concept
tags: [deep-learning, fine-tuning, pretraining, optimization]
difficulty: intermediate
sources: [llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Transfer Learning

> **One-liner:** Transfer learning means reusing what a model already learned on one big job as the starting point for a new, smaller job, instead of starting from scratch.

> Note: Raschka doesn't use the phrase "transfer learning" in §§6.4, 6.5 or 7.5. He talks about
> "initializing a model with pretrained weights" and "fine-tuning". "Transfer learning" is the
> standard ML name for that idea, and this page uses it as a label.

## 🟢 Intuition (Dummy)
If you already play the guitar, learning the ukulele is quick. You don't relearn what a chord is
or how rhythm works, you only adjust to four strings. Your old skill **transfers**.

An LLM that read billions of words during [[pretraining]] already "knows" grammar, word meanings
and a lot about the world. To teach it to spot spam, we don't start with a blank brain. We take
the pretrained GPT-2, keep almost everything, and train only a little bit more on about a thousand
labeled messages (1,045 in the training split, derived from 70 % of 1,494). It goes from coin-flip accuracy to about 96 % in about 5.65 minutes of training
(book pp. 193, 197, 200).

## 🟡 Mechanics (Practitioner)

### Step 1: start from pretrained weights (§6.4, Listing 6.6, book p. 182)
```python
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {"vocab_size": 50257, "context_length": 1024,
               "drop_rate": 0.0, "qkv_bias": True}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])   # emb_dim 768, n_layers 12, n_heads 12
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model = GPTModel(BASE_CONFIG)          # random weights...
load_weights_into_gpt(model, params)   # ...overwritten by OpenAI's GPT-2 weights
model.eval()
```
Sanity check: "Every effort moves you" → "Every effort moves you forward. The first step is to
understand the importance of your work" — coherent text, so the weights loaded fine (book p. 182).

But asking it *"Is the following text 'spam'? Answer with 'yes' or 'no': …"* just makes it echo the
prompt. A pretrained-only model can't follow instructions (book p. 183). So we adapt it.

### Step 2: adapt the model to the new task
- **Classification (§6.5)** — swap the output layer for a 2-class [[classification-head]], freeze
  most layers ([[layer-freezing]]), and train the last transformer block, the final LayerNorm and
  the new head (book pp. 183–187).
- **Instructions (§7.5)** — load GPT-2 **medium (355M)** with the *same* loading code (only the
  `CHOOSE_MODEL` string changes) and fine-tune all of it on instruction–response pairs
  (book pp. 226–227). The 124M model was too small: it lacks the
  [[model-capacity|capacity]] to learn good instruction following (book pp. 226–227).

### The math: same optimizer, different starting point
Training is ordinary [[gradient-descent]]. The only change is where we start:

$$\boldsymbol\theta_0=\boldsymbol\theta_\text{pretrained},\qquad
\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta\,\nabla_{\boldsymbol\theta}\mathcal L_\text{new}(\boldsymbol\theta_t)$$

Reading it:
- $\boldsymbol\theta$ = all the model's weights; the subscript $t$ = the training step number.
- $\boldsymbol\theta_0$ = the starting weights. In transfer learning they are the pretrained ones,
  not random numbers.
- $\eta$ ("eta") = the [[learning-rate]], how big each step is.
- $\nabla_{\boldsymbol\theta}\mathcal L_\text{new}$ = the [[gradient]] of the new task's loss
  (e.g. spam cross-entropy) — the direction that makes the loss go up fastest; the minus sign
  walks the other way.

With freezing, only a subset moves:
$\boldsymbol\theta_\text{train}\leftarrow\boldsymbol\theta_\text{train}-\eta\nabla_{\boldsymbol\theta_\text{train}}\mathcal L$,
while $\boldsymbol\theta_\text{frozen}$ stays fixed (book p. 186).

### Why the lower layers transfer (sidebar, book p. 185)
Lower layers of a language model "generally capture basic language structures and semantics"
that apply to many tasks; the last layers hold more task-specific patterns. So fine-tuning only
the top is often enough, and cheaper.

### What you get (book numbers)
| | 124M classification (Ch. 6) | 355M instruction (Ch. 7) |
|---|---|---|
| Starting point | GPT-2 small weights | GPT-2 medium weights (1.42 GB download) |
| Before fine-tuning | ~46–49 % accuracy (≈ random) | repeats the prompt, no passive voice |
| After | 97.21 / 97.32 / 95.67 % train/val/test | "The meal is cooked every day by the chef." |

(book pp. 193, 200, 227–229, 232)

## 🔴 Theory (Expert)
- **Initialization as an implicit prior.** Starting at $\boldsymbol\theta_\text{pretrained}$ and
  taking few, small steps keeps the solution near it — a form of implicit
  [[regularization]] similar to an $L^2$ penalty $\lambda\lVert\boldsymbol\theta-\boldsymbol\theta_\text{pretrained}\rVert^2$
  (approximately so for linear least-squares models trained with early-stopped gradient descent;
  this connection is standard ML theory, not from the book). This is why 1,045 training
  messages are enough to tune a 124M-parameter model without obvious [[overfitting]]
  (book p. 199).
- **Feature reuse.** Freezing everything but a linear head reduces to fitting a
  [[logistic-regression]] on fixed features $\mathbf h(\mathbf x)$ — the "linear probe". Unfreezing
  more layers lets the features themselves bend toward the task; the author found that adding the
  last transformer block "noticeably" helps (book p. 186; appendix B, p. 296).
- **Knowledge comes from pretraining.** Appendix B cites Gekhman et al. (2024): LLMs acquire
  factual knowledge mostly in pretraining; fine-tuning mainly teaches how to *use* it, and new
  facts introduced in fine-tuning are learned slowly and raise hallucination (book pp. 297–298).
- **Parameter-efficient variants.** Instead of moving $\boldsymbol\theta$, learn a small update
  $\Delta W=AB$ of low rank ([[low-rank-adaptation]], appendix E) — transfer learning with far
  fewer trainable numbers. See also [[parameter-efficient-fine-tuning]].
- **Domain gap.** Transfer works best when source and target data share structure (both English
  text here). Appendix B (p. 289) contrasts two routes to a domain model: BloombergGPT, a GPT
  pretrained *from scratch* on finance plus general text, which beat ChatGPT on financial tasks;
  and Google's medical work (Singhal et al., 2023, "Towards Expert-Level Medical Question
  Answering with Large Language Models"), which *adapts and fine-tunes an existing* LLM. The
  book also lists continued-pretraining strategies (Ibrahim et al., 2024, p. 294).

## Connections
- Builds on: [[pretraining]], [[pretrained-weights]], [[gradient-descent]], [[stochastic-gradient-descent]], [[gpt-model-configuration]]
- Leads to: [[fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[layer-freezing]], [[low-rank-adaptation]]
- Used by: [[text-classification]], [[classification-head]], [[foundation-model]]
- Related: [[model-capacity]], [[regularization]], [[baseline-evaluation]], [[gpt-2]]
- People: [[sebastian-raschka]], [[alec-radford]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.4, book pp. 181–183 (Listing 6.6, model configs, sanity check, failed spam prompt)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 185–187 (sidebar on lower vs last layers, freezing)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6–6.7, book pp. 193, 197–200 (accuracies before/after, no overfitting)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5, book pp. 226–229 (Listing 7.7, 355M choice, capacity, baseline output)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6, book p. 232 (passive-voice output after fine-tuning)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 289, 294, 296–298
