---
title: Three Stages of Building an LLM
type: concept
tags: [llm, training-pipeline, pretraining, fine-tuning, roadmap]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch07-fine-tuning-to-follow-instructions, llm-ch03-coding-attention-mechanisms, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Three Stages of Building an LLM

> **One-liner:** You build an LLM in three stages — make the machine, teach it to read a mountain of text, then coach it for one specific job.

*Also covers:* data preparation & sampling, the LLM development cycle, the LLM training pipeline.

## 🟢 Intuition (Dummy)
Think of training a chef.

1. **Build the kitchen (stage 1).** Cut the ingredients into pieces the kitchen can handle
   (turn text into numbers), install the clever "which ingredients go together" machine
   (the attention mechanism), and assemble the whole kitchen (the GPT architecture).
2. **General cooking school (stage 2).** The chef cooks thousands of random recipes from the
   internet. Nobody grades them by hand; the recipe itself says what comes next. After this the
   chef can cook almost anything a bit. That is a **foundation model**. School is very expensive,
   so you can also hire a chef who already graduated (load pretrained weights).
3. **Specialize (stage 3).** Either teach the chef to sort dishes into "good / spoiled"
   (a classifier) or to take orders from customers politely (a personal assistant).

The whole book follows this plan, one chapter per box.

## 🟡 Mechanics (Practitioner)

### Figure 1.9 — the map of the book (book p. 14)
Caption: "The three main stages of coding an LLM are implementing the LLM architecture and data
preparation process (stage 1), pretraining an LLM to create a foundation model (stage 2), and
fine-tuning the foundation model to become a personal assistant or text classifier (stage 3)."

| Stage | Box in the figure | Steps | Book chapter (from the TOC / Fig. 7.21) |
|---|---|---|---|
| **1** | Building an LLM | 1) Data preparation & sampling | Ch. 2 |
| | | 2) [[attention-mechanism|Attention mechanism]] | Ch. 3 |
| | | 3) LLM architecture | Ch. 4 |
| (arrow 1 → 2) | | 4) [[pretraining|Pretraining]] | Ch. 5 |
| **2** | Foundation model | 5) [[llm-training-loop|Training loop]], 6) [[llm-evaluation|Model evaluation]], 7) Load [[pretrained-weights]] | Ch. 5 |
| **3** | Classifier | 8) Fine-tuning on a *dataset with class labels* | Ch. 6 |
| | Personal assistant | 9) Fine-tuning on an *instruction dataset* | Ch. 7 |

The same figure comes back as Fig. 2.1 (book p. 18), Fig. 4.1 (book p. 93) and, fully
annotated, as Fig. 7.21 (book p. 248): "In chapter 4, we implemented a GPT-like LLM
architecture", "In chapter 5, we pretrained an LLM", "we also loaded pretrained model weights",
then ch. 6 (classify texts) and ch. 7 (follow instructions).

### Stage 1 — data preparation and sampling
"Implements the data sampling and understand the basic mechanism" (figure annotation, book
p. 14). Ch. 2 turns text into model input in three steps (book p. 20): split text into words →
convert to [[token|tokens]] / [[token-id|token IDs]] → turn tokens into embedding vectors. On the
way it uses [[byte-pair-encoding]] and [[sliding-window-sampling]] to make input–target pairs
(book pp. 17–18). The rule behind every training pair (standard notation):

$$\text{input}=(x_1,\dots,x_{t}),\qquad \text{target}=x_{t+1}$$

- $x_1,\dots,x_t$ = the tokens seen so far (the context).
- $x_{t+1}$ = the next token, which acts as the label. So the labels come "for free" from the
  text: this is [[self-supervised-learning]].

### Stage 2 — pretraining a foundation model
"Pretrains the LLM on unlabeled data to obtain a foundation model for further fine-tuning"
(book p. 14). The [[training-loop]] minimizes the next-token [[cross-entropy-loss]] (ch. 5):

$$\mathcal{L}(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(x_t\mid x_1,\dots,x_{t-1})$$

- $\mathcal{L}(\theta)$ = the loss (the "badness score") for parameters $\theta$.
- $-\frac1T\sum_{t=1}^T$ = minus the average over all $T$ positions.
- $\log P_\theta(x_t\mid\cdot)$ = the [[log-probability]] the model assigns to the correct next token.
  Making the loss small = making the true next tokens likely = [[maximum-likelihood-estimation]].

The book stresses cost: pretraining from scratch demands "thousands to millions of dollars in
computing costs", so stage 2 trains on a small dataset for teaching, and also shows how to load
openly available weights (book p. 15). For scale: GPT-3 pretraining is estimated at **$4.6 million**
in cloud credits (§1.5, book p. 11).

### Stage 3 — fine-tuning
Adapt the foundation model "to follow instructions such as answering queries or classifying
texts—the most common tasks in many real-world applications and research" (book p. 15).
- **Step 8, [[classification-fine-tuning]]:** labeled classes → a classifier (ch. 6). The loss is
  cross-entropy over the class labels, not over the whole vocabulary.
- **Step 9, [[instruction-fine-tuning]]:** instruction–response pairs → a personal assistant (ch. 7).
  The ch. 7 summary lists the recipe: format with a [[prompt-template]], split into
  [[train-validation-test-split|train/validation/test]], a custom [[collate-function]] that pads,
  builds target IDs and masks padding; start from **GPT-2 medium, 355 million parameters**; train
  with "a training loop similar to pretraining"; score test answers with another LLM
  ([[llm-as-a-judge]], an 8-billion-parameter Llama via Ollama) (book p. 249).

### Optional step after stage 3
§7.9.1 names **[[preference-fine-tuning]]** as an optional step after instruction fine-tuning, to
align the model with user preferences; the book's repo has a DPO notebook
(`04_preference-tuning-with-dpo`) (book p. 247). See [[direct-preference-optimization]].

### After the roadmap (§7.9.2–7.9.3, book pp. 248–249)
- The repository README has a "Bonus Material" section beyond the nine steps (book p. 248).
- To stay current: recent arXiv papers (cs.LG), X and Reddit (r/LocalLLaMA), and the author's
  blog (book p. 248).
- For real-world (non-educational) work the author recommends tools such as **Axolotl** and
  **LitGPT** (book p. 249).

## 🔴 Theory (Expert)
- **Two optimizations, one model.** Stage 2 and stage 3 are both
  [[empirical-risk-minimization]], on different datasets and losses:
  $\theta_{\text{pre}}=\arg\min_\theta \hat R_{\text{web}}(\theta)$, then
  $\theta_{\text{ft}}\approx\arg\min_\theta \hat R_{\text{task}}(\theta)$ found by
  [[gradient-descent]] *starting from* $\theta_{\text{pre}}$. The initialization is the whole
  point: it is [[transfer-learning]]. The summary says the foundation model "can be fine-tuned
  more efficiently" for downstream tasks (book p. 16).
- **Why the split pays off.** Pretraining needs "large datasets consisting of billions of
  words" (book p. 15) but no labels; fine-tuning needs labels but only a "smaller, labeled target dataset"
  (book p. 15). Labels are the scarce resource, so they are spent only where needed.
- **Specialization vs. generality.** "LLMs fine-tuned on custom datasets can outperform general
  LLMs on specific tasks" (book p. 16) — a bias/variance trade: fine-tuning narrows the
  distribution the model is good at. Overly long fine-tuning risks forgetting general skills
  (external context: catastrophic forgetting), which the book manages with
  [[layer-freezing]] (ch. 6) and, in Appendix E, [[low-rank-adaptation]].
- **Where the book's map stops.** Real pipelines (external context) add preference alignment
  (RLHF/DPO), safety tuning and evaluation suites; the book's Fig. 7.21 ends at step 9 and lists
  preference tuning as "what's next" (book pp. 247–248).

### From llm-ch03-coding-attention-mechanisms
Figure 3.1 (book p. 51) places Ch. 3 at **stage 1, step 2: "Attention mechanism"**, between step 1 (data preparation & sampling, Ch. 2) and step 3 (LLM architecture, Ch. 4). Figure 3.2 (book p. 51) splits this step into four variants that build on each other: (1) simplified self-attention, (2) self-attention with trainable weights, (3) causal attention, (4) multi-head attention. See [[self-attention]], [[scaled-dot-product-attention]], [[causal-attention]], [[multi-head-attention]].

## Connections
- Builds on: [[large-language-model]], [[decoder-only-transformer]], [[next-token-prediction]], [[loss-function]], [[gradient-descent]], [[empirical-risk-minimization]]
- Stage 1: [[tokenization]], [[byte-pair-encoding]], [[sliding-window-sampling]], [[token-embedding]], [[positional-embedding]], [[attention-mechanism]], [[gpt-architecture]]
- Stage 2: [[pretraining]], [[pretraining-datasets]], [[llm-training-loop]], [[llm-evaluation]], [[pretrained-weights]], [[foundation-model]]
- Stage 3: [[fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[preference-fine-tuning]]
- People: [[sebastian-raschka]], [[alec-radford]] (GPT-2 weights used in stage 2–3)

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** Ch. 5 is roadmap stage 2, steps 4–7. Fig. 5.1 (book p. 129) highlights step 4 (pretraining), step 5 (training loop), step 6 (model evaluation) and step 7 (loading pretrained weights). Fig. 5.2 (book p. 130) splits the chapter into its own plan: text generation, text evaluation, training and validation losses, the training function, decoding strategies, saving and loading weights, and loading OpenAI's weights. Figs. 5.5, 5.8, 5.10, 5.13 and 5.16 repeat this plan with the current step highlighted. Note that Fig. 5.1 labels step 7 "Load pretrained weights", while its caption says *saving and loading* model weights.

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7 + Ch. 1 Summary, book pp. 14–16 (Fig. 1.9, costs, two-step training).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 2 intro + §2.1, book pp. 17–20 (Fig. 2.1, data pipeline preview).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book p. 11 ($4.6M GPT-3 estimate); §4.1, book p. 93 (Fig. 4.1).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.9 + Ch. 7 Summary, book pp. 247–249 (Fig. 7.21, preference fine-tuning, GPT-2 medium 355M, LLM-as-judge).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 3 intro, book pp. 50–52 (Fig. 3.1 roadmap step 2; Fig. 3.2 four attention variants).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 6 intro, book pp. 169–170 (Fig. 6.1, stage 3 classification fine-tuning) — see [[llm-ch06-fine-tuning-for-classification]].
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 5 intro, book pp. 128–130 (Figs. 5.1–5.2); roadmap repeats in Figs. 5.5, 5.8, 5.10, 5.13, 5.16 (book pp. 138–159).
