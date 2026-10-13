---
title: Custom / Domain-Specific LLMs
type: concept
tags: [llm, domain-adaptation, fine-tuning, pretraining, deployment]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Custom / Domain-Specific LLMs

> **One-liner:** A custom LLM is a language model built or tuned for one field or task (finance, medicine, your company), and it can beat big general-purpose models at that one thing.

## 🟢 Intuition (Dummy)
A general-purpose LLM like ChatGPT is a brilliant generalist: it knows a bit about everything.
A **custom LLM** is a specialist, like a doctor who trained only in cardiology. Ask both about
a heart scan and the specialist usually wins, even if the generalist is "smarter" overall.

The book gives two famous examples: **BloombergGPT**, made for finance, and LLMs tailored to
**medical question answering** (book p. 5). Both beat general models on their home turf.

There are also practical reasons to build your own, beyond accuracy (book pp. 5–6):
- **Privacy:** you don't have to send confidential data to a third-party provider such as OpenAI.
- **On-device:** a smaller model can run right on a laptop or phone. Apple is exploring this.
- **Speed and cost:** running locally means lower latency and lower server bills.
- **Control:** you decide when and how the model is updated or changed.

That's a big part of why the book teaches you to build an LLM from scratch. Once you know the
mechanics, you can pretrain or fine-tune open models on *your* data (book p. 5).

## 🟡 Mechanics (Practitioner)

### Two ways to get a custom model
The examples in appendix B show two different routes (book pp. 289, 294):

| Route | What you do | Example |
|---|---|---|
| **Pretrain (from scratch) on domain + general text** | Train a new GPT on a data mix that includes a big domain corpus | **BloombergGPT** (Wu et al. 2023): a GPT pretrained on finance data from scratch, trained on both general and finance corpora. It did better than ChatGPT on financial tasks and stayed good on general benchmarks |
| **Adapt an existing LLM** | Start from pretrained weights and [[fine-tuning|fine-tune]] on domain data | Singhal et al. 2023, "Towards Expert-Level Medical Question Answering with Large Language Models" (Google Research / Google DeepMind): existing LLMs adapted and fine-tuned to outperform general LLMs in a medical context. (The paper's model is Med-PaLM 2; the book does not name it.) |

The second route is the one the book codes, because pretraining is very expensive. GPT-3's
pretraining is estimated at about **\$4.6 million** in cloud credits (book p. 11). The book loads
open GPT-2 weights and fine-tunes them in two ways: [[classification-fine-tuning]] (a spam
filter, Ch. 6) and [[instruction-fine-tuning]] (an assistant, Ch. 7).

### The math behind "specialize" (our formalization)
Training minimizes average loss over the data you train on ([[empirical-risk-minimization]]):
$$\hat\theta = \arg\min_{\theta}\; \frac{1}{n}\sum_{i=1}^{n} \ell\big(f_\theta(x_i),\, y_i\big)$$
Symbol by symbol:
- $\theta$ = the model's weights; $\hat\theta$ ("theta-hat") = the weights we end up with;
- $(x_i, y_i)$ = training example $i$ (for next-token training, $x_i$ is the text so far and $y_i$ the next token);
- $n$ = number of training examples;
- $f_\theta(x_i)$ = the model's prediction; $\ell$ = the per-example [[loss-function]] (here [[cross-entropy-loss]]);
- $\arg\min_\theta$ = "the $\theta$ that makes this average as small as possible."

A **general** model draws $(x_i,y_i)$ from a broad mix of the internet. A **custom** model draws
them, fully or mostly, from the domain you care about. Same formula, different data. The model's
limited capacity then goes into the patterns that matter for your domain.

With **fine-tuning** you also start from a good point instead of random weights:
$$\theta_0 = \theta_{\text{pretrained}},\qquad \theta_{s+1} = \theta_s - \eta\,\nabla_\theta \mathcal{L}_{\text{domain}}(\theta_s)$$
- $\theta_0$ = the starting weights, copied from a pretrained model ([[pretrained-weights]], [[transfer-learning]]);
- $s$ = the step number; $\eta$ = the learning rate (e.g. $5\times10^{-5}$ in Ch. 6–7);
- $\nabla_\theta \mathcal{L}_{\text{domain}}$ = the [[gradient]] of the loss on *domain* data;
- in practice the book uses the [[adamw-optimizer]], not this plain [[gradient-descent]] step.

### Custom models per customer, cheaply: LoRA
If every customer needs their own model, storing a full copy each time is wasteful. With
[[low-rank-adaptation]] the pretrained weights stay frozen, and you save only small extra matrices
per customer or application (§E.1, book p. 324). In the book's spam example that is **2,666,528**
trainable LoRA parameters instead of **124,441,346** (book p. 332), about 47× fewer to store per
customization. See [[parameter-efficient-fine-tuning]].

## 🔴 Theory (Expert)
- **Why a specialist can beat a bigger generalist.** A general model minimizes risk under a broad
  mixture $\sum_d \pi_d\, P_d$ of data sources ([[mixture-model]], [[pretraining-datasets]]). If your
  target domain $P_{\text{target}}$ has a small weight $\pi_{\text{target}}$, the general model
  spends little capacity on it. Training or fine-tuning on $P_{\text{target}}$ moves the
  minimizer toward the risk you actually care about. This is our framing, not the book's. The
  book states the empirical fact: custom LLMs can outperform general ones on specific tasks
  (book pp. 5, 16).
- **Keep general skills.** BloombergGPT trained on *both* general and finance text, and stayed
  good on general benchmarks (appendix B, book pp. 289, 294). Training only on narrow data risks
  losing general ability (catastrophic forgetting, outside the book) and [[overfitting]] the
  small domain set. The book's fine-tuning runs watch the training vs. validation loss gap for
  this reason (§6.7, §7.6).
- **Fine-tuning shapes behavior more than knowledge.** Appendix B cites Gekhman et al. (2024):
  LLMs pick up factual knowledge mostly during pretraining, and fine-tuning on new facts is slow
  and can increase hallucination (book pp. 297–298). So for a truly new domain vocabulary or
  knowledge base, continued pretraining (Ibrahim et al. 2024, appendix B p. 294) or
  [[retrieval-augmented-generation]] are common choices. RAG is outside this book.
- **Deployment math.** On-device feasibility is about memory. Roughly, bytes ≈ parameter count ×
  bytes per parameter ([[parameter-count-and-memory]], [[floating-point-precision]]). That's why
  "smaller custom LLMs" are the ones that fit on laptops and phones (book p. 5). The Ollama judge
  in §7.8 is the same idea in practice: an 8 B model running locally in about 16 GB of RAM
  (book p. 241, [[local-llm-inference-ollama]]).
- **Specialist vs. generalist trade-off.** It is "easier to develop a specialized model than a
  generalist model that works well on various tasks" (book p. 171). A classification fine-tuned
  model is the extreme case: it can only output its trained labels.

## Connections
- Builds on: [[large-language-model]], [[pretraining]], [[fine-tuning]], [[foundation-model]], [[pretrained-weights]], [[transfer-learning]], [[empirical-risk-minimization]], [[loss-function]]
- Methods: [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]], [[parameter-efficient-fine-tuning]]
- Related: [[pretraining-datasets]], [[mixture-model]], [[overfitting]], [[model-capacity]], [[parameter-count-and-memory]], [[local-llm-inference-ollama]], [[retrieval-augmented-generation]], [[llm-applications]], [[gpt-3]]
- People: [[sebastian-raschka]], [[edward-j-hu]] (LoRA)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 5–6 (custom vs. general LLMs; BloombergGPT; medical QA; privacy, on-device, latency/cost, autonomy).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book p. 11 (GPT-3 pretraining cost \$4.6 M).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7 summary, book p. 16 (LLMs fine-tuned on custom datasets can outperform general LLMs).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.1, book p. 171 (specialist vs. generalist).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8, book p. 241 (local 8 B model, 16 GB RAM).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix E §E.1, book p. 324; §E.4, book p. 332 (per-customer LoRA matrices; 2,666,528 vs 124,441,346 parameters).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 289, 294, 297–298 (Wu et al. 2023 BloombergGPT; Singhal et al. 2023 medical QA; Ibrahim et al. 2024; Gekhman et al. 2024).
