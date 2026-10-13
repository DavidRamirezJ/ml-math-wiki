---
title: Foundation (Base) Model
type: concept
tags: [llm, pretraining, transfer-learning, gpt]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Foundation (Base) Model

> **One-liner:** A foundation model (also called a base model) is the LLM you get right after pretraining — good at continuing text, not yet specialised, and the starting point for every fine-tuned model.

## 🟢 Intuition (Dummy)
Think of a house. Pretraining pours the **foundation**: it's big, expensive and
everything else sits on it. On that one foundation you can build very different
houses — a spam filter, a chat assistant, a medical Q&A bot. That's literally the
book's picture: in Figure 1.9 everything in stage 3 grows out of a single box labelled
**"Foundation model"** (book p. 14).

What can a foundation model already do? It's a strong **autocomplete**. Give it half
a sentence and it finishes it. It also has a little **few-shot** ability — show it a
couple of examples of a task and it can sometimes copy the pattern (book p. 7). What it
*can't* reliably do is follow orders like "Convert this text into passive voice"
(book p. 205). For that, you fine-tune it.

The book uses three names for the same thing: *pretrained LLM*, *base model*, and
*foundation model* (book p. 7). They are called that because, being pretrained,
they're "very versatile for further fine-tuning on downstream tasks" (book p. 11).

## 🟡 Mechanics (Practitioner)

### Where it sits in the pipeline
$$\underbrace{\text{random weights}}_{\text{Stage 1: build}}\;\xrightarrow{\;\text{pretraining}\;}\;\underbrace{\theta_{\text{base}}}_{\text{Stage 2: foundation model}}\;\xrightarrow{\;\text{fine-tuning}\;}\;\underbrace{\theta_{\text{task}}}_{\text{Stage 3: classifier / assistant}}$$
Reading it:
- $\theta$ = the model's full set of [[model-parameters]];
- $\theta_{\text{base}}$ = the parameters after [[pretraining]] — *this vector of numbers is the foundation model*;
- $\theta_{\text{task}}$ = the parameters after [[fine-tuning]] for a specific job;
- the arrows = training runs (gradient descent on different datasets).

In Figure 1.9 (book p. 14), stage 2 is steps 4–7: pretraining, training loop, model
evaluation, load pretrained weights; stage 3 branches into step 8 (classifier) and
step 9 (personal assistant). Figures 6.1, 7.1 and 7.21 redraw the same map (book
pp. 170, 205, 248).

### What a base model can do (book p. 7, Figure 1.3)
- **[[text-completion]]** — finish a half-written sentence.
- **[[few-shot-learning]]** (limited) — learn a new task from only a few examples in the prompt.

What it is trained on: raw, unlabeled text with [[next-token-prediction]] (book p. 6).

### Real examples in the book
| Base model | Facts from the book |
|---|---|
| [[gpt-3]] | base model of the first ChatGPT; 96 layers, **175 billion** parameters; pretrained on 300 B tokens for an estimated \$4.6 M (book pp. 7, 11, 13) |
| [[gpt-2]] 124M | `n_vocab 50257, n_ctx 1024, n_embd 768, n_head 12, n_layer 12` (book p. 162); used as the base for spam classification (Ch. 6) |
| GPT-2 355M | base for instruction tuning (Ch. 7, book p. 226) |
| Llama 3 8B base | judged score **58.51** vs **82.6** for the instruction-tuned Llama 3 8B (book p. 247) |

### Loading a foundation model instead of building one
Because pretraining costs "tens to hundreds of thousands of dollars" even for GPT-2
scale (book p. 161), the book downloads OpenAI's weights:
```python
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
gpt = GPTModel(NEW_CONFIG)
load_weights_into_gpt(gpt, params)
```
(book pp. 161–166). The quick sanity check is to generate text — a correctly loaded
base model writes *"Every effort moves you toward finding an ideal new way to practice
something!…"* (book p. 167).

**The base model is a completion engine, not an assistant.** In Ch. 7, given an
Alpaca-style prompt asking to convert *"The chef cooks the meal every day."* to
passive voice, the base GPT-2 355M only repeats the input sentence and part of the
instruction (book pp. 228–229). Because base models are text-completion models,
`generate` returns input + output joined together (book p. 228).

## 🔴 Theory (Expert)
- **Formal view.** A foundation model is a parameter vector $\theta_{\text{base}}
  \approx\arg\min_\theta \mathbb E_{x\sim\mathcal D_{\text{pre}}}\big[-\log p_\theta(x)\big]$
  (approximate because optimisation is stopped early, usually after about one epoch).
  Fine-tuning solves a second problem $\min_\theta\mathcal L_{\text{task}}(\theta)$
  starting at $\theta_{\text{base}}$. The base model is thus both a *density model* of
  web-scale text and an *initialisation* — see [[transfer-learning]].
- **Why one base serves many tasks.** Next-token modelling on diverse data forces
  representations of syntax, semantics and some world knowledge (book p. 11); tasks
  like translation then appear as [[emergent-behavior]] (book p. 14). Fine-tuning
  re-uses these internal representations — the reason the book's lower layers can be
  frozen in Ch. 6 (book p. 185).
- **Base vs. instruct gap.** The 58.51 → 82.6 jump for Llama 3 8B (book p. 247)
  quantifies what instruction tuning adds on top of identical pretraining. Appendix B
  (Gekhman et al. 2024, book pp. 297–298) argues the *knowledge* lives mostly in the
  base model; tuning teaches its use.
- **Terminology.** "Foundation model" as a term was popularised by the Stanford CRFM
  report (Bommasani et al., 2021) — outside knowledge, not cited by the book. The
  book also lists Meta's "Llama 2: Open Foundation and Fine-Tuned Chat Models" in
  appendix B (book p. 290).
- **Open weights and custom models.** Many base models are released openly and can be
  used as general tools or fine-tuned on custom data (book p. 12); domain-specific
  models built this way can beat general LLMs on their domain (book p. 16) — see
  [[custom-llm]].

## Connections
- Builds on: [[pretraining]], [[next-token-prediction]], [[self-supervised-learning]], [[large-language-model]]
- Leads to: [[fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[custom-llm]], [[low-rank-adaptation]]
- Capabilities: [[text-completion]], [[few-shot-learning]], [[zero-shot-learning]], [[emergent-behavior]]
- Examples: [[gpt-3]], [[gpt-2]], [[gpt-architecture]]
- Related: [[pretrained-weights]], [[transfer-learning]], [[llm-building-roadmap]], [[baseline-evaluation]], [[model-parameters]]
- People: [[alec-radford]], [[tom-brown]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 6–7 (base/foundation model, text completion, few-shot, GPT-3); §1.5, book pp. 11–12; §1.6, book p. 13; §1.7, book pp. 14–16 (Figure 1.9, summary).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 160–167 (loading GPT-2 weights).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.1, book p. 170 (Figure 6.1); §6.5, book p. 185; §6.8, book p. 203 (load pretrained model as base model).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.1, book p. 205; §7.5, book pp. 226–229; §7.9, book pp. 247–249 (Figure 7.21, Llama 3 scores).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 290, 297–298.
