---
title: Edward J. Hu
type: person
tags: [deep-learning, llm, fine-tuning, parameter-efficient-fine-tuning]
sources: [llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Edward J. Hu (contemporary)

> **Why they matter to ML:** first author of LoRA, the low-rank trick that made fine-tuning huge language models cheap enough for almost anyone.

## Life (brief)
Edward J. Hu is a machine-learning researcher. When LoRA was published (2021) he was at Microsoft,
working with co-authors Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu
Wang and Weizhu Chen. He later did doctoral research at Mila in Montréal with
[[yoshua-bengio]]. (All biographical details are general knowledge. The book cites him only as
"Hu et al.", book p. 323.)

## Key contributions
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al. 2021, arXiv 2106.09685). Freeze
  the pretrained weight $W$ and learn only a low-rank update, $\Delta W \approx AB$ with inner
  dimension $r \ll d$. Read: "the change to the weights is approximated by multiplying a tall thin
  matrix $A$ by a short wide matrix $B$; $r$ is how thin they are, much smaller than the layer
  width $d$". The book builds its whole Appendix E on this paper and fine-tunes GPT-2 small with
  2,666,528 trainable parameters instead of 124,441,346 (book pp. 323, 332). From the paper (general
  knowledge): on GPT-3 175B, LoRA cut trainable parameters by about 10,000× and GPU memory by about 3×,
  with no extra inference latency once $AB$ is merged into $W$.
  → [[low-rank-adaptation]], [[lora-rank-and-alpha]], [[parameter-efficient-fine-tuning]], [[low-rank-approximation]]
- **Hyperparameter transfer across model width** (general knowledge). Co-author with Greg Yang of the
  "Tensor Programs" work on infinite-width feature learning and μP / μTransfer (2021–2022). The idea is
  to tune [[hyperparameter|hyperparameters]] on a small model and transfer them to a large one.
  → [[weight-initialization]], [[learning-rate]]

## Connections
- Concepts: [[low-rank-adaptation]], [[lora-rank-and-alpha]], [[parameter-efficient-fine-tuning]], [[low-rank-approximation]], [[fine-tuning]]
- People: [[yoshua-bengio]] (later doctoral advisor), [[sebastian-raschka]] (implements LoRA from scratch in Appendix E)
- Appears in sources: [[llm-appE-parameter-efficient-fine-tuning-with-lora]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.1, book p. 323: "LoRA, proposed by Hu et al. (https://arxiv.org/abs/2106.09685)".
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book pp. 328–336 (the book's implementation of the method).
- First name, affiliations, co-authors, the paper's GPT-3 results, and the μP work are general knowledge (arXiv records), not from the book.
