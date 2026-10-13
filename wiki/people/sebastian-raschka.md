---
title: Sebastian Raschka
type: person
tags: [llm, pytorch, education, fine-tuning, author]
sources: [llm-ch01-understanding-large-language-models, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Sebastian Raschka (active 2010s–present)

> **Why they matter to ML:** he wrote *Build a Large Language Model (From Scratch)*, the book that walks you through coding a GPT-style [[large-language-model]] in [[pytorch]] one piece at a time — from [[tokenization]] all the way to [[instruction-fine-tuning]].

## Life (brief)
Sebastian Raschka is the author and the "I" narrator of the book. The book gives his public homes: the code repository `rasbt/LLMs-from-scratch` on GitHub, and his blogs at magazine.sebastianraschka.com and sebastianraschka.com/blog (book pp. 248–249). He also says he is actively involved in developing **LitGPT** (Lightning AI), an open library for using and fine-tuning bigger LLMs (book p. 249).

General knowledge, not from the book: he has a PhD from Michigan State University, was an assistant professor of statistics at the University of Wisconsin–Madison, and then joined Lightning AI to work on LLM research and education. He is also known for the books *Python Machine Learning* and *Machine Learning with PyTorch and Scikit-Learn* (the second one, co-written with Yuxi (Hayden) Liu and Vahid Mirjalili, is cited in appendix B, book p. 298) and for the `mlxtend` Python library. Appendix B also cites his book *Machine Learning Q and AI* (2023), whose first chapter compares embedding spaces with latent spaces (book p. 290).

## Key contributions
His contributions here are mostly **teaching** ones: he turns big ideas into small, runnable code. The ones that show up in the book:

- **The "build it yourself" roadmap.** His main claim: "building an LLM from scratch is the most effective way to gain a deep understanding of how LLMs work" (book p. 248). The book follows three stages — build the architecture, [[pretraining]], then [[fine-tuning]]. → [[llm-building-roadmap]], [[gpt-architecture]], [[llm-training-loop]]
- **The companion repository** `rasbt/LLMs-from-scratch`. It hosts all code, plus the small datasets the book downloads: the short story `the-verdict.txt` for pretraining (book pp. 21–22, 314) and `instruction-data.json` for instruction tuning (book p. 207). It is also the fallback if downloading OpenAI's [[pretrained-weights]] fails (book p. 162). → [[gpt-2]], [[instruction-dataset]]
- **Fine-tuning experiments.** In chapter 6 he says that training only the new output layer would be enough, but "as I found in experiments, fine-tuning additional layers can noticeably improve the predictive performance of the model" (book p. 186) — so the book also unfreezes the last transformer block and the final LayerNorm. Appendix B adds his articles "Using and Finetuning Pretrained Transformers" and "Finetuning Large Language Models" on the types of fine-tuning (book p. 295), and his article "Finetuning Large Language Models" (a second link) reports that tuning the last [[transformer-block]] plus the output layer "improves the predictive performance substantially" (book p. 296). → [[classification-fine-tuning]], [[layer-freezing]], [[classification-head]]
- **One output node vs. two.** For yes/no (binary) classification you *could* use a single output node, but then you must change the loss. His article "Losses Learned—Optimizing Negative Log-Likelihood and Cross-Entropy in PyTorch" explains this (book pp. 183, 295). → [[cross-entropy-loss]], [[binary-cross-entropy]], [[sigmoid-function]]
  - The two losses side by side, so you can read them. Two-node version (softmax + cross-entropy) for one example with true class $y$:
    $$L = -\log \frac{e^{z_y}}{e^{z_0} + e^{z_1}}$$
    Symbol by symbol: $z_0, z_1$ = the two [[logits]] (raw scores) the model outputs; $e^{z}$ = "e to the power $z$", which makes every score positive; the fraction is the [[softmax]] probability of the correct class $y$; $\log$ = natural log; the minus sign turns "high probability" into "low loss".
  - One-node version (sigmoid + binary cross-entropy) with label $y \in \{0,1\}$ and single logit $z$:
    $$L = -\big[\,y \log \sigma(z) + (1-y)\log(1-\sigma(z))\,\big], \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$
    Symbol by symbol: $\sigma(z)$ = the [[sigmoid-function]], which squashes $z$ into a probability between 0 and 1; when $y=1$ only the first term is alive, when $y=0$ only the second term is alive. (Writing out this pair is our addition; the book only points to the article.)
- **PyTorch teaching material.** Appendix A points to his "Scientific Computing in Python: Introduction to NumPy and Matplotlib" (book p. 258) and to "Accelerating PyTorch Model Training: Using Mixed-Precision and Fully Sharded Data Parallelism" (book p. 288). Appendix B lists his lecture videos on [[logistic-regression]] loss, [[one-hot-encoding]] with cross-entropy, [[tensor|tensors]], a free calculus refresher, and an article on [[gradient-accumulation]] (book pp. 293–298). → [[pytorch]], [[distributed-data-parallel]], [[floating-point-precision]]
- **Beyond the book.** For [[preference-fine-tuning]] he points to his articles "LLM Training: RLHF and Its Alternatives" and "Tips for LLM Pretraining and Evaluating Reward Models" (book p. 298). For [[model-capacity|model evaluation]] in general, his 2018 arXiv paper "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning" (book p. 298). → [[direct-preference-optimization]], [[cross-validation]], [[llm-evaluation]]

## Connections
- Concepts: [[large-language-model]], [[gpt-architecture]], [[llm-building-roadmap]], [[pretraining]], [[fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[pytorch]], [[cross-entropy-loss]], [[binary-cross-entropy]], [[layer-freezing]], [[gradient-accumulation]]
- People: [[alec-radford]] (GPT-2, whose weights the book loads), [[andrej-karpathy]] (another from-scratch GPT teacher)
- Appears in sources: [[llm-ch05-pretraining-on-unlabeled-data]], [[llm-ch06-fine-tuning-for-classification]], [[llm-ch07-fine-tuning-to-follow-instructions]], [[llm-appA-introduction-to-pytorch]], [[llm-appD-training-loop-bells-and-whistles]]

> **From [[llm-ch01-understanding-large-language-models]]:** Ch. 1 is where he first speaks as the "I" narrator. He gives his reason to build an LLM yourself: "Coding an LLM from the ground up is an excellent exercise to understand its mechanics and limitations" (book p. 5). He states that "For simplicity, I use the term 'LLM' to refer to transformer-based LLMs similar to GPT" (book p. 10). He also lays out the 3-stage, 9-step plan of the book (Fig. 1.9, book p. 14; see [[llm-building-roadmap]]).

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book p. 5 (why build your own LLM); §1.4 sidebar, book p. 10 (his use of "LLM"); §1.7, book pp. 14–15 (Fig. 1.9 roadmap).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 161–162 (repo `rasbt/LLMs-from-scratch`, fallback for weight download).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 183, 186 (single output node article; tuning more layers helps).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.2, book p. 207 (instruction dataset hosted in his repo); §7.6, book p. 230 (his own runs on an A100 GPU); §7.9, book pp. 248–249 (thesis, blogs, LitGPT).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1, book pp. 251–257 (narrator; recommends PyTorch 2.4.0; repo for setup); §A.2, book p. 258 (NumPy article); §A.9, book p. 288 (mixed-precision / FSDP article).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1, book p. 314 (repo hosts `the-verdict.txt`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 290–298 (his books, videos, and articles listed per chapter: *Machine Learning Q and AI* p. 290; L8.2 logistic-regression loss video p. 293; one-hot/cross-entropy video and code p. 294; fine-tuning articles pp. 295–296; RLHF / reward-model articles, *Machine Learning with PyTorch and Scikit-Learn*, tensors video, model-evaluation paper, calculus chapter, gradient-accumulation article p. 298).
- Biography (PhD, Wisconsin–Madison, Lightning AI, other books, `mlxtend`) is general knowledge, not from the book.
