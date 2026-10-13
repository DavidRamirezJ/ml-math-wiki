---
title: "LLM Ch.1 — Understanding Large Language Models"
type: source
tags: [llm, transformer, nlp, source]
created: 2026-10-13
updated: 2026-10-13
---

# LLM Ch.1 — Understanding Large Language Models

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 1, **book pp. 1–16** (PDF pages 23–38)
- **Type:** book chapter. *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025)
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13 (each section was checked line by line against the extracted text, and Figures 1.1, 1.4–1.9 were checked against rendered pages)
- *Page rule:* book page N = PDF page N + 22.
- **Final coverage audit:** 2026-10-13. All 7 sections, the sidebars, Table 1.1, Figs. 1.1–1.9, and the summary were checked against the extracted text. All 47 owned pages exist and have the three tiers (or the person template). Small additions: the Fig. 1.4 definition of "embedding representation" in [[vector]], the "consumer hardware" promise (p. 12) in [[pretraining]], and a Ch. 1 note on [[sebastian-raschka]].

This chapter is a map, not a math chapter. It has **no formulas, no code, and no exercises**. It gives you the vocabulary (LLM, transformer, pretraining, fine-tuning, GPT, BERT, zero-/few-shot) and the plan for the rest of the book. Where the wiki pages add math (for example, the next-token probability factorization), they mark it as a wiki addition, not book text.

## Section coverage (all of Ch. 1 accounted for)
None of the sections in Ch. 1 has numbered subsections (there is no 1.x.y). The rows below also list the sidebars and the chapter summary so that nothing is skipped.

| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Chapter intro ("This chapter covers") | 1–2 | ✅ [[large-language-model]], [[natural-language-processing]], [[transformer-architecture]] |
| 1.1 | What is an LLM? | 2–4 | ✅ [[large-language-model]], [[model-parameters]], [[next-token-prediction]], [[generative-ai]], [[artificial-intelligence]], [[machine-learning]], [[deep-learning]], [[feature-extraction]], [[empirical-risk-minimization]], [[loss-function]], [[supervised-learning]] |
| 1.1 · Fig. 1.1 | AI ⊃ ML ⊃ DL ⊃ LLMs, with GenAI | 3 | ✅ [[artificial-intelligence]], [[machine-learning]], [[deep-learning]], [[generative-ai]] |
| 1.2 | Applications of LLMs | 4–5 | ✅ [[llm-applications]], [[machine-translation]], [[text-classification]], [[autoregressive-text-generation]] |
| 1.3 | Stages of building and using LLMs | 5–7 | ✅ [[custom-llm]], [[pretraining]], [[fine-tuning]], [[self-supervised-learning]], [[foundation-model]], [[gpt-3]], [[text-completion]], [[few-shot-learning]], [[instruction-fine-tuning]], [[classification-fine-tuning]], [[pytorch]] |
| 1.4 | Introducing the transformer architecture | 7–10 | ✅ [[transformer-architecture]], [[encoder-decoder-architecture]], [[self-attention]], [[attention-mechanism]], [[bert]], [[masked-language-modeling]], [[gpt-architecture]], [[zero-shot-learning]], [[few-shot-learning]], [[text-completion]], [[vector]] |
| 1.4 · sidebar | Transformers vs. LLMs | 10 | ✅ [[transformer-architecture]], [[large-language-model]], [[recurrent-neural-network]] |
| 1.5 | Utilizing large datasets | 10–12 | ✅ [[pretraining-datasets]], [[token]], [[tokenization]], [[gpt-3]], [[foundation-model]], [[pretrained-weights]], [[mixture-model]], [[probability-distribution]] |
| 1.5 · Table 1.1 + sidebar | GPT-3 dataset details | 11 | ✅ [[pretraining-datasets]], [[gpt-3]] |
| 1.6 | A closer look at the GPT architecture | 12–14 | ✅ [[gpt-architecture]], [[decoder-only-transformer]], [[next-token-prediction]], [[self-supervised-learning]], [[autoregressive-text-generation]], [[emergent-behavior]], [[maximum-likelihood-estimation]], [[gpt-3]] |
| 1.7 | Building a large language model | 14–15 | ✅ [[llm-building-roadmap]], [[llm-training-loop]], [[llm-evaluation]], [[pretrained-weights]] |
| — | Summary | 15–16 | ✅ [[large-language-model]], [[emergent-behavior]], [[custom-llm]], [[fine-tuning]] |
| — | Exercises | — | none in Ch. 1 (Appendix C has no Ch. 1 solutions) |
| App. B | Ch. 1 references | 289–290 | ✅ listed below; people pages linked |

## TL;DR (5 bullets)
- An **LLM** is a deep neural network that is trained on huge amounts of text to **predict the next word**. "Large" means two things: many [[model-parameters]] (tens to hundreds of billions) and a very large training dataset.
- LLMs work so well for two reasons: the **[[transformer-architecture]]** (its [[self-attention]] lets the model "pay selective attention" to different parts of the input) and **vast training data**.
- You build an LLM in **two stages**. First, [[pretraining]] on raw, unlabeled text with [[self-supervised-learning]] (the next word is the label). This gives a [[foundation-model]]. Second, [[fine-tuning]] on a smaller labeled dataset, either [[instruction-fine-tuning]] or [[classification-fine-tuning]].
- The original transformer has an **encoder and a decoder** (built for translation). [[bert]] keeps only the encoder and learns by filling in masked words. **GPT** keeps only the decoder and generates text one word at a time ([[autoregressive-text-generation]]).
- Although GPT is trained only on next-word prediction, it shows **[[emergent-behavior]]**: translation, classification, summarization, plus [[zero-shot-learning]] and [[few-shot-learning]]. The book builds a GPT-like model in 3 stages and 9 steps ([[llm-building-roadmap]]).

## Key concepts introduced
**Foundations (§1.1)**
- [[large-language-model]] — a neural network that understands, generates, and responds to human-like text
- [[natural-language-processing]] — the field that LLMs "ushered into a new era"
- [[artificial-intelligence]] · [[machine-learning]] · [[deep-learning]] — the nested hierarchy of Fig. 1.1 (DL = networks with three or more layers)
- [[generative-ai]] — deep networks that create new content (text, images, other media)
- [[model-parameters]] — the adjustable weights that training optimizes
- [[next-token-prediction]] — the simple pretraining task behind everything
- [[feature-extraction]] — manual in traditional ML (spam trigger words, "!" counts), learned automatically in DL
- [[supervised-learning]] · [[empirical-risk-minimization]] · [[loss-function]] — the spam filter "minimizing the error in its predictions on a training dataset"

**Applications (§1.2)**
- [[llm-applications]] — translation, novel text, sentiment analysis, summarization, content creation, chatbots, knowledge retrieval
- [[machine-translation]] · [[text-classification]]

**Training stages (§1.3)**
- [[custom-llm]] — domain-specific models (BloombergGPT, medical QA) that can beat general LLMs
- [[pretraining]] · [[fine-tuning]] · [[self-supervised-learning]] · [[foundation-model]]
- [[text-completion]] · [[few-shot-learning]] — basic capabilities of a base model such as [[gpt-3]]
- [[instruction-fine-tuning]] (instruction–answer pairs) · [[classification-fine-tuning]] (text + class label)
- [[pytorch]] — the library the book uses

**Transformer (§1.4)**
- [[transformer-architecture]] · [[encoder-decoder-architecture]] · [[self-attention]] · [[attention-mechanism]]
- [[vector]] — the encoder turns text into embedding vectors
- [[bert]] · [[masked-language-modeling]] — encoder-only, good at classification
- [[gpt-architecture]] · [[decoder-only-transformer]] — decoder-only, good at generation
- [[zero-shot-learning]] · [[few-shot-learning]]

**Data (§1.5)**
- [[pretraining-datasets]] — the GPT-3 mixture of Table 1.1
- [[token]] · [[tokenization]] — named here, taught in Ch. 2
- [[mixture-model]] · [[probability-distribution]] — the sampling proportions behave like a categorical distribution over sources (wiki interpretation)
- [[pretrained-weights]] — load open weights to skip expensive pretraining

**GPT (§1.6)**
- [[autoregressive-text-generation]] — each output word becomes input for the next step
- [[maximum-likelihood-estimation]] — the formal reading of next-word training (wiki addition)
- [[emergent-behavior]] — abilities that were not trained directly

**Roadmap (§1.7)**
- [[llm-building-roadmap]] · [[llm-training-loop]] · [[llm-evaluation]]

**People**
- [[ashish-vaswani]] (transformer) · [[jacob-devlin]] (BERT) · [[tom-brown]] (GPT-3) · [[alec-radford]] (original GPT) · [[long-ouyang]] (InstructGPT) · [[alexey-dosovitskiy]] (Vision Transformer) · [[albert-gu]] and [[tri-dao]] (Mamba) · [[sebastian-raschka]] (author)

## Notable results / numbers
| Fact | Value | Book p. |
|---|---|---|
| Size of typical LLMs | tens to hundreds of billions of parameters | 2 |
| Deep learning means | neural networks with **3 or more layers** | 3 |
| Fig. 1.3 pretraining corpus | "trillions of words" of raw, unlabeled text | 6 |
| Transformer paper | 2017, "Attention Is All You Need" (arXiv 1706.03762); built for English→German and English→French | 7 |
| Fig. 1.4 translation example | "This is an example" → "Das ist ein Beispiel"; the decoder adds "Beispiel" | 8 |
| Fig. 1.6 zero-shot example | "Translate English to German: breakfast =>" → "Frühstück" | 10 |
| Fig. 1.6 few-shot example | "gaot => goat, sheo => shoe, pohne =>" → "phone" | 10 |
| GPT-3 data (Table 1.1) | CommonCrawl 410 B tokens (60%), WebText2 19 B (22%), Books1 12 B (8%), Books2 55 B (8%), Wikipedia 3 B (3%) | 11 |
| Total vs. used tokens | 499 B tokens available; GPT-3 trained on **300 B** | 11 |
| CommonCrawl storage | 410 B tokens ≈ 570 GB | 11 |
| LLaMA added sources | Arxiv (92 GB), StackExchange (78 GB) | 11 |
| Open replacement dataset | Dolma, three trillion tokens (Soldaini et al. 2024) | 11 |
| GPT-3 pretraining cost | about **$4.6 million** in cloud credits | 11 |
| Original transformer depth | encoder and decoder blocks repeated **6 times** | 13 |
| GPT-3 scale | **96** transformer layers, **175 billion** parameters, 2020 | 13 |
| Cost to pretrain from scratch | "thousands to millions of dollars" | 15 |
| Book roadmap (Fig. 1.9) | **3 stages, 9 steps** | 14 |

**Derived from Table 1.1 (wiki arithmetic, not stated in the book):** the proportions add up to 101% (rounding). They are *not* each source's share of the 499 B tokens. High-quality sources are upweighted: Wikipedia is seen about $0.03 \times 300 / 3 = 3$ times, WebText2 about 3.5 times, Books1 2 times, while Books2 and CommonCrawl are seen only about 0.44 times. See [[pretraining-datasets]].

**The math behind "predict the next word" (wiki addition; the book states it in words only):**
$$p_\theta(x_1,\dots,x_T)=\prod_{t=1}^{T} p_\theta(x_t \mid x_1,\dots,x_{t-1}), \qquad \mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$
Here $x_t$ is the token at position $t$ and $\theta$ are the model parameters. The left formula is the chain rule of probability. The right one is the loss that Ch. 5 implements. Minimizing it is [[maximum-likelihood-estimation]]. See [[next-token-prediction]].

### Figures
- **Fig. 1.1** (p. 3): AI ⊃ ML ⊃ DL ⊃ LLMs. GenAI sits inside DL and partly overlaps LLMs.
- **Fig. 1.2** (p. 5): ChatGPT writes a poem from user instructions.
- **Fig. 1.3** (p. 6): raw text → pretrained LLM (foundation model) → fine-tuned LLM.
- **Fig. 1.4** (p. 8): the original transformer in 8 numbered steps (encoder sees the full input; decoder writes one word at a time).
- **Fig. 1.5** (p. 9): BERT fills in masked words; GPT completes a partial sentence.
- **Fig. 1.6** (p. 10): text completion, zero-shot, and few-shot use.
- **Fig. 1.7** (p. 12): "The model is simply trained to predict the next" → "word".
- **Fig. 1.8** (p. 13): decoder-only GPT generating over 3 iterations; each output feeds the next input.
- **Fig. 1.9** (p. 14): the 3-stage, 9-step plan of the book.

### The 9-step roadmap (Fig. 1.9) and where the book does each step
| Stage | Steps | Chapter (from the TOC) |
|---|---|---|
| 1. Building an LLM | 1) Data preparation & sampling · 2) Attention mechanism · 3) LLM architecture | Ch. 2 · Ch. 3 · Ch. 4 |
| 2. Foundation model | 4) Pretraining · 5) Training loop · 6) Model evaluation · 7) Load pretrained weights | Ch. 5 |
| 3. Fine-tuning | 8) Classifier (dataset with class labels) · 9) Personal assistant (instruction dataset) | Ch. 6 · Ch. 7 |

## Exercises
Ch. 1 has **no exercises**. Appendix C starts its solutions at Ch. 2 (25 exercises in total for Ch. 2–7 and Appendix A).

## References cited (Appendix B, Ch. 1 group, book pp. 289–290)
| # | Reference | Why the book cites it |
|---|---|---|
| 1 | Wu et al. (2023), "BloombergGPT: A Large Language Model for Finance", arXiv 2303.17564 | A [[custom-llm]] pretrained on finance data beats ChatGPT on financial tasks |
| 2 | Singhal et al. (2023), "Towards Expert-Level Medical Question Answering with Large Language Models", arXiv 2305.09617 | Med-PaLM 2: a fine-tuned LLM beats general LLMs in medicine |
| 3 | Vaswani et al. (2017), "Attention Is All You Need", arXiv 1706.03762 | The original [[transformer-architecture]] ([[ashish-vaswani]]) |
| 4 | Devlin et al. (2018), "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv 1810.04805 | The original encoder-style model, [[bert]] ([[jacob-devlin]]) |
| 5 | Brown et al. (2020), "Language Models are Few-Shot Learners", arXiv 2005.14165 | [[gpt-3]], the template for the book's LLM ([[tom-brown]]) |
| 6 | Dosovitskiy et al. (2020), "An Image is Worth 16x16 Words", arXiv 2010.11929 | Vision Transformer: transformers are not limited to text ([[alexey-dosovitskiy]]) |
| 7 | Peng et al. (2023), "RWKV: Reinventing RNNs for the Transformer Era", arXiv 2305.13048 | A non-transformer (recurrent) LLM |
| 8 | Poli et al. (2023), "Hyena Hierarchy", arXiv 2302.10866 | A non-transformer (convolutional) LLM |
| 9 | Gu and Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv 2312.00752 | A non-transformer (state-space) LLM ([[albert-gu]], [[tri-dao]]) |
| 10 | Touvron et al. (2023), "Llama 2: Open Foundation and Fine-Tuned Chat Models", arXiv 2307.09288 | An openly available GPT-like model |
| 11 | Gao et al. (2020), "The Pile: An 800GB Dataset of Diverse Text for Language Modeling", arXiv 2101.00027 | More on [[pretraining-datasets]] |
| 12 | Ouyang et al. (2022), "Training Language Models to Follow Instructions with Human Feedback", arXiv 2203.02155 | InstructGPT, how GPT-3 became ChatGPT ([[long-ouyang]]) |

Also cited in the chapter body: Radford et al., "Improving Language Understanding by Generative Pre-Training" (original GPT, p. 12, [[alec-radford]]) and Soldaini et al. (2024), Dolma, arXiv 2402.00159 (p. 11).

> **Errata noted:** Appendix B prints the Llama 2 link as `2307.092881`. The correct ID is `2307.09288` (a footnote marker ran into the link). The book does not name the authors of "Attention Is All You Need" or BERT in the chapter body. The names come from Appendix B.

## What this changes in the wiki
- **Opens a new branch of the wiki:** LLMs and transformers. Ch. 1 creates the top-level concept pages that the rest of the book (Ch. 2–7) fills with math: [[large-language-model]], [[transformer-architecture]], [[gpt-architecture]], [[pretraining]], [[fine-tuning]], and [[llm-building-roadmap]].
- **Connects to existing MML pages:** next-word training is [[maximum-likelihood-estimation]] and [[empirical-risk-minimization]] with a cross-entropy [[loss-function]]. The GPT-3 data mixture links to [[probability-distribution]] and [[mixture-model]]. Fine-tuning on small datasets links to [[overfitting]]. Embeddings link to [[vector]].
- **New person pages:** [[ashish-vaswani]], [[jacob-devlin]], [[tom-brown]], [[alec-radford]], [[long-ouyang]], [[alexey-dosovitskiy]], [[albert-gu]], [[tri-dao]].
- **No contradictions** with existing pages. One caution for readers: the book uses "LLM" to mean a *transformer-based, GPT-like* LLM. The sidebar on p. 10 says that not all transformers are LLMs (for example, vision transformers), and that not all LLMs are transformers (RWKV, Hyena, Mamba).
- **Next:** Ch. 2 (working with text data) builds [[tokenization]], [[token]] IDs, and embeddings. Ch. 3 builds [[self-attention]].
