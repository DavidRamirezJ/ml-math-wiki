---
title: LLM Pretraining Datasets and Data Mixtures
type: concept
tags: [llm, pretraining, datasets, sampling, mixture-model, tokens]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data]
created: 2026-10-13
updated: 2026-10-13
---

# LLM Pretraining Datasets and Data Mixtures

> **One-liner:** An LLM is pretrained on a gigantic pile of ordinary, unlabeled text, and the recipe says how much of each source (web, books, Wikipedia…) goes into the mix.

## 🟢 Intuition (Dummy)
Before an LLM can do anything clever, it has to *read*. A lot. Think hundreds of billions of
words: web pages, books, Wikipedia, research papers, code Q&A. This reading material is the
**pretraining dataset**, also called the training **corpus**.

Two things make it special:
1. **No labels needed.** Nobody tags the text. The model makes its own "quiz questions" by
   hiding the next word and guessing it ([[self-supervised-learning]]). So people call it
   **raw text** (book p. 6).
2. **It's a recipe, not just a pile.** Like a smoothie, you choose the proportions. GPT-3's
   recipe puts a *small* amount of high-quality Wikipedia in on purpose, and gives it more weight
   than its size alone would. Most of the volume is filtered web crawl, but less of it than you
   might expect.

Why so much and so varied? Because breadth is what makes the model good at many things:
grammar, meaning, context, and even some general knowledge (book p. 11).

## 🟡 Mechanics (Practitioner)

### Raw text (§1.3, book p. 6)
"Raw" means regular text **without any labeling information**. It can still be cleaned, for
example by removing formatting characters or documents in unknown languages (book p. 6).
Fig. 1.3 sizes it at **trillions of words** from internet texts, books, Wikipedia and research
articles.

### Scale and diversity (§1.5, book pp. 10–11)
Training sets for popular GPT- and BERT-like models are "diverse and comprehensive text corpora
encompassing billions of words," covering a vast array of topics and both natural and computer
languages (book p. 10). GPT-3, whose recipe is in Table 1.1, was the base model for the first
version of ChatGPT. The main takeaway: scale and diversity let these models do well on diverse
tasks (syntax, semantics, context, even some general knowledge) (book p. 11).

### Tokens: the unit of counting (§1.5, book pp. 10–11)
Dataset sizes are counted in [[token|tokens]]. A token is roughly a word or a punctuation
character, so "number of tokens ≈ words + punctuation" (book p. 11). Turning text into tokens
is [[tokenization]] (Ch. 2).

### The GPT-3 recipe (Table 1.1, book p. 11)

| Dataset | Description | Tokens $N_d$ | Share of training data $p_d$ |
|---|---|---|---|
| CommonCrawl (filtered) | Web crawl data | 410 B | 60% |
| WebText2 | Web crawl data | 19 B | 22% |
| Books1 | Internet-based book corpus | 12 B | 8% |
| Books2 | Internet-based book corpus | 55 B | 8% |
| Wikipedia | High-quality text | 3 B | 3% |

Facts from the sidebar "GPT-3 dataset details" (book p. 11):
- the subsets total $\sum_d N_d = 410+19+12+55+3 = 499$ B tokens, but GPT-3 trained on only
  $N_{\text{train}} = 300$ B tokens. The GPT-3 authors don't say why;
- the shares add to 101%, which is 100% "adjusted for rounding errors";
- CommonCrawl alone is about **570 GB** of storage;
- later models such as Meta's **LLaMA** added sources like **Arxiv** papers (92 GB) and
  **StackExchange** code Q&As (78 GB);
- GPT-3's data was never released. A public stand-in is **Dolma**, an open corpus of **three
  trillion tokens** (Soldaini et al. 2024);
- such collections may contain copyrighted work, and the rules depend on use case and country.

### Data-mixture sampling (our formalization of Table 1.1)
Training draws each piece of text from source $d$ with probability $p_d$, a categorical
[[probability-distribution]], i.e. a [[mixture-model]] over sources:
$$P_{\text{train}}(\text{text}) = \sum_{d\in\mathcal D} p_d\, P_d(\text{text}),\qquad \sum_{d\in\mathcal D} p_d = 1$$
Symbol by symbol:
- $\mathcal D$ = the set of sources {CommonCrawl, WebText2, Books1, Books2, Wikipedia};
- $P_d$ = "what text from source $d$ looks like" (its own distribution);
- $p_d$ = the mixture weight, the share of training tokens taken from source $d$ (the 60%, 22%, … column);
- $\sum_d p_d = 1$ = the weights add up to 100%, like any probability vector.

**How many times is each source read?** If $N_d$ is the size of source $d$, the number of passes
(epochs) over it is
$$e_d = \frac{p_d \cdot N_{\text{train}}}{N_d}$$
- $p_d\cdot N_{\text{train}}$ = how many training tokens come from source $d$;
- divide by its size $N_d$ to get "how many times over" it was read.

Worked numbers (derived from the table, not printed in the book):

| Source | $e_d$ | Meaning |
|---|---|---|
| Wikipedia | $0.03\times300/3 = 3.0$ | read about 3 times |
| WebText2 | $0.22\times300/19 \approx 3.5$ | read 3 to 4 times |
| Books1 | $0.08\times300/12 = 2.0$ | read twice |
| Books2 | $0.08\times300/55 \approx 0.44$ | less than half read |
| CommonCrawl | $0.60\times300/410 \approx 0.44$ | less than half read |

So the mix **upweights** small high-quality sources and **downweights** the huge web crawl.
Wikipedia is $3/499\approx 0.6\%$ of the tokens but 3% of training. CommonCrawl is
$410/499\approx 82\%$ of the tokens but 60% of training.

### Scale in the book vs. the real world (§2.2, book pp. 21–22)
The book trains its toy model on one public-domain short story, Edith Wharton's **"The
Verdict"**: **20,479 characters**, which its simple regex tokenizer splits into **4,690 tokens**
(without whitespace; book pp. 22, 24). That's about $20479/4690\approx 4.4$ characters per token (our arithmetic).
The NOTE on book p. 22 says real LLM work uses millions of articles and hundreds of thousands of
books, many gigabytes of text. One story is enough to learn the mechanics on a laptop. The book
picks it because it's in the **public domain**, so it may be used for training (book p. 21).

### Open datasets the book points to (appendix B, book pp. 290, 294–295)
- **The Pile** (Gao et al. 2020), an 800 GB diverse text dataset curated by EleutherAI.
- **Dolma** (Soldaini et al. 2024), three trillion tokens.
- **RefinedWeb** (Penedo et al. 2023), web-only data for Falcon.
- **RedPajama** (Together AI).
- **FineWeb**, more than 15 trillion tokens of cleaned, deduplicated English CommonCrawl data.
- **Project Gutenberg** recipe (book's bonus code): 60,000 public-domain books.
- Fully documented recipes with data details: **Pythia** (Biderman et al. 2023), **OLMo** (Groeneveld et al. 2024).

## 🔴 Theory (Expert)
- **Pretraining = ERM on the mixture.** The pretraining loss is the average next-token
  [[cross-entropy-loss]] over text sampled from $P_{\text{train}}$:
  $$\mathcal L(\theta) = \mathbb E_{\text{text}\sim P_{\text{train}}}\Big[-\tfrac1T\textstyle\sum_{t}\log p_\theta(x_t\mid x_{<t})\Big] = \sum_d p_d\,\mathcal L_d(\theta)$$
  Here $\mathcal L_d$ = the loss on source $d$ alone, and $T$ = the length of one text. By
  linearity of expectation the total is a $p_d$-weighted sum of per-source losses. So **the
  mixture weights are literally the importance weights of each source in the objective**
  ([[empirical-risk-minimization]], [[expected-value-and-mean]], [[maximum-likelihood-estimation]]).
  Choosing $p_d \ne N_d/\sum N_d$ is a deliberate change of the target distribution. This is
  our derivation; the book gives only the table.
- **Repetition vs. diversity.** $e_d>1$ means the model sees the same Wikipedia text about 3 times.
  More passes give more weight to quality text, but too many risk memorization
  ([[overfitting]]). $e_d<1$ means the model never sees most of the web crawl, which is fine when
  the source is huge and redundant. The book notes the GPT-3 authors never explained why they
  stopped at 300 B of 499 B tokens (book p. 11).
- **Why scale helps generalization.** A larger, more diverse sample makes empirical risk a better
  estimate of the true risk over "all text," and it covers more tasks implicitly. That's the
  source of the zero-/few-shot and other [[emergent-behavior]] the book describes (book pp. 11, 14).
  It's also why base models are useful as [[foundation-model]]s for downstream
  [[fine-tuning]] on much smaller labeled sets (book pp. 11–12).
- **Cost.** GPT-3 pretraining is estimated at about **\$4.6 million** in cloud credits (book p. 11).
  Appendix B cites a Lambda Labs estimate that training GPT-3 on a single RTX 8000 GPU would take
  **665 years** (book p. 293). This is why the book pretrains only a tiny educational model on
  consumer hardware and then loads open [[pretrained-weights]] (book p. 12).
- **A number that doesn't quite add up.** 570 GB for 410 B tokens is about 1.4 bytes per token (our
  arithmetic). Typical English BPE tokens are around 4 characters, so the two figures probably
  don't describe exactly the same thing (raw vs. filtered size, or compressed storage). The book
  doesn't explain it. Treat 570 GB as a rough order of magnitude.
- **Domain mixes.** Custom models change the mixture on purpose. BloombergGPT trained on both
  general and finance corpora (appendix B, book p. 294). See [[custom-llm]].
- **Licensing is part of the data.** Public-domain text (Wikisource, Project Gutenberg) is safe to
  use. Large web crawls may contain copyrighted works (book pp. 11, 21).

## Connections
- Builds on: [[token]], [[tokenization]], [[probability-distribution]], [[mixture-model]], [[expected-value-and-mean]], [[histogram]]
- Used by: [[pretraining]], [[self-supervised-learning]], [[next-token-prediction]], [[foundation-model]], [[large-language-model]], [[gpt-3]]
- Leads to: [[fine-tuning]], [[custom-llm]], [[sliding-window-sampling]], [[byte-pair-encoding]]
- Related: [[empirical-risk-minimization]], [[maximum-likelihood-estimation]], [[cross-entropy-loss]], [[overfitting]], [[emergent-behavior]], [[pretrained-weights]]
- People: [[tom-brown]] (GPT-3 paper), [[sebastian-raschka]]

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** The NOTE on book p. 141 says that a larger corpus could replace *The Verdict*: the book's bonus code prepares more than 60,000 public-domain books from Project Gutenberg. §5.2 ends by contrasting the 5,145-token toy text with real pretraining, which uses much larger datasets and often only about one epoch (book p. 151). Appendix B for Ch. 5 lists the open corpora Dolma, The Pile, RefinedWeb, RedPajama and FineWeb (more than 15 trillion tokens) (book pp. 293–295).

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book p. 6 (raw text, filtering, trillions of words, Fig. 1.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book pp. 10–12 (Table 1.1, sidebar "GPT-3 dataset details", tokens, Dolma, \$4.6 M, open weights).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book p. 14 (emergent behavior from multilingual data).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.2, book pp. 21–24 ("The Verdict", 20,479 characters, 4,690 tokens, data-scale NOTE, public domain).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 290, 293–295 (665-year GPT-3 estimate; The Pile, Pythia, OLMo, Gutenberg, BloombergGPT, Dolma, RefinedWeb, RedPajama, FineWeb).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3, book p. 141 (Project Gutenberg NOTE); §5.2, book p. 151 (one-epoch practice).
