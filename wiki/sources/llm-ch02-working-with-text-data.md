---
title: "LLM Ch.2 — Working with Text Data"
type: source
tags: [llm, tokenization, embeddings, data-loading, source]
created: 2026-10-13
updated: 2026-10-13
---

# LLM Ch.2 — Working with Text Data

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 2, **book pp. 17–49** (PDF pages 39–71)
- **Type:** book chapter — *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025)
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13 (every section's notes checked line by line against the extracted text and rendered figure pages)
- *Page rule:* PDF page = book page + 22.

This chapter is **step 1 of stage 1** in the book's three-stage roadmap (Figure 2.1, p. 18): before you can build attention or a GPT model, you have to turn raw text into numbers the network can learn from. The chapter walks the whole input pipeline: text → tokens → token IDs → token embeddings + positional embeddings → input embeddings, plus the sliding-window sampler that makes the training pairs.

## Section coverage (all of Ch. 2 accounted for)

None of the eight sections has numbered subsections (no 2.x.y headings in the book). Rows marked "·" are the unnumbered parts inside a section, listed so nothing is skipped.

| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Chapter intro (goals, Ch. 1 recap, Fig. 2.1 roadmap) | 17–18 | ✅ [[llm-building-roadmap]], [[pretraining]], [[fine-tuning]], [[decoder-only-transformer]] |
| 2.1 | Understanding word embeddings | 18–20 | ✅ [[word-embeddings]], [[embedding-dimension]], [[vector-space]] |
| · | Embeddings for video/audio/text (Fig. 2.2) | 18–19 | ✅ [[word-embeddings]], [[vector]] |
| · | Sentence/paragraph embeddings and RAG | 19 | ✅ [[retrieval-augmented-generation]] |
| · | Word2Vec and the 2-D scatterplot (Fig. 2.3) | 19–20 | ✅ [[word2vec]], [[dimensionality-reduction]] |
| · | Dimensionality (GPT-2: 768, GPT-3: 12,288) | 19–20 | ✅ [[embedding-dimension]] |
| 2.2 | Tokenizing text | 21–24 | ✅ [[tokenization]], [[token]] |
| · | "The Verdict" corpus, regex splitting, whitespace and case choices (Figs. 2.4–2.5) | 21–24 | ✅ [[tokenization]] |
| · | NOTE: real LLMs use many GB of text (millions of articles); one story suffices to teach | 22 | ✅ [[tokenization]] |
| 2.3 | Converting tokens into token IDs | 24–29 | ✅ [[token-id]], [[vocabulary]] |
| · | Building the vocabulary (Fig. 2.6, Listing 2.2) | 25–26 | ✅ [[vocabulary]] |
| · | `SimpleTokenizerV1`, encode/decode, inverse vocabulary (Figs. 2.7–2.8, Listing 2.3) | 26–28 | ✅ [[tokenizer-encode-decode]] |
| · | `KeyError: 'Hello'` — the unknown-word failure | 28 | ✅ [[out-of-vocabulary-words]] |
| 2.4 | Adding special context tokens | 29–32 | ✅ [[special-tokens]], [[end-of-text-token]] |
| · | `<\|unk\|>`, `<\|endoftext\|>`, `SimpleTokenizerV2` (Figs. 2.9–2.10, Listing 2.4) | 29–32 | ✅ [[special-tokens]], [[end-of-text-token]], [[out-of-vocabulary-words]] |
| · | [BOS], [EOS], [PAD]; what GPT actually uses | 32 | ✅ [[special-tokens]], [[padding-and-truncation]] |
| 2.5 | Byte pair encoding | 33–35 | ✅ [[byte-pair-encoding]], [[subword-tokenization]] |
| · | tiktoken GPT-2 encoding, vocab size 50,257 | 33–34 | ✅ [[byte-pair-encoding]], [[vocabulary]] |
| · | Unknown words → subwords (Fig. 2.11), Exercise 2.1 | 34 | ✅ [[subword-tokenization]], [[out-of-vocabulary-words]] |
| · | How BPE builds its vocabulary (frequency merges) | 34–35 | ✅ [[byte-pair-encoding]], [[histogram]] |
| 2.6 | Data sampling with a sliding window | 35–41 | ✅ [[sliding-window-sampling]], [[next-token-prediction]] |
| · | Input–target pairs shifted by 1 (Figs. 2.12–2.13) | 35–37 | ✅ [[next-token-prediction]], [[self-supervised-learning]], [[context-length]] |
| · | `GPTDatasetV1`, `create_dataloader_v1` (Listings 2.5–2.6) | 37–38 | ✅ [[pytorch-dataset-and-dataloader]], [[tensor]] |
| · | Stride, batch size, overlap (Fig. 2.14), Exercise 2.2 | 39–41 | ✅ [[sliding-window-sampling]], [[mini-batch]], [[hyperparameter]] |
| 2.7 | Creating token embeddings | 41–44 | ✅ [[token-embedding]], [[embedding-layer]] |
| · | `torch.nn.Embedding` as a row lookup (Figs. 2.15–2.16) | 41–44 | ✅ [[embedding-layer]], [[matrix]] |
| · | Lookup = one-hot × weight matrix | 43 | ✅ [[one-hot-encoding]], [[linear-mapping]] |
| 2.8 | Encoding word positions | 43–48 | ✅ [[positional-embedding]], [[input-embeddings]] |
| · | Same ID → same vector; attention has no order (Fig. 2.17) | 43–45 | ✅ [[positional-embedding]] |
| · | Absolute vs relative positions (Fig. 2.18) | 45 | ✅ [[absolute-positional-embedding]], [[relative-positional-embedding]] |
| · | Learned (GPT) vs fixed (original transformer) encodings | 45 | ✅ [[sinusoidal-positional-encoding]] |
| · | 50,257 × 256 token layer, context length, broadcast addition (Fig. 2.19) | 46–48 | ✅ [[context-length]], [[broadcasting]], [[input-embeddings]] |
| — | Summary | 48–49 | — (recap of the above) |
| App. B | References for Ch. 2 | 290–291 | see *References cited* below |
| App. C | Solutions to Exercises 2.1–2.2 | 300–301 | see *Exercises* below |

> ✅ **Every section of Ch. 2 has a concept page.** The summary and the appendix entries have no new content of their own.

## TL;DR (5 bullets)
- **Neural networks cannot read text.** Text is categorical, so each word must become a continuous [[vector]] — an *embedding*, a map from discrete objects to points in a continuous [[vector-space]] (p. 19).
- **Text → tokens → IDs.** A [[tokenization|tokenizer]] splits text into [[token|tokens]]; a [[vocabulary]] maps each unique token to an integer [[token-id]]. Simple word-level vocabularies break on unseen words, which is why [[special-tokens]] like `<|unk|>` and `<|endoftext|>` exist.
- **GPT uses byte pair encoding.** [[byte-pair-encoding|BPE]] (via `tiktoken`, 50,257 tokens) splits unknown words into subwords or single characters, so it never needs `<|unk|>` and can encode any text.
- **Training pairs come from the text itself.** A [[sliding-window-sampling|sliding window]] cuts the token stream into inputs and targets that are the inputs shifted by one token — the raw material for [[next-token-prediction]].
- **Embeddings are a learnable lookup table.** An [[embedding-layer]] is a $|V| \times d$ [[matrix]] whose rows are picked by token ID (the same as one-hot × matrix). Because the lookup ignores position, GPT adds learned [[absolute-positional-embedding|absolute positional embeddings]] to get the final [[input-embeddings]].

## Key concepts introduced
- [[word-embeddings]] — words as points in $\mathbb{R}^d$, where nearby means similar
- [[word2vec]] — an early, popular way to learn word embeddings from context
- [[embedding-dimension]] — the size $d$ of each embedding vector (768 for small GPT-2, 12,288 for GPT-3)
- [[retrieval-augmented-generation]] — named as a use of sentence/paragraph embeddings; out of the book's scope
- [[dimensionality-reduction]] — projecting high-dimensional embeddings to 2-D to look at them
- [[tokenization]] — splitting text into tokens
- [[token]] — the unit a model reads (word, punctuation, subword, or character)
- [[vocabulary]] — the token ↔ integer table
- [[token-id]] — the integer that stands for one token
- [[tokenizer-encode-decode]] — `encode` (text → IDs) and `decode` (IDs → text)
- [[out-of-vocabulary-words]] — words the vocabulary does not know
- [[special-tokens]] — `<|unk|>`, `<|endoftext|>`, [BOS], [EOS], [PAD]
- [[end-of-text-token]] — the GPT separator between unrelated documents (ID 50256)
- [[padding-and-truncation]] — making sequences in a batch the same length
- [[byte-pair-encoding]] — frequency-based merging of characters into subwords
- [[subword-tokenization]] — the general family BPE belongs to
- [[histogram]] — BPE's merges are chosen from pair-frequency counts
- [[sliding-window-sampling]] — inputs and shifted-by-one targets with a stride
- [[next-token-prediction]] / [[self-supervised-learning]] — targets come from the text, no labels needed
- [[context-length]] — how many tokens the model sees at once
- [[pytorch-dataset-and-dataloader]], [[tensor]], [[mini-batch]], [[hyperparameter]] — the PyTorch data machinery
- [[token-embedding]] / [[embedding-layer]] — the trainable lookup table
- [[one-hot-encoding]], [[matrix]], [[linear-mapping]] — the math behind the lookup
- [[positional-embedding]], [[absolute-positional-embedding]], [[relative-positional-embedding]], [[sinusoidal-positional-encoding]] — how order gets in
- [[broadcasting]] — how a $4\times256$ position tensor adds to an $8\times4\times256$ batch
- [[input-embeddings]] — token + position, the final input to the GPT model

## Notable results / numbers
- **Figure 2.2 embeddings** (p. 19), all 3-D: video $[1.23,-0.31,0.89]$, audio $[-0.15,0.45,2.11]$, text $[1.78,0.18,-2.10]$.
- **Embedding sizes** (p. 20): smallest [[gpt-2]] models (117M and 125M parameters) → 768 dims; largest [[gpt-3]] (175B) → 12,288 dims.
- **"The Verdict"** (Edith Wharton, public domain): 20,479 characters (p. 22) → **4,690** regex tokens (p. 24) → **1,130** unique tokens (p. 25) → **1,132** with `<|endoftext|>` = 1130 and `<|unk|>` = 1131 (p. 30). With BPE the same story is **5,145** tokens (p. 35).
- **SimpleTokenizerV2 round trip** (p. 32): "Hello" and "palace" are not in "The Verdict", so both become `<|unk|>` and are lost on decode — encode-then-decode is not the identity.
- **GPT-2 BPE vocabulary** = **50,257** tokens; `<|endoftext|>` has the largest ID, **50256** (p. 34). Space alone is ID 220.
- **Figure 2.11** (p. 34): "Akwirw ier" → "Ak" 33901, "w" 86, "ir" 343, "w" 86, " " 220, "ier" 959.
- **First sliding window** (p. 36): x = `[290, 4920, 2241, 287]` (" and established himself in"), y = `[4920, 2241, 287, 257]` (y adds " a").
- **Data-loader defaults** (p. 38): `batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0`. `drop_last` avoids loss spikes from a short last batch. The book says LLMs are commonly trained with inputs of at least 256 tokens (p. 39).
- **Stride = max_length** uses every token once with no overlap; more overlap could mean more [[overfitting]] (p. 41).
- **Batch-size trade-off** (pp. 39–40): small batches need less memory but give noisier updates.
- **Embedding lookup** (pp. 42–43): with `torch.manual_seed(123)`, a $6\times3$ weight matrix; ID 3 → row 3 = $[-0.4015, 0.9666, -1.1481]$. IDs `[2,3,5,1]` → a $4\times3$ output.
- **Realistic shapes** (pp. 46–47): token layer $50{,}257 \times 256$; batch IDs $8\times4$ → token embeddings $8\times4\times256$; positional layer output $4\times256$; sum by [[broadcasting]] → $8\times4\times256$.
- **Core formula** (our formalisation of p. 45): $\mathbf{x}_t = E_{\text{tok}}[\text{id}_t] + E_{\text{pos}}[t]$, with $E_{\text{tok}}\in\mathbb{R}^{|V|\times d}$ and $E_{\text{pos}}\in\mathbb{R}^{L\times d}$.
- **Book quirk:** the Figure 2.17 caption says "token ID 5", but the figure shows token ID 2 ("fox") repeated (p. 44).

## Exercises (Appendix C, pp. 300–301)
| # | Task (book p.) | Solution idea |
|---|---|---|
| 2.1 | Encode the unknown words "Akwirw ier" with the tiktoken GPT-2 tokenizer, decode each ID on its own, then decode the whole list (p. 34). | You get `[33901, 86, 343, 86, 220, 959]` → "Ak", "w", "ir", "w", " ", "ier". Decoding the full list gives back exactly "Akwirw ier". BPE never fails: it falls back to known subwords and single characters ([[byte-pair-encoding]]). |
| 2.2 | Run the data loader with `max_length=2, stride=2` and with `max_length=8, stride=2` (p. 39). | With stride = max_length the rows do not overlap (`[40, 367]`, `[2885, 1464]`, …). With stride 2 < max_length 8, each row is the previous row shifted left by 2 (overlap 6). In general overlap = max_length − stride; more overlap gives more samples but more redundancy ([[sliding-window-sampling]]). |

## References cited (Appendix B, pp. 290–291)
1. Raschka (2023), *Machine Learning Q and AI*, ch. 1 — embedding spaces vs latent spaces ([[sebastian-raschka]]).
2. Sennrich et al. (2015), "Neural Machine Translation of Rare Words with Subword Units", arXiv 1508.07909 — in-depth BPE ([[rico-sennrich]]).
3. OpenAI, GPT-2 `encoder.py` — the open-sourced BPE tokenizer used to train GPT-2.
4. OpenAI Tokenizer web UI (platform.openai.com/tokenizer) — interactive view of GPT's BPE.
5. Karpathy, "minbpe" — a minimal BPE tokenizer you can read and train yourself ([[andrej-karpathy]]).
6. Kudo & Richardson (2018), "SentencePiece" — an alternative tokenizer used by some LLMs.
7. Song et al. (2020), "Fast WordPiece Tokenization", arXiv 2012.15524 — WordPiece, used by [[bert]].

## What this changes in the wiki
- **New LLM layer on top of the MML foundation.** Ch. 2 creates the whole tokenization and input-embedding cluster (about 25 concept pages, from [[tokenization]] to [[input-embeddings]]) plus person pages [[rico-sennrich]] and [[andrej-karpathy]].
- **Links back to linear algebra.** The embedding lookup is a concrete use of [[matrix]] row selection and [[linear-mapping]] (one-hot vector × weight matrix). Embeddings give [[vector-space]] and [[dimensionality-reduction]] a practical ML meaning. Those MML pages get LLM sections; no contradictions with the MML material.
- **Links to optimisation and statistics.** Batch size ties into [[stochastic-gradient-descent]] and [[mini-batch]]; overlapping windows tie into [[overfitting]]; BPE merges tie into [[histogram]] counting; trainable embeddings tie into [[backpropagation]].
- **Forward pointers.** "No access to words past the target" (Fig. 2.12) foreshadows the [[causal-attention-mask]]; position-agnostic attention motivates [[positional-embedding]] and leads straight into [[self-attention]] in Ch. 3.
- **Clearly marked as ours, not the book's:** the $E:\mathcal{V}\to\mathbb{R}^d$ formalisation, the BPE argmax-merge formula, the row-count formula $\lceil (N-L)/\text{stride}\rceil$, the $1/B$ gradient-variance remark, the sinusoidal formula, and the $\mathcal{N}(0,1)$ default init of `nn.Embedding`.
