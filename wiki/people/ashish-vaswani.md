---
title: Ashish Vaswani
type: person
tags: [researcher, google-brain, transformer, attention]
sources: [llm-ch01-understanding-large-language-models, llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Ashish Vaswani (contemporary)

> **Why they matter to ML:** lead author of "Attention Is All You Need" (2017), the paper that introduced the transformer and scaled dot-product self-attention — the engine inside every modern LLM.

## Life (brief)
Computer scientist; at Google Brain when the transformer paper was written, later co-founder of
AI start-ups (Adept, Essential AI) (general knowledge, not in the book). The book cites the paper
only as "Vaswani et al." and does not give biographical detail.

## Key contributions
- **The transformer architecture** — "Attention Is All You Need" (2017) by Vaswani et al.,
  arXiv 1706.03762: "the original transformer architecture" (Appendix B, book p. 289). Built for
  [[machine-translation]] (English→German and English→French), with an **encoder** that reads the
  input and a **decoder** that writes the output one word at a time (§1.4, book pp. 7–8). Encoder
  and decoder blocks were each repeated **6** times (§1.6, book p. 13).
  → [[transformer-architecture]], [[encoder-decoder-architecture]]
- **Self-attention as scaled dot-product attention** — "The concept of self-attention as scaled
  dot-product attention was introduced in the original transformer paper" (Appendix B, book
  p. 291). It came "only three years" after Bahdanau attention for RNNs and was inspired by it
  (§3.2, book pp. 54–55):
  $$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
  ($Q,K,V$ = query, key, value matrices; $d_k$ = key dimension; standard notation, built in ch. 3.)
  → [[self-attention]], [[scaled-dot-product-attention]], [[query-key-value]], [[multi-head-attention]]
- **Fixed positional encodings** — the original transformer used fixed, predefined (sinusoidal — external detail, the book gives no formula)
  positional encodings, which GPT replaced with learned absolute positional embeddings (§2.8,
  book p. 45). → [[sinusoidal-positional-encoding]], [[absolute-positional-embedding]]
- **Post-LayerNorm** placement in the original model, later replaced by Pre-LN in GPT-2
  (§4.5, book p. 115). → [[pre-layernorm-vs-post-layernorm]]

## Connections
- Concepts: [[transformer-architecture]], [[self-attention]], [[scaled-dot-product-attention]], [[multi-head-attention]], [[encoder-decoder-architecture]], [[decoder-only-transformer]], [[attention-mechanism]]
- Related people: [[dzmitry-bahdanau]] (attention for RNNs, the precursor), [[alec-radford]] (GPT took the decoder half), [[jacob-devlin]] (BERT took the encoder half)
- Appears in sources: [[llm-ch01-understanding-large-language-models]], [[llm-ch03-coding-attention-mechanisms]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 7–10 (transformer, encoder/decoder; cited by title + arXiv only).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book p. 13 (encoder/decoder blocks repeated 6 times).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book p. 45 (fixed positional encodings of the original transformer).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.2, book pp. 54–55 (self-attention follows Bahdanau attention).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book p. 115 (Post-LayerNorm in the original transformer).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book p. 289 (Ch. 1 refs) and p. 291 (Ch. 3 refs).
- Biographical detail is general knowledge; the book names only "Vaswani et al.".
