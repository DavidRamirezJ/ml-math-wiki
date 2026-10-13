---
title: Andrej Karpathy
type: person
tags: [education, gpt, tokenization, open-source]
sources: [llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Andrej Karpathy (b. 1986)

> **Why they matter to ML:** he writes tiny, readable, from-scratch versions of big ideas. His **nanoGPT** (a GPT-2 in a few hundred lines) shaped how this book splits its GPT model into modules, and his **minbpe** is the book's suggested place to learn how to code a BPE tokenizer yourself.

## Life (brief)
Andrej Karpathy did his PhD at Stanford with Fei-Fei Li on deep learning for images and language, and created the Stanford course CS231n. He was a founding member of OpenAI, led AI for Tesla Autopilot, briefly went back to OpenAI, and now focuses on AI education (e.g. his "Neural Networks: Zero to Hero" video series). (These biographical details are general knowledge. The book mentions him only in appendix B.)

## Key contributions
- **nanoGPT** ("NanoGPT, a Repository for Training Medium-Sized GPTs", github.com/karpathy/nanoGPT). The book calls it "a minimalist yet efficient implementation of a GPT-2 model, similar to the model implemented in this book." The book's code is different, but nanoGPT "inspired the reorganization of a large GPT Python parent class implementation into smaller submodules." → [[gpt-architecture]], [[gpt-2]], [[pytorch-nn-module]]
  - In practice that means the book's Chapter 4 builds the model from small Lego bricks: `LayerNorm`, `GELU`, `FeedForward`, `MultiHeadAttention`, then a `TransformerBlock`, and finally `GPTModel`, which stacks 12 blocks. → [[layer-normalization]], [[gelu]], [[feed-forward-network]], [[multi-head-attention]], [[transformer-block]]
  - Both target the same GPT-2 small config. The book's version is:
    ```python
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }
    ```
    → [[gpt-model-configuration]], [[model-parameters]]
  - The model's job is [[next-token-prediction]]: it outputs [[logits]] $\mathbf{z} \in \mathbb{R}^{|V|}$ for each position. Symbol by symbol: $\mathbf{z}$ = the score vector; $\in$ = "is a member of"; $\mathbb{R}^{|V|}$ = the set of real-number vectors of length $|V|$, the [[vocabulary]] size (50,257). A [[softmax]] turns those scores into probabilities over the next [[token]].
- **minbpe** ("A Minimal Implementation of a BPE Tokenizer", github.com/karpathy/minbpe). The book points readers here who want to code and train a [[byte-pair-encoding]] tokenizer "from the ground up," because the book itself uses OpenAI's ready-made `tiktoken` library and calls a full BPE implementation out of scope. → [[byte-pair-encoding]], [[subword-tokenization]], [[tokenization]]
  - The core loop minbpe implements (general knowledge, matching the book's §2.5 sketch): count every neighbouring pair of tokens, merge the most frequent one into a new token, repeat.
    $$(a^*,b^*) = \arg\max_{(a,b)} \mathrm{count}(a,b)$$
    Symbol by symbol: $(a,b)$ = two tokens side by side; $\mathrm{count}(a,b)$ = how often they appear together; $\arg\max$ = "the pair that wins" ([[argmax]]); $(a^*,b^*)$ = the pair you glue together next. GPT-2's real tokenizer ends with 50,257 tokens after this process. → [[histogram]]

## Connections
- Concepts: [[gpt-architecture]], [[gpt-2]], [[transformer-block]], [[pytorch-nn-module]], [[gpt-model-configuration]], [[byte-pair-encoding]], [[tokenization]], [[pretraining]], [[large-language-model]]
- People: [[alec-radford]] (GPT-2, which nanoGPT reproduces), [[rico-sennrich]] (BPE for subword tokenization), [[sebastian-raschka]] (the book's author, whose code structure nanoGPT inspired)
- Appears in sources: [[llm-ch02-working-with-text-data]], [[llm-ch04-implementing-a-gpt-model]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 2 references, book p. 291 (minbpe as a minimal, readable BPE implementation).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 4 references, book p. 293 (nanoGPT: minimalist GPT-2 implementation that inspired the book's submodule structure).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 33–35 (BPE out of scope, `tiktoken`, vocabulary 50,257); §4.1, book p. 95 (`GPT_CONFIG_124M`).
- Biography and the description of minbpe's merge loop are general knowledge, not from the book.
