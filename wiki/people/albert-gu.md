---
title: Albert Gu
type: person
tags: [state-space-models, sequence-modeling, non-transformer-llm]
sources: [llm-ch01-understanding-large-language-models]
created: 2026-10-13
updated: 2026-10-13
---

# Albert Gu (active 2010s–present)

> **Why they matter to ML:** lead author of **Mamba** and the structured state-space model line (S4), the best-known attempt to build LLMs *without* attention.

## Life (brief)
Albert Gu did his PhD at Stanford (advised by Christopher Ré), where he developed S4, "Structured State Spaces." He became an assistant professor at Carnegie Mellon University and co-founded the company Cartesia. (Biographical details are general knowledge. The book names him only in the appendix B citation "Gu and Dao.")

## Key contributions
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu and Dao 2023, arXiv 2312.00752), with [[tri-dao]]. The book lists it with RWKV and Hyena as "experimental (but less popular) LLM architectures" that show "not all LLMs need to be based on the transformer architecture." → [[large-language-model]], [[transformer-architecture]]
  - **Why bother?** Section 1.4 says the main reason for these alternatives "is to improve the computational efficiency of LLMs." Self-attention compares every token with every other token, so its cost grows like $n^2$ for $n$ tokens. Mamba's cost grows like $n$. → [[self-attention]], [[context-length]]
  - **The core recurrence (external context):** $\mathbf{h}_t = \bar{A}_t\,\mathbf{h}_{t-1} + \bar{B}_t\,x_t$ and $y_t = C_t\,\mathbf{h}_t$. Symbol by symbol: $t$ = the position in the sequence; $x_t$ = the input at that step; $\mathbf{h}_t$ = the **hidden state**, a fixed-size memory [[vector]] that summarizes everything so far; $\bar{A}_t$ = a [[matrix]] that decides how much old memory to keep; $\bar{B}_t$ = how much of the new input to write in; $C_t$ = how to read the memory out into the output $y_t$. This looks like an [[recurrent-neural-network|RNN]] update. → [[hidden-state]], [[linear-mapping]]
  - **"Selective"** means $\bar{A}_t$, $\bar{B}_t$, $C_t$ depend on the current input $x_t$ (the small $t$ subscript shows this). So the model can choose to remember an important token or forget filler. Older state-space models used the same fixed matrices at every step. → [[long-range-dependencies]]
- **Where it stands.** The book says it "remains to be seen" whether such architectures can match transformer-based LLMs, and uses "LLM" to mean transformer-based, GPT-like models. → [[gpt-architecture]]

## Connections
- Concepts: [[large-language-model]], [[transformer-architecture]], [[self-attention]], [[recurrent-neural-network]], [[hidden-state]], [[long-range-dependencies]], [[context-length]]
- People: [[tri-dao]] (Mamba co-author), [[ashish-vaswani]] (transformer)
- Appears in sources: [[llm-ch01-understanding-large-language-models]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book p. 10 ("Transformers vs. LLMs" sidebar: LLMs based on recurrent and convolutional architectures; efficiency motivation; "remains to be seen").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 1 references, book p. 290 (Mamba, Gu and Dao 2023, with RWKV and Hyena).
- Biography and the state-space recurrence are general knowledge from the Mamba paper, not from the book.
