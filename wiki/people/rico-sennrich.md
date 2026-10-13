---
title: Rico Sennrich
type: person
tags: [tokenization, machine-translation, normalization, nlp]
sources: [llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Rico Sennrich (active 2010s–present)

> **Why they matter to ML:** he brought **byte pair encoding (BPE)** into neural language processing, which is the tokenizer behind GPT-2, GPT-3 and the original ChatGPT, and he co-invented **RMSNorm**, the cheap normalization layer that most modern LLMs use.

## Life (brief)
Rico Sennrich is a computational linguist who works on [[machine-translation]]. He did the BPE work at the University of Edinburgh and later became a professor at the University of Zurich. (These biographical details are general knowledge. The book names him only in appendix B citations: "Sennrich et al." and "Zhang and Sennrich.")

## Key contributions
- **Subword units for neural machine translation, via BPE** (Sennrich, Haddow & Birch, "Neural Machine Translation of Rare Words with Subword Units", 2015, arXiv 1508.07909). The book cites this paper for "an in-depth discussion" of [[byte-pair-encoding]] as a tokenization method. Today it is simply called **BPE tokenization** or [[subword-tokenization]]. → [[byte-pair-encoding]], [[tokenization]], [[out-of-vocabulary-words]]
  - **The problem it fixes:** a word-level tokenizer has a fixed [[vocabulary]], so any new word becomes `<|unk|>` and its meaning is lost. Sennrich's idea was: break rare words into smaller pieces that *are* in the vocabulary. The book's Figure 2.11 shows this: the made-up words `"Akwirw ier"` become 6 tokens, "Ak", "w", "ir", "w", " ", "ier", with IDs `[33901, 86, 343, 86, 220, 959]`. No `<|unk|>` is needed. → [[special-tokens]]
  - **How the vocabulary is built** (the book describes this in words, §2.5): start with every single character, then keep merging the pair of neighbouring tokens that appears most often. In LaTeX:
    $$(a^*,b^*) = \arg\max_{(a,b)} \mathrm{count}(a,b),\qquad V_{t+1} = V_t \cup \{a^*b^*\}$$
    Symbol by symbol: $(a,b)$ = a pair of tokens that sit next to each other in the training text; $\mathrm{count}(a,b)$ = how many times that pair shows up; $\arg\max$ = "pick the pair with the biggest count" (see [[argmax]]); $(a^*,b^*)$ = that winning pair; $V_t$ = the vocabulary after $t$ merges; $\cup$ = "add to the set"; $a^*b^*$ = the two pieces glued into one new token. You repeat this until counts fall below a cutoff or you hit the target size, which is $|V| = 50{,}257$ for GPT-2. → [[histogram]]
  - In the book you never code BPE yourself. You call OpenAI's `tiktoken` library:
    ```python
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    integers = tokenizer.encode("Akwirw ier")
    ```
    → [[tokenizer-encode-decode]], [[token-id]]
- **RMSNorm** (Zhang & Sennrich, "Root Mean Square Layer Normalization", 2019, arXiv 1910.07467), with [[biao-zhang]]. The book says RMSNorm "simplifies" LayerNorm by normalizing with only the root mean square of the inputs, without subtracting the mean first, so "it does not center the data before computing the scale." It is cheaper, which is why many newer LLMs use it instead of GPT-2's [[layer-normalization]]. → [[rmsnorm]]
  - The standard formula (from the paper; the book gives it only in words):
    $$\mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon},\qquad \mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})}\odot \boldsymbol{\gamma}$$
    Symbol by symbol: $\mathbf{x}$ = one token's [[vector]] of activations; $d$ = its length (the [[embedding-dimension]], 768 for GPT-2 124M); $x_i$ = the $i$-th number in it; $\sum_{i=1}^{d} x_i^2$ = add up all the squares; $\frac{1}{d}$ = turn the sum into an average; $\epsilon$ = a tiny number so we never divide by zero ([[numerical-stability]]); $\sqrt{\cdot}$ = square root, giving the "typical size" of the entries; $\odot$ = multiply entry by entry; $\boldsymbol{\gamma}$ = a learned scale vector.
  - **Compare with LayerNorm**, which is $\gamma \odot \dfrac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$. RMSNorm drops the mean $\mu$ and the shift $\beta$. So it only rescales; it does not recentre. → [[standardization-z-score]], [[covariance-and-variance]]

## Connections
- Concepts: [[byte-pair-encoding]], [[subword-tokenization]], [[tokenization]], [[vocabulary]], [[out-of-vocabulary-words]], [[machine-translation]], [[rmsnorm]], [[layer-normalization]], [[gpt-2]]
- People: [[biao-zhang]] (RMSNorm co-author), [[alec-radford]] (GPT-2, which adopted a byte-level BPE), [[jimmy-lei-ba]] (LayerNorm, which RMSNorm simplifies), [[andrej-karpathy]] (minbpe, a from-scratch BPE implementation)
- Appears in sources: [[llm-ch02-working-with-text-data]], [[llm-ch04-implementing-a-gpt-model]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 2 references, book p. 290 (Sennrich et al. 2015 cited for an in-depth discussion of BPE).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 4 references, book pp. 292–293 (RMSNorm description; Zhang and Sennrich 2019).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 33–35 (BPE in GPT-2/GPT-3/ChatGPT, vocabulary 50,257, Figure 2.11 "Akwirw ier" example, iterative merging with a frequency cutoff).
- Biography, the merge formula in LaTeX, and the RMSNorm formula are general knowledge from the original papers, not printed in the book.
