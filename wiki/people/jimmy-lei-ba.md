---
title: Jimmy Lei Ba
type: person
tags: [deep-learning, normalization, optimization]
sources: [llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Jimmy Lei Ba (active 2010s–present)

> **Why they matter to ML:** he is the first author of **Layer Normalization**, the layer that keeps every GPT-2 transformer block numerically steady, and a co-author of **Adam**, the optimizer family (as AdamW) that trains the book's LLM.

## Life (brief)
Jimmy Lei Ba did his PhD at the University of Toronto in [[geoffrey-hinton]]'s group, where he wrote the LayerNorm paper with Jamie Ryan Kiros and Hinton. He later joined the University of Toronto faculty, and he was among the founding team of the AI company xAI in 2023. (These biographical details are general knowledge. The book names him only as "Ba" in one appendix B citation.)

## Key contributions
- **Layer Normalization** (Ba, Kiros & Hinton, "Layer Normalization", 2016, arXiv 1607.06450). The book says the paper shows that LayerNorm "stabilizes the hidden state dynamics [of] neural networks by normalizing the summed inputs to the neurons within a hidden layer, significantly reducing training time compared to previously published methods" (book p. 292). Today it's simply called **LayerNorm**. → [[layer-normalization]]
  - **The formula** (the book gives it in code, Listing 4.2; here in LaTeX):
    $$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i,\qquad \sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2,\qquad y_i = \gamma_i\,\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}} + \beta_i$$
    Symbol by symbol: $x_i$ = the $i$-th activation of one token's vector; $n$ = the vector's length (`emb_dim`, 768 in GPT-2 124M); $\mu$ = the average ([[expected-value-and-mean]]); $\sigma^2$ = the variance, divided by $n$ not $n-1$ (the book's `unbiased=False`, see [[bessels-correction-biased-variance]]); $\varepsilon$ = `eps = 1e-5`, guards against dividing by zero; $\gamma_i$ = `scale` (starts at 1) and $\beta_i$ = `shift` (starts at 0), both learned; $y_i$ = the normalized output.
  - **The book's numbers (book pp. 100–104):** a `Linear(5, 6) + ReLU` layer gives row 1 `[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000]` (mean 0.1324, variance 0.0231; this first manual example uses PyTorch's default *unbiased* `var`, dividing by $n-1$). After normalization it becomes `[0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719]`, mean ≈ 0, variance 1. The `LayerNorm` class on the same data prints variances `0.9995` and `0.9997`; they're a hair under 1 because of $\varepsilon$.
    ```python
    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))
        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift
    ```
  - **Why it won for transformers:** it normalizes across the features of *one* token, not across the batch, so it works the same for any batch size (book pp. 104–105). → [[batch-normalization]]
  - **What came after:** [[pre-layernorm-vs-post-layernorm]] (GPT-2 puts the norm *before* attention and the feed-forward net) and [[rmsnorm]] by [[biao-zhang]] and [[rico-sennrich]], which drops the mean and the shift.
- **Adam optimizer** (Kingma & Ba, "Adam: A Method for Stochastic Optimization", 2014). Adam keeps running averages of each parameter's gradient and squared gradient to give every weight its own step size. The book trains its GPT with **AdamW**, the version with decoupled [[weight-decay]] (general knowledge; the book doesn't cite the Adam paper itself). → [[adamw-optimizer]], [[stochastic-gradient-descent]]

## Connections
- Concepts: [[layer-normalization]], [[rmsnorm]], [[batch-normalization]], [[pre-layernorm-vs-post-layernorm]], [[transformer-block]], [[standardization-z-score]], [[adamw-optimizer]], [[vanishing-and-exploding-gradients]]
- People: [[geoffrey-hinton]] (co-author and PhD advisor), [[biao-zhang]] and [[rico-sennrich]] (RMSNorm, which simplifies LayerNorm)
- Appears in sources: [[llm-ch04-implementing-a-gpt-model]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 4 references, book p. 292 ("Layer Normalization" (2016) by Ba, Kiros, and Hinton, arXiv 1607.06450, with the quoted summary).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 99–105 (LayerNorm motivation, worked numbers, Listing 4.2, biased-variance sidebar, LayerNorm vs. batch normalization).
- Biography, the full author name, the Adam paper and the LaTeX formula are general knowledge, not printed in the book.
