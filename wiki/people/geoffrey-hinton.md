---
title: Geoffrey Hinton
type: person
tags: [deep-learning, neural-networks, normalization, regularization, backpropagation]
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Geoffrey Hinton (b. 1947)

> **Why they matter to ML:** he is one of the "godfathers of deep learning." He helped make [[backpropagation]] popular, and his group gave GPT-style LLMs two of their standard parts: **dropout** and **layer normalization**.

## Life (brief)
Geoffrey Hinton is a British-Canadian computer scientist. He spent most of his career at the University of Toronto and also worked at Google from 2013 to 2023. He shared the 2018 Turing Award with [[yoshua-bengio]] and Yann LeCun for deep learning, and the 2024 Nobel Prize in Physics with John Hopfield. (These biographical details are general knowledge. The book names him only in two appendix B citations: as a co-author of LayerNorm, and as one of the "et al." on the dropout paper.)

## Key contributions
- **Backpropagation for neural networks** (Rumelhart, Hinton & Williams, 1986). This paper showed that you can train multi-layer networks by sending the error backward with the [[chain-rule]]. Every model in the LLM book trains this way, through PyTorch autograd. → [[backpropagation]], [[automatic-differentiation]], [[gradient-descent]]
- **Dropout** (Srivastava, Hinton et al., 2014; first proposed by Hinton's group in 2012), led by his student [[nitish-srivastava]]. The idea: randomly ignore some hidden units during training, so the network can't depend on any one set of them. The book uses it in the [[attention-weights]] (rate 0.5 in the demo) and sets `drop_rate = 0.1` for GPT-2 (book pp. 78–80, 292). See [[nitish-srivastava]] for the formula and the book's numbers. → [[dropout]], [[regularization]], [[overfitting]]
- **Layer Normalization** (Ba, Kiros & Hinton, "Layer Normalization", 2016, arXiv 1607.06450), with [[jimmy-lei-ba]]. The book summarizes it as a technique that "stabilizes the hidden state dynamics [of] neural networks by normalizing the summed inputs to the neurons within a hidden layer, significantly reducing training time" (book p. 292). Today it is called **LayerNorm**. GPT-2 puts one before the attention, one before the feed-forward net in every [[transformer-block]], and one more before the output layer. → [[layer-normalization]]
  - **The problem it fixes (book p. 99):** in deep networks the gradients can get very small or very large ([[vanishing-and-exploding-gradients]]), so training becomes unstable. LayerNorm rescales each layer's outputs so they have **mean 0 and variance 1**.
  - **The formula** (the book gives it in words and code; here it is in LaTeX):
    $$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i,\qquad \sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2,\qquad y_i = \gamma_i\,\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}} + \beta_i$$
    Symbol by symbol: $\mathbf{x} = (x_1,\dots,x_n)$ = one token's vector of activations; $n$ = how many numbers it has (`emb_dim`, 768 for GPT-2 124M); $\sum_{i=1}^{n}$ = "add up over all positions $i$"; $\mu$ = their average ([[expected-value-and-mean]]); $\sigma^2$ = their variance, the average squared distance from $\mu$ ([[covariance-and-variance]]); $\sqrt{\sigma^2}$ = the standard deviation; $\varepsilon$ = `eps = 1e-5`, a tiny number so we never divide by zero; $\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}$ = the [[standardization-z-score]] of entry $i$; $\gamma_i$ (`scale`, starts at 1) and $\beta_i$ (`shift`, starts at 0) = learned numbers that let the model undo the normalization if that helps; $y_i$ = the output.
  - **The book's worked numbers (book pp. 100–102):** a `Linear(5, 6) + ReLU` layer on a seeded random batch gives row 1 = `[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000]`, with mean $0.1324$ and variance $0.0231$ (row 2: mean $0.2170$, variance $0.0398$). In this first demo the book calls `out.var(...)` with PyTorch's default, which divides by $n-1$ (here $0.1154/5 \approx 0.0231$), and it divides by `torch.sqrt(var)` with no `eps`. The normalized row 1 is `[0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719]`, with mean $\approx -5.96\times10^{-8}$ (zero up to rounding) and variance $1$. For example $(0.3470 - 0.1324)/\sqrt{0.0231} \approx 1.41$.
    ```python
    mean = out.mean(dim=-1, keepdim=True)
    var  = out.var(dim=-1, keepdim=True)      # default: divides by n-1
    out_norm = (out - mean) / torch.sqrt(var)
    ```
    The book's reusable `LayerNorm` class (book p. 103) is different in two ways: it adds `eps = 1e-5`, and it uses `unbiased=False` (divide by $n$, not $n-1$). This matches GPT-2 and TensorFlow's default, and it lets the pretrained GPT-2 weights load in chapter 6. For a large `emb_dim` the difference between $n$ and $n-1$ is negligible. → [[bessels-correction-biased-variance]]
  - **LayerNorm vs. BatchNorm:** LayerNorm normalizes across the features of *one* example. Batch normalization normalizes across the *batch*. That is why LayerNorm doesn't care about batch size, which makes it a good fit for LLMs (book p. 105). → [[batch-normalization]]
  - **Later work:** [[pre-layernorm-vs-post-layernorm]] (where the norm goes), and [[rmsnorm]] by [[biao-zhang]] and [[rico-sennrich]], which drops the mean and the shift.

## Connections
- Concepts: [[backpropagation]], [[dropout]], [[layer-normalization]], [[batch-normalization]], [[rmsnorm]], [[regularization]], [[overfitting]], [[vanishing-and-exploding-gradients]], [[standardization-z-score]], [[transformer-block]], [[neural-network]], [[deep-learning]]
- People: [[nitish-srivastava]] (his student, first author of Dropout), [[jimmy-lei-ba]] (first author of LayerNorm), [[yoshua-bengio]] (Turing Award co-recipient)
- Appears in sources: [[llm-ch03-coding-attention-mechanisms]] (dropout), [[llm-ch04-implementing-a-gpt-model]] (LayerNorm)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 3 and Ch. 4 references, book p. 292 (Srivastava et al. 2014 Dropout; Ba, Kiros & Hinton 2016 Layer Normalization, arXiv 1607.06450).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5.2, book pp. 78–80 (dropout in attention).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 99–105 (LayerNorm motivation, mean/variance, `eps`, `scale`/`shift`, worked numbers, BatchNorm sidebar). Note: §4.2 itself does not name the LayerNorm paper. Only appendix B does.
- The biography (Turing Award, Nobel Prize, Google years), the 1986 backpropagation paper, and the LayerNorm formula in LaTeX are general knowledge. They are not printed in the book.
