---
title: Dan Hendrycks
type: person
tags: [deep-learning, activation-functions, evaluation, ai-safety]
sources: [llm-ch04-implementing-a-gpt-model, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Dan Hendrycks (active 2010s–present)

> **Why they matter to ML:** he co-invented **GELU**, the smooth activation function inside every GPT-2 feed-forward layer, and he led the **MMLU** benchmark that the book lists for testing an LLM's general knowledge.

> ⚠️ **Spelling:** the book's appendix B writes the GELU author as "Hendricks" (book p. 293). The correct spelling is **Hendrycks**.

## Life (brief)
Dan Hendrycks did his PhD in computer science at UC Berkeley. He wrote the GELU paper with Kevin Gimpel while a student, and became known for work on robustness, out-of-distribution detection and benchmarks. He later founded and directs the Center for AI Safety. (These biographical details are general knowledge. The book names him only in one appendix B citation, and links MMLU without naming its authors.)

## Key contributions
- **GELU, the Gaussian Error Linear Unit** (Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", 2016, arXiv 1606.08415). The book says GELU "combines the properties of both the classic ReLU activation function and the normal distribution's cumulative distribution function to model layer outputs, allowing for stochastic regularization and nonlinearities" (book p. 293). → [[gelu]], [[activation-function]]
  - **Exact formula** (book p. 105):
    $$\mathrm{GELU}(x) = x\cdot\Phi(x)$$
    Symbol by symbol: $x$ = one input number (GELU acts on each entry of a tensor separately); $\Phi(x)$ = the [[cumulative-distribution-function]] of the standard [[gaussian-distribution]], i.e. the probability that a random $Z \sim \mathcal{N}(0,1)$ is at most $x$. So GELU keeps $x$ "in proportion to how likely it is to be bigger than random noise": big positive $x$ passes almost untouched ($\Phi\approx1$), big negative $x$ is squashed to about 0 ($\Phi\approx0$). It's a soft version of [[relu]]'s hard gate.
  - **The approximation GPT-2 was trained with** (book p. 105, found by curve fitting):
    $$\mathrm{GELU}(x) \approx 0.5\,x\left(1+\tanh\!\left[\sqrt{2/\pi}\,\left(x+0.044715\,x^3\right)\right]\right)$$
    Symbol by symbol: $\tanh$ = hyperbolic tangent, an S-shaped curve from $-1$ to $1$; $0.5(1+\tanh[\cdot])$ rescales it to run from 0 to 1, so it imitates $\Phi(x)$; $\sqrt{2/\pi}$ and $0.044715$ = constants chosen to make the fit close.
    ```python
    class GELU(nn.Module):
        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            ))
    ```
  - **What the book shows (Fig. 4.8, book p. 106):** on `torch.linspace(-3, 3, 100)`, GELU is smooth and has a non-zero gradient "for almost all negative values (except at approximately x = –0.75)", the bottom of its dip. (The dip depth, about $-0.17$, is our calculation, not printed in the book.) ReLU is flat at 0 with a sharp corner. The book argues the smoothness can give "better optimization properties during training." So GELU is **not** a [[monotonic-function]].
  - **Where it lives:** the GPT-2 [[feed-forward-network]] expands 768 → 3,072 dims, applies GELU, then projects back to 768 (book §4.3). → [[transformer-block]], [[gpt-2]]
- **MMLU benchmark** (Hendrycks et al., "Measuring Massive Multitask Language Understanding", 2020, arXiv 2009.03300). The book lists MMLU as an example of "short-answer and multiple-choice benchmarks ... which test the general knowledge of a model" (book p. 236), without naming the authors. → [[llm-evaluation]]
- Also known for (general knowledge): the softmax-confidence baseline for out-of-distribution detection (with Gimpel, 2016) and robustness benchmarks such as ImageNet-C.

## Connections
- Concepts: [[gelu]], [[activation-function]], [[relu]], [[swiglu]], [[cumulative-distribution-function]], [[gaussian-distribution]], [[feed-forward-network]], [[monotonic-function]], [[llm-evaluation]]
- People: [[alec-radford]] (GPT-2, which adopted the tanh-approximated GELU), [[jimmy-lei-ba]] (LayerNorm, the other normalization-era building block in the same chapter)
- Appears in sources: [[llm-ch04-implementing-a-gpt-model]], [[llm-ch07-fine-tuning-to-follow-instructions]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 4 references, book p. 293 ("Gaussian Error Linear Units (GELUs)" (2016) by "Hendricks" and Gimpel, arXiv 1606.08415, with the quoted description).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book pp. 105–106 (exact GELU, the tanh approximation, Listing 4.3, Fig. 4.8 GELU vs. ReLU).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.7, book p. 236 (MMLU listed as a general-knowledge benchmark, arXiv 2009.03300).
- Biography, the MMLU authorship and the other contributions are general knowledge, not printed in the book.
