---
title: SwiGLU Activation
type: concept
tags: [deep-learning, activation-function, llm, gating]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# SwiGLU Activation

> **One-liner:** SwiGLU splits a layer's output into two branches and lets one branch, softened by a sigmoid-based "Swish" curve, decide how much of the other branch gets through.

*Also called:* SwiGLU activation, Swish-gated linear unit (`swiglu-activation`).

> **Scope note.** The book only *names* SwiGLU (§4.3, p. 105) as one of two notable LLM activation
> functions beside [[gelu]], describing it as a "more complex and smooth" function built on
> "sigmoid-gated linear units" that offers "improved performance" over ReLU. It is **not implemented**
> in the book — the book's GPT-2 uses GELU. Everything below the 🟢 tier marked *(standard)* comes from
> the wider literature (Shazeer 2020, "GLU Variants Improve Transformer"), not from the book.

## 🟢 Intuition (Dummy)
Think of a water pipe with a tap. [[relu]] and [[gelu]] are taps that look at the water itself to decide
how much to let through. SwiGLU uses **two pipes**: one carries the water (the information), and the
other controls the tap (the gate). Because the gate is its own learned pipe, the network can learn
*separately* "what to say" and "how loudly to say it." That extra flexibility tends to make large
language models a bit better for the same amount of compute, which is why many modern LLMs (e.g.
Llama-family models) use it instead of GELU.

## 🟡 Mechanics (Practitioner)
**What the book says (p. 105):** "GELU and SwiGLU are more complex and smooth activation functions
incorporating Gaussian and sigmoid-gated linear units, respectively." So: GELU ↔ Gaussian, SwiGLU ↔
sigmoid gating.

**Building blocks (standard):**

1. The [[sigmoid-function]]: $\sigma(z)=\dfrac{1}{1+e^{-z}}$ — squashes $z$ into $(0,1)$.
2. **Swish** (also called SiLU when $\beta=1$):
   $$\mathrm{Swish}_\beta(z)=z\cdot\sigma(\beta z)$$
   - $z$ = the input number, $\sigma(\beta z)$ = a soft gate between 0 and 1, $\beta$ = a sharpness
     constant (usually 1). Like GELU, it is "input × soft gate" — just with a sigmoid gate instead of the
     Gaussian CDF.
3. **GLU (gated linear unit):** take two linear projections of the same input and multiply them
   elementwise, one of them passed through a gate.

**SwiGLU (standard):**
$$\mathrm{SwiGLU}(\mathbf x)=\mathrm{Swish}(W\mathbf x)\odot(V\mathbf x)$$
- $\mathbf x$ = a token's vector (e.g. 768 numbers),
- $W, V$ = two separate learned weight matrices (the "gate" branch and the "value" branch),
- $\mathrm{Swish}(W\mathbf x)$ = the gate, computed entry by entry,
- $\odot$ = elementwise (Hadamard) product — multiply matching entries.

**SwiGLU feed-forward network (standard),** replacing the book's `Linear → GELU → Linear`:
$$\mathrm{FFN}_{\text{SwiGLU}}(\mathbf x)=W_2\big(\mathrm{Swish}(W_1\mathbf x)\odot V\mathbf x\big)$$
- $W_1, V\in\mathbb R^{h\times d}$ = two "up" projections, $W_2\in\mathbb R^{d\times h}$ = the "down" projection,
- $d$ = embedding size, $h$ = hidden size (biases are usually dropped).

**Compare with the book's FFN** ([[feed-forward-network]], §4.3): $W_2\,\mathrm{GELU}(W_1\mathbf x+\mathbf b_1)+\mathbf b_2$
uses **two** matrices; SwiGLU uses **three**. To keep the parameter count equal, $h$ is often set to
about $\tfrac{8}{3}d$ instead of $4d$. For $d=768$: GELU FFN ≈ $2\cdot768\cdot3072\approx 4.72$M weights; a
SwiGLU FFN with $h=2048$ has $3\cdot768\cdot2048\approx 4.72$M — the same budget.

**PyTorch sketch (standard, not from the book):**
```python
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)   # gate branch
        self.v  = nn.Linear(d, h, bias=False)   # value branch
        self.w2 = nn.Linear(h, d, bias=False)   # down projection
    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.v(x))
```
Shapes work exactly like the book's FFN: `(batch, tokens, d) → (batch, tokens, h) → (batch, tokens, d)`.

## 🔴 Theory (Expert)
- **GLU family (Dauphin et al. 2017; Shazeer 2020).** $\mathrm{GLU}(\mathbf x)=\sigma(W\mathbf x)\odot V\mathbf x$.
  Variants swap the gate nonlinearity: ReGLU (ReLU), GEGLU (GELU), SwiGLU (Swish). Shazeer reported lower
  perplexity for GEGLU/SwiGLU FFNs in T5-style transformers at matched parameters and FLOPs.
- **Multiplicative interaction.** Unlike $\phi(W\mathbf x)$, the product $\phi(W\mathbf x)\odot V\mathbf x$ is
  a *second-order* (bilinear-like) function of $\mathbf x$. Each hidden unit can express products of two
  linear features, giving richer function classes per parameter.
- **Gradient behavior.** $\frac{d}{dz}\,z\sigma(z)=\sigma(z)+z\sigma(z)(1-\sigma(z))$ — smooth and non-zero
  for almost all $z$, with a single minimum near $z\approx-1.28$ (value ≈ $-0.28$), the Swish analogue of
  GELU's dip at $-0.75$. The gate branch also passes gradient to the value branch linearly, which helps
  optimization.
- **Relation to GELU.** GELU's tanh approximation can be rewritten as $x\,\sigma\big(2\sqrt{2/\pi}(x+0.044715x^3)\big)$
  ≈ $x\,\sigma(1.702x)$ = $\mathrm{Swish}_{1.702}(x)$. So GELU ≈ Swish with $\beta\approx1.702$; SwiGLU's
  novelty is the *separate* gate projection, not the curve itself.
- **Bookkeeping in LLMs.** Because it adds a third matrix, a "4× expansion" from the GPT-2 recipe
  ([[gpt-model-configuration]]) does not transfer directly; the $8/3$ rule (often rounded to a hardware-
  friendly multiple) preserves [[parameter-count-and-memory]].

## Connections
- Builds on: [[activation-function]], [[sigmoid-function]], [[gelu]], [[linear-layer]], [[matrix-multiplication]]
- Leads to: [[feed-forward-network]] (SwiGLU variant)
- Used by: modern decoder-only LLMs (e.g. Llama-style models) — see [[decoder-only-transformer]], [[transformer-block]]
- Related: [[relu]], [[parameter-count-and-memory]], [[rmsnorm]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book p. 105 (SwiGLU named as a smooth, sigmoid-gated LLM activation; not implemented).
- Outside the book (standard references): N. Shazeer (2020), "GLU Variants Improve Transformer," arXiv:2002.05202; Y. Dauphin et al. (2017), "Language Modeling with Gated Convolutional Networks."
