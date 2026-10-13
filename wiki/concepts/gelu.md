---
title: GELU (Gaussian Error Linear Unit)
type: concept
tags: [deep-learning, activation-function, llm, gpt, gaussian]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# GELU (Gaussian Error Linear Unit)

> **One-liner:** GELU is a smooth version of ReLU: instead of a hard "keep or zero" switch, it keeps each input in proportion to how likely a bell-curve number is to be smaller than it.

*Also called:* GELU activation (`gelu-activation`).

## 🟢 Intuition (Dummy)
ReLU is a light switch: a negative number gets turned off (0), a positive number passes through
untouched. It works, but the switch has a sharp "click" at zero, and once a number is negative the
switch gives zero feedback — nothing to learn from.

GELU is a **dimmer** instead of a switch. Big positive numbers pass almost fully. Big negative numbers
are dimmed almost to zero. Numbers near zero get *partly* through. How much gets through? GELU asks:
"if I drew a random number from a bell curve, how likely is it to be smaller than you?" For a big number
that chance is close to 100%, so it passes; for a very negative number it's close to 0%, so it's dimmed.

The result looks like ReLU from far away, but it's smooth, and it even dips a tiny bit below zero for
small negative inputs. That smoothness is why GPT-2 (and the book's GPT) uses GELU.

## 🟡 Mechanics (Practitioner)
**Exact definition (book §4.3, p. 105):**
$$\mathrm{GELU}(x) = x\cdot\Phi(x)$$
- $x$ = one input number (GELU is applied to each entry of a tensor separately),
- $\Phi(x)$ = the [[cumulative-distribution-function]] of the standard [[gaussian-distribution]]
  $\mathcal N(0,1)$, i.e. $\Phi(x)=P(Z\le x)$ for a standard normal $Z$ — a number between 0 and 1,
- $x\cdot\Phi(x)$ = "the input, scaled by the probability that a standard normal is below it."

A standard identity (not in the book) explains the name: $\Phi(x)=\tfrac12\big[1+\operatorname{erf}(x/\sqrt2)\big]$,
where $\operatorname{erf}$ is the Gaussian **error** function.

**The tanh approximation GPT-2 was trained with (book p. 105):**
$$\mathrm{GELU}(x)\approx 0.5\cdot x\cdot\left(1+\tanh\!\left[\sqrt{\tfrac{2}{\pi}}\cdot\left(x+0.044715\cdot x^{3}\right)\right]\right)$$
Symbol by symbol:
- $0.5\,(1+\tanh[\dots])$ = a stand-in for $\Phi(x)$: $\tanh$ outputs values in $(-1,1)$, so adding 1 and halving maps them into $(0,1)$,
- $\tanh$ = hyperbolic tangent, an S-shaped curve,
- $\sqrt{2/\pi}\approx 0.7979$ = a scale constant,
- $0.044715\cdot x^3$ = a small cubic correction, "found via curve fitting" (the book's words),
- the leading $x$ = the same "input × gate" structure as the exact version.

The book says this version is "computationally cheaper" and that "the original GPT-2 model was also
trained with this approximation."

**How close is it?** (computed, not in the book)

| $x$ | $-3$ | $-1$ | $-0.75$ | $0$ | $0.5$ | $1$ | $2$ | $3$ |
|---|---|---|---|---|---|---|---|---|
| exact $x\Phi(x)$ | −0.0040 | −0.1587 | −0.1700 | 0 | 0.3457 | 0.8413 | 1.9545 | 2.9960 |
| tanh approx. | −0.0036 | −0.1588 | −0.1700 | 0 | 0.3457 | 0.8412 | 1.9546 | 2.9964 |

The maximum gap on $[-6,6]$ is about $0.0005$.

**Listing 4.3 — GELU as a PyTorch module (book p. 105):**
```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```
No learnable parameters; output shape = input shape.

**Plotting GELU vs. ReLU (book p. 106, Fig. 4.8):**
```python
gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)     # 100 sample points in [-3, 3]
y_gelu, y_relu = gelu(x), relu(x)
```
What the figure shows:
- [[relu]] is piecewise linear: flat at 0 for $x<0$, a sharp corner at $x=0$, then the line $y=x$.
- GELU is smooth, approximates ReLU, and has a **non-zero gradient for almost all negative values** —
  except at about $x=-0.75$, where the curve bottoms out (minimum ≈ $-0.17$).

**Why it helps (book p. 106):** smoothness "can lead to better optimization properties during training,
allowing for more nuanced adjustments to the model's parameters." ReLU's corner "can sometimes make
optimization harder, especially in networks that are very deep or have complex architectures." And
because GELU gives a small non-zero output for negative inputs, "neurons that receive negative input can
still contribute to the learning process."
(The book's text says "figure 4.18, right" — a typo for figure 4.8.)

**Where GELU lives in GPT.** In the [[feed-forward-network]] of every [[transformer-block]]:
`Linear(768 → 3072) → GELU → Linear(3072 → 768)` for GPT-2 124M (book p. 107). In §4.4 the book also
builds a toy 5-layer net where every layer is `nn.Sequential(nn.Linear(...), GELU())` to study
[[shortcut-connections]] (book p. 111). When loading OpenAI's GPT-2 weights (§5.5, p. 166), `ff.layers[1]`
is the GELU and receives no weights, because it has none.

## 🔴 Theory (Expert)
- **Derivative.** By the product rule,
  $$\frac{d}{dx}\mathrm{GELU}(x)=\Phi(x)+x\,\varphi(x),\qquad \varphi(x)=\tfrac{1}{\sqrt{2\pi}}e^{-x^2/2}$$
  ($\varphi$ = the standard normal pdf, since $\Phi'=\varphi$). Setting it to zero gives the unique
  stationary point $x^{\ast}\approx-0.7518$, with $\mathrm{GELU}(x^{\ast})\approx-0.1700$ — matching the book's
  "approximately $x=-0.75$." GELU is therefore **non-monotonic** and **non-convex**, unlike ReLU.
- **Limits.** $\mathrm{GELU}(x)\to x$ as $x\to+\infty$ and $\to 0^-$ as $x\to-\infty$ (because $\Phi(x)$ shrinks
  faster than $|x|$ grows), so it is asymptotically ReLU.
- **Probabilistic origin (Hendrycks & Gimpel 2016, cited in the book's appendix B, p. 293).** Consider a
  stochastic gate $m\sim\mathrm{Bernoulli}(\Phi(x))$ and output $m\cdot x$. Its expectation is
  $\mathbb E[m x]=x\,\Phi(x)$ — exactly GELU. So GELU is the *deterministic expected value* of an
  input-dependent, dropout-like mask: inputs that are larger (relative to $\mathcal N(0,1)$) are more likely
  to be kept. Appendix B summarizes it as combining ReLU-like nonlinearity with the normal CDF and giving a
  form of stochastic regularization. See [[expected-value-and-mean]], [[dropout]].
- **Why the tanh form works.** Using $\tfrac12(1+\tanh z)=\mathrm{sigmoid}(2z)$, the approximation is
  $x\cdot\mathrm{sigmoid}\!\big(2\sqrt{2/\pi}\,(x+0.044715x^3)\big)$. The constant $\sqrt{2/\pi}$ matches the
  slope of $\Phi$ at 0 ($\varphi(0)=1/\sqrt{2\pi}$, and $\tfrac{d}{dx}\tfrac12(1+\tanh(cx))|_0=c/2$, so
  $c=\sqrt{2/\pi}$); the cubic term fixes the tails. A cruder variant is $x\,\sigma(1.702x)$, which links
  GELU to Swish/SiLU and to [[swiglu]] gating.
- **Normalization interplay.** Because [[layer-normalization]] (applied before the FFN in Pre-LN GPT)
  keeps pre-activations roughly zero-mean/unit-variance, $\Phi$ being the *standard* normal CDF is a
  natural scale: inputs of a typical magnitude land in the curved region of the gate.
- **Errata.** The book's chapter 4 references spell the first author "Hendricks"; the correct spelling is
  **Hendrycks** (Dan Hendrycks).

## Connections
- Builds on: [[activation-function]], [[relu]], [[cumulative-distribution-function]], [[gaussian-distribution]], [[derivative]]
- Leads to: [[swiglu]], [[feed-forward-network]]
- Used by: [[transformer-block]], [[gpt-2]], [[gpt-architecture]], [[shortcut-connections]] (§4.4 experiment)
- Related: [[sigmoid-function]], [[dropout]], [[expected-value-and-mean]], [[taylor-series]] (curve-fitted polynomial inside tanh), [[vanishing-and-exploding-gradients]]
- People: [[dan-hendrycks]], [[alec-radford]] (GPT-2)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book p. 104 (roadmap: GELU replaces ReLU in LLMs).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book pp. 105–108 (exact and tanh GELU, Listing 4.3, Fig. 4.8, FFN use).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4, book pp. 109–113 (Linear + GELU deep network for the shortcut experiment).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5–4.7, book pp. 113–117, 121, 127 (GELU inside the transformer block / GPT model; ch. 4 summary: FFNs "that use the GELU activation function").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book p. 166 (GELU layer has no weights to load).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book p. 293 (Hendrycks & Gimpel 2016, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415).
