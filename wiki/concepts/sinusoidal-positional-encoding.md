---
title: Sinusoidal Positional Encoding
type: concept
tags: [transformer, embeddings, trigonometry, linear-algebra]
difficulty: advanced
sources: [llm-ch02-working-with-text-data]
created: 2026-10-13
updated: 2026-10-13
---

# Sinusoidal Positional Encoding

> **One-liner:** The original transformer wrote each word's position as a pattern of sine and cosine waves, a fixed formula instead of something learned.

## 🟢 Intuition (Dummy)
Think of a clock with many hands. The second hand spins fast, the minute hand
slower, the hour hand slower still. If you take a photo of all the hands, you can
read off the exact time, and no two moments in the day look the same.

Sinusoidal encoding gives each position in a sentence a "clock photo" like that.
Some coordinates wiggle fast as you move from word to word, others crawl very
slowly. Put them all together and every position gets its own unique pattern, and
nothing has to be learned — it's just a formula.

What the book says: OpenAI's GPT models use learned
[[absolute-positional-embedding|absolute positional embeddings]] "rather than being
fixed or predefined like the positional encodings in the original transformer
model" (book p. 45). That's the only mention; the book doesn't give the formula. The
formula and everything after it on this page come from the original transformer
paper (Vaswani et al., 2017), not from the LLM book.

## 🟡 Mechanics (Practitioner)

### The formula (Vaswani et al., 2017 — not in the book)
For position $t$ and coordinate pair $i = 0, 1, \dots, d/2 - 1$:

$$PE_{(t,\,2i)} = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \qquad PE_{(t,\,2i+1)} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

Symbol by symbol:
- $PE$ = the positional-encoding table; $PE_{(t,j)}$ = entry in row $t$, column $j$.
- $t$ = the position (0, 1, 2, …).
- $2i$ and $2i+1$ = an even column and the odd column right after it. They share one frequency: the even one gets sine, the odd one cosine.
- $d$ = the [[embedding-dimension]] (must match the token embeddings, because they get added).
- $10000^{2i/d}$ = how "slow" that pair is. For $i=0$ it's $10000^0 = 1$ (fastest); for the last pair it's almost $10000$ (slowest).
- It's convenient to write $\omega_i = 1/10000^{2i/d}$ (the angular frequency), so the pair is $(\sin \omega_i t,\ \cos \omega_i t)$.

Then, exactly like in GPT, the encoding is **added** to the token vector:
$\mathbf{x}_t = E_{\text{tok}}[\text{id}_t] + PE_t$ (see [[input-embeddings]]).

### Worked example, $d = 4$
Two pairs: $\omega_0 = 1$, $\omega_1 = 1/10000^{2/4} = 1/100 = 0.01$.

| $t$ | $\sin(t)$ | $\cos(t)$ | $\sin(0.01t)$ | $\cos(0.01t)$ |
|---|---|---|---|---|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 1.0000 |
| 2 | 0.9093 | −0.4161 | 0.0200 | 0.9998 |
| 3 | 0.1411 | −0.9900 | 0.0300 | 0.9996 |

The first pair (fast hand) changes a lot each step; the second (slow hand) barely
moves. Over thousands of positions, the slow pair is what keeps far-apart positions
from colliding.

### A PyTorch sketch (ours, not the book's)
```python
import torch
def sinusoidal_pe(L, d):
    t = torch.arange(L).unsqueeze(1)                 # [L, 1]
    omega = 1.0 / 10000 ** (torch.arange(0, d, 2) / d)   # [d/2]
    pe = torch.zeros(L, d)
    pe[:, 0::2] = torch.sin(t * omega)               # [L,1]*[d/2] -> [L,d/2] by broadcasting
    pe[:, 1::2] = torch.cos(t * omega)
    return pe
```
Unlike `nn.Embedding(context_length, d)`, this tensor has **no trainable
parameters**. Note the [[broadcasting]] in `t * omega`.

## 🔴 Theory (Expert)
- **Shifts are rotations.** For one frequency $\omega$, the angle-addition formulas
  $\sin(a+b)=\sin a\cos b+\cos a\sin b$ and $\cos(a+b)=\cos a\cos b-\sin a\sin b$ give
  $$\begin{pmatrix}\sin\omega(t+k)\\ \cos\omega(t+k)\end{pmatrix}=\begin{pmatrix}\cos\omega k & \sin\omega k\\ -\sin\omega k & \cos\omega k\end{pmatrix}\begin{pmatrix}\sin\omega t\\ \cos\omega t\end{pmatrix}.$$
  So $PE_{t+k} = M_k\,PE_t$, where $M_k$ is block-diagonal with 2×2
  [[rotation]] blocks and depends on $k$ only. Moving by $k$ positions is a
  [[linear-mapping|linear map]] — this was the paper's stated motivation (the model
  "could easily learn to attend by relative positions").
- **Dot products depend only on distance.** Using $\sin a\sin b+\cos a\cos b=\cos(a-b)$:
  $$PE_t \cdot PE_{t+k} = \sum_{i=0}^{d/2-1}\cos(\omega_i k),$$
  independent of $t$ and symmetric in $\pm k$. Each block is [[orthogonal-matrix|orthogonal]],
  so $\lVert PE_t\rVert^2 = d/2$ for every $t$ (constant [[norm]]). Check on the table above:
  $PE_1\cdot PE_1 = 0.8415^2+0.5403^2+0.0100^2+1.0000^2 \approx 2 = d/2$.
- **Geometric spectrum.** Wavelengths $2\pi/\omega_i = 2\pi\cdot 10000^{2i/d}$ run from
  $2\pi$ up to almost $2\pi\cdot 10000$, a geometric progression. Fast pairs resolve
  neighbours; slow pairs separate distant positions. It's like a multi-scale Fourier
  feature map of $t$.
- **Defined for any $t$.** The formula has no length table, so it is defined beyond
  the training length (though models do not automatically *perform* well there).
  A learned table stops at row $L-1$ (book p. 47, truncation).
- **Absolute in form, relative-friendly in effect.** It is an absolute code (one vector
  per $t$), but its rotation structure is the bridge to
  [[relative-positional-embedding|relative]] methods such as RoPE, which apply the same
  rotations to queries and keys.

## Connections
- Builds on: [[positional-embedding]], [[absolute-positional-embedding]], [[vector]], [[rotation]], [[inner-product]]
- Contrast: learned [[absolute-positional-embedding]] (GPT), [[relative-positional-embedding]]
- Leads to: [[transformer-architecture]], [[encoder-decoder-architecture]]
- Uses: [[broadcasting]], [[orthogonal-matrix]], [[norm]]
- People: [[ashish-vaswani]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book p. 45 (only reference: GPT's learned embeddings vs "fixed or predefined" encodings of the original transformer; no formula given).
- Formula, motivation, and properties: Vaswani et al. (2017), "Attention Is All You Need", §3.5 — not in `raw/`. Worked numbers and identities computed for this page.
