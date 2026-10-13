---
title: Relative Positional Embeddings
type: concept
tags: [llm, embeddings, transformer, attention]
difficulty: advanced
sources: [llm-ch02-working-with-text-data]
created: 2026-10-13
updated: 2026-10-13
---

# Relative Positional Embeddings

> **One-liner:** Relative positional embeddings tell the model how far apart two words are ("3 steps away"), instead of which exact seat each word sits in.

## 🟢 Intuition (Dummy)
Picture giving directions. You can say "the café is at 125 Main Street" (an
*absolute* address), or "the café is two doors down from the bank" (a *relative*
direction). The second kind still works if the whole street gets renumbered.

Relative positional embeddings use the second style. The book says the model
learns relationships in terms of "how far apart" rather than "at which exact
position" (book p. 45). The payoff: the model can "generalize better to sequences
of varying lengths, even if it hasn't seen such lengths during training"
(book p. 45). A phrase like "not very good" means the same thing at the start of a
text or 900 tokens in, and a relative scheme sees it the same way in both places.

The book only describes this idea in one paragraph. The GPT models it builds use
[[absolute-positional-embedding|absolute]] embeddings instead (book p. 45). Everything
in the Mechanics tier after "What the book gives you", and most of the Theory tier, is general background, not
from the book.

## 🟡 Mechanics (Practitioner)

### What the book gives you (book p. 45)
- The thing being encoded is the relative position or **distance** between tokens,
  not each token's exact position. (Our notation: the offset $k = j - i$; the book
  gives no formula.)
- Both relative and absolute schemes aim to help the LLM understand "the order and
  relationships between tokens"; the choice "often depends on the specific
  application and the nature of the data" (book p. 45).

### How it's usually done (background, beyond the book)
Relative schemes don't add a vector to the input. They act **inside attention**,
where token $i$ compares itself with token $j$. Three well-known styles:

1. **Learned offset vectors** (Shaw et al., 2018). Add a learned vector
   $\mathbf{a}_{j-i}$ to the key when computing the score:
   $$\omega_{ij} = \frac{\mathbf{q}_i \cdot (\mathbf{k}_j + \mathbf{a}_{j-i})}{\sqrt{d_k}}$$
   - $\omega_{ij}$ = the [[attention-scores|attention score]] of query $i$ for key $j$.
   - $\mathbf{q}_i$, $\mathbf{k}_j$ = query and key vectors ([[query-key-value]]).
   - $\mathbf{a}_{j-i}$ = a learned vector that depends only on the offset $j-i$ (offsets are usually clipped to a maximum, e.g. $\pm 16$).
   - $\cdot$ = [[inner-product|dot product]]; $\sqrt{d_k}$ = the usual [[scaled-dot-product-attention|scaling]].

2. **Linear distance bias** (ALiBi, Press et al., 2021). Just subtract a penalty
   that grows with distance:
   $$\omega_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} - m\,|i-j|$$
   - $m$ = a fixed slope (different per attention head); $|i-j|$ = distance. Far-away tokens get lower scores.

3. **Rotary embeddings, RoPE** (Su et al., 2021). Rotate $\mathbf{q}$ and $\mathbf{k}$
   by an angle proportional to their position, in 2-D pairs of coordinates. For one pair
   with frequency $\theta$:
   $$\mathbf{q}_i' = R(i\theta)\,\mathbf{q}_i,\quad \mathbf{k}_j' = R(j\theta)\,\mathbf{k}_j,\qquad R(\phi)=\begin{pmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{pmatrix}$$
   - $R(\phi)$ = a 2-D [[rotation]] matrix by angle $\phi$.
   - Then $\mathbf{q}_i'\cdot\mathbf{k}_j' = \mathbf{q}_i^\top R(i\theta)^\top R(j\theta)\mathbf{k}_j = \mathbf{q}_i^\top R\big((j-i)\theta\big)\mathbf{k}_j$ — only the offset $j-i$ survives.

**Tiny worked example (RoPE, one pair, $\theta = 1$ rad).** Take
$\mathbf{q} = \mathbf{k} = (1, 0)$. At positions $(i,j) = (2,5)$ and $(10,13)$ the offset is 3
both times, so both scores equal $(1,0)\,R(3)\,(1,0)^\top = \cos 3 \approx -0.990$.
Same offset → same score, wherever the pair sits.

## 🔴 Theory (Expert)
- **Shift invariance.** A scheme is relative if every attention logit can be
  written as $f(\mathbf{x}_i, \mathbf{x}_j, j-i)$. Then shifting the whole sequence by
  $c$ leaves all logits unchanged — exactly the property behind the book's claim of
  better length generalization (book p. 45).
- **RoPE's key identity.** Rotation matrices form a group: $R(\alpha)^\top = R(-\alpha)$ and
  $R(\alpha)R(\beta)=R(\alpha+\beta)$. So $R(i\theta)^\top R(j\theta) = R((j-i)\theta)$.
  Rotations are [[orthogonal-matrix|orthogonal]], so $\lVert R\mathbf{q}\rVert = \lVert\mathbf{q}\rVert$:
  position changes direction, never [[norm|length]]. RoPE is "absolute in the input,
  relative in the score".
- **Link to sinusoids.** The [[sinusoidal-positional-encoding]] of the original
  transformer already has the property that $\mathbf{p}_{t+k}$ is a rotation of
  $\mathbf{p}_t$ per frequency pair, so dot products of pure position codes depend only
  on $k$. RoPE moves that rotation from the input into the query/key products.
- **Limits.** Clipped offset tables (Shaw) see all distances beyond the clip as equal.
  Length extrapolation of RoPE is good but not unlimited; rotation angles never seen in
  training can still hurt.
- **Why the book uses absolute anyway.** It reproduces GPT-2, whose weights include a
  learned absolute table `wpe` (book pp. 45, 165). Loading those weights requires the
  same scheme.

## Connections
- Builds on: [[positional-embedding]], [[self-attention]], [[attention-scores]], [[inner-product]], [[rotation]]
- Contrast: [[absolute-positional-embedding]]
- Related: [[sinusoidal-positional-encoding]], [[scaled-dot-product-attention]], [[query-key-value]]
- Leads to: [[long-range-dependencies]], [[context-length]]
- People: [[ashish-vaswani]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book p. 45 (definition, "how far apart", length generalization, choice depends on application; GPT uses absolute instead).
- Background beyond the book (not in `raw/`): Shaw, Uszkoreit & Vaswani (2018) "Self-Attention with Relative Position Representations"; Press, Smith & Lewis (2021) "Train Short, Test Long" (ALiBi); Su et al. (2021) "RoFormer" (RoPE).
