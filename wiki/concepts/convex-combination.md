---
title: Convex Combination (Weighted Average)
type: concept
tags: [linear-algebra, geometry, attention, probability]
difficulty: beginner
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Convex Combination (Weighted Average)

> **One-liner:** A convex combination is a weighted average where every weight is zero or positive and all the weights add up to 1.

## 🟢 Intuition (Dummy)
Think of mixing paint. You have a few pots of colour, and you make a new colour by taking 50% of
the red pot, 30% of the yellow pot and 20% of the blue pot. You can't take a *negative* amount of a
pot, and the percentages must add up to 100%. Whatever you make, it can never be "more red than
the red pot" — the result always sits somewhere *between* the colours you started with.

That "somewhere in between" rule is the whole idea. In an LLM, every token builds its new
[[context-vector]] by mixing the vectors of all the tokens it can see, with percentages that the
[[attention-mechanism]] decides. The percentages are the [[attention-weights]], they are always
positive and always sum to 1, so the new vector is a blend that lives "between" the token vectors.

## 🟡 Mechanics (Practitioner)

### Definition
$$
\mathbf z = \sum_{j=1}^{T}\alpha_j\,\mathbf x^{(j)},\qquad \alpha_j \ge 0,\qquad \sum_{j=1}^{T}\alpha_j = 1 .
$$
Symbol by symbol:
- $\mathbf x^{(1)},\dots,\mathbf x^{(T)}$ = the $T$ [[vector|vectors]] you are mixing (the superscript in brackets is just a label "vector number $j$", not a power).
- $\alpha_j$ ("alpha j") = the weight (the percentage) given to vector $j$.
- $\sum_{j=1}^{T}$ = "add up the following for $j = 1, 2, \dots, T$".
- $\alpha_j \ge 0$ = no negative weights. $\sum_j\alpha_j = 1$ = the weights make 100%.
- $\mathbf z$ = the result, the convex combination.

If you drop the "$\ge 0$" rule you get an *affine* combination (see [[affine-space]]); if you drop
both rules you get a plain *linear* combination (see [[vector-space]]).

### Where the weights come from in attention
The book shows two ways to turn raw [[attention-scores]] $\omega_{2j}$ into weights (§3.3.1, pp. 59–60):
$$
\alpha^{\text{tmp}}_{2j} = \frac{\omega_{2j}}{\sum_{t=1}^{T}\omega_{2t}}
\qquad\text{vs.}\qquad
\alpha_{2j} = \frac{\exp(\omega_{2j})}{\sum_{t=1}^{T}\exp(\omega_{2t})} .
$$
- Left: divide each score by the total. This sums to 1, but it only gives non-negative weights if all scores are non-negative.
- Right: [[softmax]]. Because $\exp(\cdot) > 0$ always, the weights are **guaranteed positive** and sum to 1 — always a valid convex combination. The book prefers it because it handles extreme values better, has better gradient behaviour, and the weights can be read as probabilities (p. 60).

### Worked example (book numbers, §3.3.1, pp. 57–60)
Inputs for "Your journey starts with one step.":
Your [0.43, 0.15, 0.89], journey [0.55, 0.87, 0.66], starts [0.57, 0.85, 0.64],
with [0.22, 0.58, 0.33], one [0.77, 0.25, 0.10], step [0.05, 0.80, 0.55].

Softmax weights for the query "journey":
`[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]` (sum = 1).

First coordinate of the context vector (our hand check):
$$0.1385(0.43) + 0.2379(0.55) + 0.2333(0.57) + 0.1240(0.22) + 0.1082(0.77) + 0.1581(0.05) \approx 0.4419,$$
matching the book's $\mathbf z^{(2)} = $ `[0.4419, 0.6515, 0.5683]` (p. 60). Notice 0.4419 sits between
the smallest first coordinate (0.05) and the largest (0.77), exactly as the "in between" rule promises.

```python
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i      # weighted sum = convex combination
```
For all tokens at once, `attn_weights @ inputs` (p. 63): each **row** of the weight matrix sums to 1
(`attn_weights.sum(dim=-1)` prints all 1.0000), so every row of the output is a convex combination.
(The wiki calls such a matrix *row-stochastic*; the book does not use that word.)

### Variants inside the LLM
- **Trainable attention** (§3.4): the mix is over **value vectors**, $\mathbf z^{(2)} = \sum_j\alpha_{2j}\mathbf v^{(j)}$, not over raw inputs (pp. 69–70). See [[scaled-dot-product-attention]].
- **[[causal-attention|Causal attention]]** (§3.5): weights for future tokens are forced to 0 and the rest renormalized, so token $i$ is a convex combination of tokens $1\dots i$ only. Example row for "journey": `[0.5517, 0.4483, 0, 0, 0, 0]` (p. 76).
- **[[dropout|Dropout]] breaks the rule during training** (§3.5.2): after dropout with $p=0.5$, survivors are doubled and some rows no longer sum to 1 — the "journey" row became all zeros (p. 80). It is a convex combination only *on average* (see [[expected-value-and-mean]]).

## 🔴 Theory (Expert)
- **Convex hull.** The set of all convex combinations of $\mathbf x^{(1)},\dots,\mathbf x^{(T)}$ is their **convex hull** $\operatorname{conv}\{\mathbf x^{(j)}\}$, the smallest convex set containing them. So every attention output (before the value projection and dropout) lies inside the convex hull of the vectors it attends to. Consequence: each coordinate obeys $\min_j x^{(j)}_k \le z_k \le \max_j x^{(j)}_k$.
- **Probability view.** The weights $(\alpha_1,\dots,\alpha_T)$ form a categorical [[probability-distribution]] over positions. Then $\mathbf z = \mathbb E_{J\sim\alpha}[\mathbf x^{(J)}]$: the context vector is the [[expected-value-and-mean|expected value]] of the vector you would get by picking one token at random with probabilities $\alpha$. This is why "weighted average" and "expectation" are the same idea.
- **The simplex.** The set of valid weight vectors $\Delta^{T-1} = \{\boldsymbol\alpha\in\mathbb R^T:\alpha_j\ge0,\ \sum_j\alpha_j=1\}$ is the probability simplex. Softmax maps all of $\mathbb R^T$ onto the *interior* of the simplex (every weight strictly positive); masking with $-\infty$ moves you to a face of it (some weights exactly 0).
- **Renormalizing = conditioning.** Causal masking then renormalizing (Raschka's information-leakage sidebar, pp. 76–77) gives the same weights as running softmax on the unmasked scores only. In probability terms, it is conditioning the categorical distribution on the event "$J \le i$".
- **Jensen's inequality** (outside the book): for any convex function $f$, $f\!\left(\sum_j\alpha_j\mathbf x^{(j)}\right) \le \sum_j\alpha_j f(\mathbf x^{(j)})$. For example, with $f = \lVert\cdot\rVert$ (a [[norm]]), a context vector is never longer than the longest input it mixes. Convex combinations are also the basic object of [[convex-optimization]].

## Connections
- Builds on: [[vector]], [[vector-space]], [[affine-space]], [[softmax]]
- Leads to: [[context-vector]], [[attention-weights]], [[scaled-dot-product-attention]], [[convex-optimization]]
- Related: [[expected-value-and-mean]], [[probability-distribution]], [[mixture-model]] (mixture weights are also convex weights), [[causal-attention]], [[dropout]]
- Used by: [[self-attention]], [[attention-mechanism]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3.1 (normalization, softmax, context vector as weighted sum; Figs. 3.7–3.10), book pp. 56–61; §3.3.2 (rows sum to 1, `attn_weights @ inputs`), book pp. 61–64.
- Same PDF §3.4.1 (weighted sum over value vectors), book pp. 69–70; §3.5.1–3.5.2 (causal renormalization, information-leakage sidebar, dropout rows), book pp. 75–80.
