---
title: Attention Scores
type: concept
tags: [attention, transformer, linear-algebra, llm]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Attention Scores

> **One-liner:** An attention score is a raw number that says how well one token "matches" another token — big means "very relevant", small means "not so much".

## 🟢 Intuition (Dummy)
Imagine you are reading the sentence "Your journey starts with one step" and you are
focused on the word **journey**. You quickly ask, for every other word: "How much do you
have to do with me?" Each answer is a number. Those raw, unpolished numbers are the
**attention scores**.

They are not percentages yet. They can be any size, they do not add up to anything in
particular, and they can even be negative. They are just "match strengths". In the next
step, a function called [[softmax]] cleans them up into nice percentages — the
[[attention-weights]].

The way the model measures "match strength" is the [[inner-product|dot product]]: line up
two lists of numbers, multiply them position by position, add everything up. Two vectors
that point in a similar direction give a big dot product; unrelated ones give a small one.

## 🟡 Mechanics (Practitioner)

### Version 1 — simplified self-attention (no trainable weights, §3.3.1)
The book first uses the token embeddings themselves. Pick the query token $x^{(2)}$
("journey") and score it against every input $x^{(j)}$:

$$\omega_{2j} = x^{(2)} \cdot x^{(j)} = \sum_{k=1}^{d} x^{(2)}_k\, x^{(j)}_k$$

Reading it symbol by symbol:
- $\omega$ (Greek "omega") = the book's letter for an attention **score**.
- $\omega_{2j}$ = the score of query token 2 against input token $j$ (first index = who is asking, second = who is being looked at).
- $x^{(2)}$ = the embedding vector of token 2 ("journey"); the superscript in parentheses is the token's **position**, not a power.
- $\cdot$ = dot product.
- $\sum_{k=1}^{d}$ = "add up, for $k$ going from 1 to $d$"; $d$ = embedding size (here 3).
- $x^{(2)}_k$ = the $k$-th number inside vector $x^{(2)}$.

**The book's numbers** (book p. 57). Six tokens, $d = 3$:

| token | embedding |
|---|---|
| Your $x^{(1)}$ | [0.43, 0.15, 0.89] |
| journey $x^{(2)}$ | [0.55, 0.87, 0.66] |
| starts $x^{(3)}$ | [0.57, 0.85, 0.64] |
| with $x^{(4)}$ | [0.22, 0.58, 0.33] |
| one $x^{(5)}$ | [0.77, 0.25, 0.10] |
| step $x^{(6)}$ | [0.05, 0.80, 0.55] |

Worked check: $\omega_{21} = 0.55\cdot0.43 + 0.87\cdot0.15 + 0.66\cdot0.89 = 0.2365 + 0.1305 + 0.5874 = 0.9544$.
All scores for "journey" (book p. 58):

```
tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

"journey" scores highest with itself (1.4950) and with "starts" (1.4754), whose embedding
is almost the same. That is exactly the "dot product = similarity" idea from the book's
sidebar (book pp. 58–59).

```python
query = inputs[1]                          # x^(2), "journey"
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
```

### All scores at once — the score matrix (§3.3.2)
Do it for every query at the same time with one [[matrix-multiplication]]:

$$\Omega = X X^\top, \qquad \Omega_{ij} = x^{(i)} \cdot x^{(j)}$$

- $X$ = the $6\times 3$ matrix whose rows are the token embeddings (`inputs`).
- $X^\top$ = its [[matrix-transpose|transpose]] ($3\times 6$).
- $\Omega$ (capital omega) = the $6\times 6$ table of all scores; row $i$ = query $i$, column $j$ = token $j$.

```python
attn_scores = inputs @ inputs.T     # (6,3) @ (3,6) -> (6,6)
```
Book output (pp. 62–63), first two rows:
```
[[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
 [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865], ...]
```
Notice it is **symmetric** (entry [0,1] = entry [1,0] = 0.9544): without trainable weights,
"how much A matches B" equals "how much B matches A". $XX^\top$ is a [[gram-matrix]].

### Version 2 — with trainable weights (§3.4.1)
Real LLMs first project each token into a **query** and a **key** (see [[query-key-value]])
and score those instead:

$$\omega_{2i} = q^{(2)} \cdot k^{(i)}, \qquad \Omega = Q K^\top$$

- $q^{(2)} = x^{(2)} W_q$ = query vector of token 2; $k^{(i)} = x^{(i)} W_k$ = key vector of token $i$.
- $Q$, $K$ = all queries / all keys stacked as rows ($T \times d_{out}$).
- $QK^\top$ = every query dotted with every key → a $T\times T$ score matrix.

Book numbers ($d_{in}=3$, $d_{out}=2$, `torch.manual_seed(123)`): `query_2 = [0.4306, 1.4551]`,
$\omega_{22}$ = `1.8524`, and (book p. 68)
```
attn_scores_2 = tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
```
```python
attn_scores_2 = query_2 @ keys.T     # (2,) @ (2,6) -> (6,)
```
Now the matrix is **not** symmetric in general, because queries and keys use different
matrices $W_q \neq W_k$.

### What happens next
Scores are **unnormalized**. Before softmax they are divided by $\sqrt{d_k}$ (the
"scaled" in [[scaled-dot-product-attention]]), optionally masked with $-\infty$ for
[[causal-attention]] (book p. 77), and then turned into [[attention-weights]] by [[softmax]].
In multi-head attention the same score computation runs per head:
`queries @ keys.transpose(2, 3)` gives shape (b, num_heads, T, T) (Listing 3.5, book p. 87; explained pp. 88–89).

## 🔴 Theory (Expert)
- **Scores are a bilinear form.** With trainable weights,
  $\omega_{ij} = x^{(i)} W_q W_k^\top {x^{(j)}}^\top = x^{(i)} M {x^{(j)}}^\top$ with
  $M = W_q W_k^\top \in \mathbb R^{d_{in}\times d_{in}}$ of rank $\le d_{out}$. $M$ is generally
  neither symmetric nor positive definite, so this is **not** an [[inner-product]] on the
  embedding space in the strict (MML Def. 3.3) sense — it is a dot product in the
  projected query/key space. With $W_q = W_k$ it becomes $x M x'^\top$ with $M$ symmetric
  positive semidefinite (a Gram-type form). (Our analysis, not in the book.)
- **Why scaling is needed.** If query and key components are roughly independent with mean 0
  and variance 1, then $\operatorname{Var}(q\cdot k) = d_k$, so raw scores grow like
  $\sqrt{d_k}$. GPT-like models have $d_k$ far larger than 2 (embedding sizes above 1,000),
  so unscaled scores push softmax into its "step function" regime with near-zero gradients
  (book p. 69). Dividing by $\sqrt{d_k}$ keeps the score variance near 1. (Variance argument
  is our addition; the book gives the gradient argument.)
- **Translation invariance downstream.** Adding the same constant to every score in a row
  does not change the attention weights (softmax is shift-invariant; see [[softmax]]).
  Only *differences* between scores in a row matter.
- **Cost.** The score matrix has $T^2$ entries, so memory and compute grow quadratically
  with [[context-length]] — the motivation for kernels such as [[flashattention]].
- **Masking acts on scores.** Setting future scores to $-\infty$ before softmax is
  equivalent to softmax-then-zero-then-renormalize, with no leakage (book pp. 76–77; see
  [[causal-attention-mask]]).

## Connections
- Builds on: [[inner-product]], [[vector]], [[matrix-multiplication]], [[matrix-transpose]], [[token-embedding]]
- Leads to: [[attention-weights]], [[softmax]], [[scaled-dot-product-attention]], [[causal-attention-mask]]
- Used by: [[self-attention]], [[attention-mechanism]], [[multi-head-attention]], [[transformer-architecture]]
- Related: [[query-key-value]], [[gram-matrix]], [[context-vector]]
- People: [[ashish-vaswani]], [[dzmitry-bahdanau]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3.1, book pp. 56–59 (dot-product scores, sidebar "Understanding dot products"); §3.3.2, book pp. 61–63 (score matrix $XX^\top$).
- Same, §3.4.1, book pp. 65–69 (query–key scores, $\omega_{22}=1.8524$, scaling rationale sidebar p. 69).
- Same, §3.5.1, book p. 77 ($-\infty$ masking of scores); §3.6.2, book pp. 86–89 (per-head scores, Listing 3.5).
