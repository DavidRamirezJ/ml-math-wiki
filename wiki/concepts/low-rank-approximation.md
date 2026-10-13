---
title: Low-Rank Approximation (Eckart–Young)
type: concept
tags: [linear-algebra, matrix-decomposition, dimensionality-reduction, llm, fine-tuning]
difficulty: advanced
sources: [mml-ch04-matrix-decompositions, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-ch05-pretraining-on-unlabeled-data, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-06-04
updated: 2026-10-13
---

# Low-Rank Approximation (Eckart–Young)

> **One-liner:** Keep only a matrix's few biggest "stretch" pieces and you get the best possible simplified version of it — a compressed copy that loses the least information.

## 🟢 Intuition (Dummy)
A big data [[matrix]] (an image, a ratings table) can be rebuilt as a stack of simple layers, ordered
from most to least important. The first layer captures the dominant pattern, the next a finer
correction, and so on. **Low-rank approximation** says: keep just the top few layers and throw the
rest away. You get a much smaller object that still looks almost like the original — like a JPEG
keeping the broad strokes and dropping the tiny details. The remarkable fact (Eckart–Young) is that
this "keep the top few" choice is provably the *best* simplification you could make.

**In LLMs.** The same idea makes fine-tuning big language models cheap. When you adapt a model to
a new task, the *change* to each weight matrix seems to need only a few important directions. So
[[low-rank-adaptation|LoRA]] does not learn the full change. It learns a "keep the top few layers"
version of it from the start, built from two skinny matrices. A related method, [[galore]], uses
the same trick on the *gradients* to save memory during pretraining.

## 🟡 Mechanics (Practitioner)
Start from the [[singular-value-decomposition]] $A=U\Sigma V^\top$. Define **rank-1 building blocks**
from outer products of singular vectors:
$$A_i:=\mathbf u_i\mathbf v_i^\top.$$
Then the full matrix is a singular-value-weighted sum of them:
$$A=\sum_{i=1}^r \sigma_i\,\mathbf u_i\mathbf v_i^\top=\sum_{i=1}^r\sigma_i A_i.$$

**Rank-$k$ approximation** — truncate the sum at $k<r$:
$$\widehat A(k):=\sum_{i=1}^k\sigma_i\,\mathbf u_i\mathbf v_i^\top,\qquad \operatorname{rk}\big(\widehat A(k)\big)=k.$$

**Compression payoff:** the Stonehenge image example stores a $1432\times1910$ matrix
($\approx 2.7$ million numbers); a rank-5 approximation needs only
$5\cdot(1432+1910+1)=16{,}715$ numbers — about $0.6\%$ — yet the rocks are clearly recognizable.

**Spectral norm (Def 4.23):** the error metric is $\|A\|_2:=\max_{\mathbf x\ne\mathbf 0}\frac{\|A\mathbf x\|_2}{\|\mathbf x\|_2}$,
the largest stretch $A$ applies. **Thm 4.24:** $\|A\|_2=\sigma_1$.

**Eckart–Young theorem (Thm 4.25):** of *all* matrices $B$ of rank $\le k$, the truncated SVD is the
best approximation in spectral norm, and its error is the next discarded singular value:
$$\widehat A(k)=\operatorname*{arg\,min}_{\operatorname{rk}(B)=k}\|A-B\|_2,\qquad \big\|A-\widehat A(k)\big\|_2=\sigma_{k+1}.$$

**ML usage:** lossy compression, denoising (small $\sigma_i$ ≈ noise), regularizing ill-posed
problems, recommender systems (Example 4.15: a rank-2 approximation of the movie-ratings matrix
recovers the "sci-fi vs. art-house" structure and predicts unseen ratings), and the computational
core of [[principal-component-analysis]] and dimensionality reduction.

### In LLMs: LoRA and GaLore (Build a Large Language Model, App. E and App. B)
**LoRA** ([[low-rank-adaptation]]) applies the low-rank idea to the *weight update* during
[[fine-tuning]]. For a frozen layer weight $W$ (shape $d_{in}\times d_{out}$):

$$W_{\text{updated}} = W + \Delta W,\qquad \Delta W \approx AB,\qquad A\in\mathbb R^{d_{in}\times r},\ B\in\mathbb R^{r\times d_{out}}$$

Symbol by symbol:
- $\Delta W$ = the full update matrix that ordinary training would learn (same shape as $W$).
- $A$, $B$ = two small trainable matrices; $AB$ = their product, again of shape $d_{in}\times d_{out}$.
- $r$ = the **rank** (inner dimension), a [[hyperparameter]]; $\operatorname{rk}(AB)\le r$.

Storage works exactly like the Stonehenge count above, minus the singular values: a rank-$r$
factorization of an $m\times n$ matrix needs $r(m+n)$ numbers instead of $mn$. For GPT-2 small's
$768\times768$ attention matrices with $r=16$: $16\cdot1536 = 24{,}576$ numbers instead of
$589{,}824$. Over the whole model the book trains **2,666,528** LoRA parameters instead of
**124,441,346** ("almost 50×" fewer, book p. 332), reaching 96.64% validation accuracy on spam
classification (book p. 336). Details: [[lora-rank-and-alpha]], [[parameter-efficient-fine-tuning]].

**GaLore** ([[galore]], Zhao et al. 2024, listed in the book's App. B references for ch. 5,
book p. 294) applies a low-rank *projection to the gradient* so the optimizer's state is small,
while all weights are still updated. The book only cites it ("replace AdamW with GaLoreAdamW"); the
SVD connection is this wiki's note.

## 🔴 Theory (Expert)
- **Why the error is $\sigma_{k+1}$:** the residual is $A-\widehat A(k)=\sum_{i=k+1}^r\sigma_i\mathbf u_i\mathbf v_i^\top$,
  itself an SVD whose largest singular value is $\sigma_{k+1}$; by Thm 4.24 its spectral norm is $\sigma_{k+1}$.
- **Optimality (proof sketch):** suppose some rank-$\le k$ matrix $B$ beat it, $\|A-B\|_2<\sigma_{k+1}$.
  Then $\ker(B)$ has dimension $\ge n-k$, while $\operatorname{span}\{\mathbf v_1,\dots,\mathbf v_{k+1}\}$
  is a $(k+1)$-dim subspace on which $\|A\mathbf x\|_2\ge\sigma_{k+1}\|\mathbf x\|_2$. Two subspaces of
  $\mathbb R^n$ with dimensions summing to $>n$ must intersect nontrivially (rank–nullity, Thm 2.24);
  a nonzero $\mathbf x$ in both gives $\|A\mathbf x\|_2=\|(A-B)\mathbf x\|_2\le\|A-B\|_2\|\mathbf x\|_2<\sigma_{k+1}\|\mathbf x\|_2$,
  contradicting $\|A\mathbf x\|_2\ge\sigma_{k+1}\|\mathbf x\|_2$.
- The theorem also holds in the Frobenius norm, where the error is $\sqrt{\sum_{i>k}\sigma_i^2}$ — the
  version most directly tied to [[principal-component-analysis]] (minimizing reconstruction variance).
- Interpretation: $\widehat A(k)$ is the orthogonal **projection** of $A$ onto the manifold of
  rank-$\le k$ matrices. Generalizes to tensors via Tucker / CP decompositions.
- **Eckart–Young vs. LoRA.** Eckart–Young answers "given a *known* matrix, what is its best rank-$k$
  copy?" (truncated SVD). LoRA answers a different question: "learn a rank-$r$ update $AB$ directly by
  [[gradient-descent]] on a task loss", without ever computing the full-rank $\Delta W$ to compress.
  Both search the same set $\{M:\operatorname{rk}(M)\le r\}$, which every product $AB$ with inner
  dimension $r$ reaches (split a truncated SVD $U_r\Sigma_rV_r^\top$ as $A=U_r\Sigma_r$, $B=V_r^\top$).
  If the ideal update's singular values decay fast, Eckart–Young says a small $r$ loses only
  $\sigma_{r+1}$, which is the theoretical reason a small LoRA rank can work.
- The factorization $AB$ is not unique: $(AM)(M^{-1}B)=AB$ for any invertible $r\times r$ matrix $M$.
  The SVD fixes this freedom by demanding orthonormal factors; LoRA leaves it free.

## Connections
- Builds on: [[singular-value-decomposition]], [[norm]], [[basis-and-rank]]
- Leads to: [[principal-component-analysis]], data compression, denoising, recommender systems, [[low-rank-adaptation]], [[galore]]
- Used by: [[principal-component-analysis]], image compression, topic modeling, matrix completion, [[low-rank-adaptation]] (LoRA fine-tuning), [[parameter-efficient-fine-tuning]], [[galore]]
- People: [[carl-eckart]], [[gale-young]], [[edward-j-hu]] (LoRA), [[jiawei-zhao]] (GaLore)

## Sources
- `raw/mml-book.pdf` §4.6, Eqs 4.90–4.96, Def 4.23, Thms 4.24–4.25, Examples 4.15, printed pp. 129–133 (PDF pp. 134–138).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.1, book pp. 322–324 (PDF pp. 344–346): LoRA, $\Delta W\approx AB$, "low-rank" as a smaller-dimensional subspace of weight space.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book pp. 328–336: LoRA layer, 2,666,528 vs 124,441,346 parameters (p. 332), accuracies (p. 336).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, ch. 5 references, book p. 294: GaLore (Zhao et al. 2024, arXiv 2403.03507).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8, Exercise 7.4, book p. 247 (LoRA for instruction fine-tuning).
