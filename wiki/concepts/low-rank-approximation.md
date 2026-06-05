---
title: Low-Rank Approximation (Eckart–Young)
type: concept
tags: [linear-algebra, matrix-decomposition, dimensionality-reduction]
difficulty: advanced
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
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

## Connections
- Builds on: [[singular-value-decomposition]], [[norm]], [[basis-and-rank]]
- Leads to: [[principal-component-analysis]], data compression, denoising, recommender systems
- Used by: [[principal-component-analysis]], image compression, topic modeling, matrix completion
- People: [[carl-eckart]], [[gale-young]]

## Sources
- `raw/mml-book.pdf` §4.6, Eqs 4.90–4.96, Def 4.23, Thms 4.24–4.25, Examples 4.15, printed pp. 129–133 (PDF pp. 134–138).
