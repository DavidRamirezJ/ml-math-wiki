---
title: Inner Product
type: concept
tags: [analytic-geometry, inner-product, foundations, attention, llm]
difficulty: intermediate
sources: [mml-ch03-analytic-geometry, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-appA-introduction-to-pytorch]
created: 2026-06-04
updated: 2026-10-13
---

# Inner Product

> **One-liner:** An inner product is a way to multiply two vectors into a single number that tells you how much they point in the same direction.

## 🟢 Intuition (Dummy)
Two arrows can point the same way, opposite ways, or at right angles. An inner product squeezes
that relationship into one number: big and positive when they agree, zero when they're
perpendicular, negative when they oppose. Once you can do this, you get *everything* geometric for
free — the length of an arrow is its inner product with itself, the angle between two arrows comes
from their inner product, and "perpendicular" just means "inner product equals zero." It is the
single tool that turns a bag of vectors into a space with real geometry.

**In LLMs.** This "how much do two arrows agree?" number is the engine of attention. When
a language model reads "Your journey starts with one step", each word asks every other
word "how related are we?" — and the answer is a dot product. The book puts it plainly:
beyond being a math tool, the dot product "is a measure of similarity because it
quantifies how closely two vectors are aligned" (book p. 59). The same trick picks the next
word at the very end of a GPT model (see below).

## 🟡 Mechanics (Practitioner)
Start from the familiar **dot product** on $\mathbb R^n$:
$$ x^\top y = \sum_{i=1}^n x_i y_i. $$

An inner product generalizes this. A mapping $\Omega : V\times V \to \mathbb R$ is **bilinear**
(linear in each argument), and is an **inner product** $\langle x,y\rangle$ if it is also
(Defs 3.2–3.3):

- **Symmetric:** $\langle x,y\rangle = \langle y,x\rangle$
- **Positive definite:** $\langle x,x\rangle > 0$ for $x\ne \mathbf 0$, and $\langle \mathbf 0,\mathbf 0\rangle = 0$

$(V,\langle\cdot,\cdot\rangle)$ is an **inner product space**; with the dot product it is a
**Euclidean vector space**. Every inner product **induces a norm** $\|x\|=\sqrt{\langle x,x\rangle}$
([[norm]]) and hence distances ([[metric]]) and [[angle-between-vectors|angles]].

**Matrix form (Thm 3.5).** Relative to an ordered basis $B$, with coordinates $\hat x,\hat y$,
$$ \langle x,y\rangle = \hat x^\top A\,\hat y, \qquad A_{ij} := \langle b_i,b_j\rangle, $$
where $A$ is a [[symmetric-positive-definite-matrix]]. Conversely every such $A$ defines an inner product.

**Worked example (non-dot inner product, §3.2).** On $\mathbb R^2$,
$\langle x,y\rangle = x_1y_1 - (x_1y_2 + x_2y_1) + 2x_2y_2$ is a valid inner product
(symmetric matrix $A=\big[\begin{smallmatrix}1 & -1\\ -1 & 2\end{smallmatrix}\big]$, positive definite).
It gives different lengths and angles than the dot product.

**ML usage.** Inner products measure **similarity** (cosine similarity, attention scores). The
"kernel trick" replaces $\langle x,y\rangle$ with $k(x,y)$, an inner product in a higher-dimensional
feature space — the foundation of SVMs and Gaussian processes (Ch. 12).

### In LLMs (Raschka, *Build a Large Language Model*)
**1. Attention scores are dot products (§3.3.1).** For the query token $x^{(2)}$ ("journey"):
$$\omega_{2j} = x^{(2)}\cdot x^{(j)} = \sum_{k=1}^{d} x^{(2)}_k\,x^{(j)}_k$$
- $\omega_{2j}$ = attention score of token 2 against token $j$; $x^{(j)}$ = embedding of token $j$;
  $d$ = embedding size (3 here); $x^{(2)}_k$ = the $k$-th entry of $x^{(2)}$.

With "Your" = [0.43, 0.15, 0.89] and "journey" = [0.55, 0.87, 0.66]:
$0.43\cdot0.55 + 0.15\cdot0.87 + 0.89\cdot0.66 = 0.9544$ (book p. 58). The book checks it
with a loop and with `torch.dot`:
```python
res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print(res)                          # tensor(0.9544)
print(torch.dot(inputs[0], query))  # tensor(0.9544)
```
"journey" scores highest with itself (1.4950) and with the near-identical "starts" (1.4754)
— bigger dot product, more similar (book pp. 58–59). See [[attention-scores]].

**2. All pairs at once = a Gram matrix (§3.3.2, App. A.2).** Stacking embeddings as rows of
$X$, every pairwise dot product is one matrix product:
$$\Omega = XX^\top,\qquad \Omega_{ij} = \langle x^{(i)}, x^{(j)}\rangle$$
- $X^\top$ = transpose of $X$; $\Omega$ = the $T\times T$ table of scores (`inputs @ inputs.T`).

It is symmetric (e.g. $\Omega_{12} = \Omega_{21} = 0.9544$, book pp. 62–63) — the
**symmetry** axiom in action. Appendix A's toy tensor shows the same thing:
$\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}^{\!\top} = \begin{bmatrix}14&32\\32&77\end{bmatrix}$
with $14 = 1^2+2^2+3^2$ and $32 = 1\cdot4+2\cdot5+3\cdot6$ (book p. 261). See [[gram-matrix]].

**3. Query–key dot products (§3.4).** With trainable projections $q = xW_q$, $k = xW_k$
([[query-key-value]]), the score is $q^{(2)}\cdot k^{(i)}$, e.g. $\omega_{22} = 1.8524$
(book p. 68), and all scores are $QK^\top$. Scores are divided by $\sqrt{d_k}$ before
[[softmax]] because large dot products push softmax into a "step function" with tiny
gradients (book p. 69). In multi-head attention, `queries @ keys.transpose(2, 3)` computes
the dot products for every head in one batched multiplication (book pp. 87–89).

**4. Logits are dot products too (weight tying, §4.6 / §5.5).** GPT-2 reuses the token
embedding matrix (shape 50,257 × 768) as the output layer (book p. 121; book p. 166). Then
the logit for vocabulary token $v$ is
$$\text{logit}_v = \langle h_t, e_v\rangle$$
- $h_t\in\mathbb R^{768}$ = final hidden vector at position $t$; $e_v$ = token $v$'s embedding row.

So "which word comes next?" becomes "which word embedding points most in the direction of
the current hidden state?" (Interpretation ours; see [[weight-tying]], [[logits]].)

## 🔴 Theory (Expert)
- **Cauchy–Schwarz inequality:** $|\langle x,y\rangle| \le \|x\|\,\|y\|$ — the guarantee that
  $\frac{\langle x,y\rangle}{\|x\|\|y\|}\in[-1,1]$, which is what makes the angle definition
  well-posed. See [[augustin-louis-cauchy]] and [[hermann-schwarz]].
- **Inner product of functions (§3.7):** for $u,v:\mathbb R\to\mathbb R$,
  $\langle u,v\rangle := \int_a^b u(x)v(x)\,dx$. With $a=-\pi,b=\pi$, $\sin$ and $\cos$ are orthogonal,
  and $\{1,\cos x,\cos 2x,\dots\}$ is an orthogonal set — the basis of **Fourier series**. Making this
  rigorous (measures, convergence) leads to **Hilbert spaces**.
- Positive definiteness of the Gram matrix $A_{ij}=\langle b_i,b_j\rangle$ is exactly the
  **Mercer condition** that defines valid kernels; positive *semi*definiteness suffices for kernels
  (§12.4).
- **LLM attention is (usually) not an inner product on embedding space.** The trainable
  score $q^{(i)}\cdot k^{(j)} = x^{(i)}W_qW_k^\top {x^{(j)}}^\top$ is a bilinear form
  $\hat x^\top A\,\hat y$ with $A = W_qW_k^\top$ — exactly the matrix form of Thm 3.5 — but
  $A$ is generally **neither symmetric nor positive definite** (and has rank $\le d_{out}$),
  so it fails Def. 3.3. It *is* the ordinary dot product in the projected query/key space.
  Only the simplified no-weights version $XX^\top$ (book §3.3) is a genuine Gram matrix of
  an inner product. (Our analysis; the book simply calls both "dot products".)
- **Dot product vs. cosine similarity in attention.** Attention uses the *unnormalized* dot
  product $\langle q,k\rangle = \|q\|\|k\|\cos\theta$, so vector length matters as well as
  angle. By Cauchy–Schwarz, $|q\cdot k| \le \|q\|\|k\|$, and with i.i.d. unit-variance
  entries $\operatorname{Var}(q\cdot k) = d_k$ — the reason for the $1/\sqrt{d_k}$ scaling in
  [[scaled-dot-product-attention]] (variance argument ours; book gives the gradient argument, p. 69).

## Connections
- Builds on: [[vector-space]], [[linear-mapping]] (bilinearity), [[norm]]
- Leads to: [[symmetric-positive-definite-matrix]], [[metric]], [[angle-between-vectors]], [[orthogonality]], [[orthogonal-projection]]
- Used by: kernel methods / SVM (Ch. 12), Gaussian processes, cosine similarity, [[attention-scores]], [[self-attention]], [[scaled-dot-product-attention]], [[query-key-value]], [[weight-tying]]
- Related: [[gram-matrix]], [[matrix-multiplication]], [[batched-matrix-multiplication]]
- People: [[augustin-louis-cauchy]], [[hermann-schwarz]]

## Sources
- `raw/mml-book.pdf` §3.2 (Defs 3.2–3.3), §3.3 (Cauchy–Schwarz 3.17), §3.7, printed pp. 72–75, 80–81 (PDF pp. 77–80, 85–86).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3.1, book pp. 58–59 (sidebar "Understanding dot products"); §3.3.2, book pp. 61–63 ($XX^\top$).
- Same, §3.4.1, book pp. 67–69 (query–key dot products, scaling sidebar); §3.6.2, book pp. 87–89 (per-head batched dot products).
- Same, §4.6, book p. 121 and §5.5, book p. 166 (weight tying → logits as dot products).
- Same, §A.2, book p. 261 (`tensor2d @ tensor2d.T`).
