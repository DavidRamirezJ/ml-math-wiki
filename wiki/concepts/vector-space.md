---
title: Vector Space
type: concept
tags: [linear-algebra, foundations, llm, embeddings]
difficulty: intermediate
sources: [mml-ch02-linear-algebra, llm-ch02-working-with-text-data, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-06-04
updated: 2026-10-13
---

# Vector Space

> **One-liner:** A vector space is the complete collection of every vector you could ever reach by adding and scaling a starting set — nothing escapes it.

## 🟢 Intuition (Dummy)
Start with a couple of arrows. Add them, scale them, add the results, scale those... Where can
you end up? The set of *all* possible destinations is the vector space. It's "closed": you can
never operate your way *out* of it. If your arrows live in a flat plane, no amount of adding and
scaling lets you leave the plane — the plane is the vector space.

That property — "you can't escape" — is called **closure**, and it's the single most important
idea in this chapter.

### In LLMs
A language model can't do math on the *word* "fox" — words are labels, not numbers. So the very
first thing an LLM does is give every token an address in a vector space: a list of numbers like
$[1.78,\ 0.18,\ -2.10]$. Once words live in a vector space, you can add them, scale them, and
measure how close they are — and closeness starts to mean something: in a good embedding space,
"duck", "eagle" and "goose" end up as neighbours, far away from "Berlin" and "London"
(Raschka, Fig. 2.3). The vector space is the "playground" where all the LLM's math happens.

## 🟡 Mechanics (Practitioner)
A **real vector space** is a set $V$ with two operations:
- vector addition $+ : V \times V \to V$
- scalar multiplication $\cdot : \mathbb{R} \times V \to V$

satisfying these axioms (closure is built into the codomains above):
- **Associativity & commutativity** of $+$
- **Additive identity** $\mathbf{0}$: $\;\mathbf{x} + \mathbf{0} = \mathbf{x}$
- **Additive inverse** $-\mathbf{x}$: $\;\mathbf{x} + (-\mathbf{x}) = \mathbf{0}$
- **Distributivity**: $\lambda(\mathbf{x}+\mathbf{y}) = \lambda\mathbf{x}+\lambda\mathbf{y}$ and $(\lambda+\psi)\mathbf{x} = \lambda\mathbf{x}+\psi\mathbf{x}$
- **Associativity of scaling**: $\lambda(\psi\mathbf{x}) = (\lambda\psi)\mathbf{x}$
- **Scaling identity**: $1\cdot\mathbf{x} = \mathbf{x}$

**Canonical example:** $\mathbb{R}^n$ with component-wise $+$ and scaling. **Other examples:**
polynomials, real-valued functions, $m\times n$ matrices.

### In LLMs: embeddings are maps into $\mathbb{R}^d$
*Build a Large Language Model (From Scratch)* defines an **embedding** as "a mapping from discrete
objects, such as words, images, or even entire documents, to points in a continuous vector space"
(§2.1, book p. 19). Written as a formula (our notation, not printed in the book):
$$E:\mathcal{V}\to\mathbb{R}^{d},\qquad w\mapsto \mathbf{e}_w$$
Reading it symbol by symbol:
- $E$ = the embedding function; the colon means "is a function from … to …".
- $\mathcal{V}$ = the vocabulary, a finite set of discrete tokens (see [[vocabulary]]).
- $\to$ = "maps into"; $\mathbb{R}^d$ = the vector space of all lists of $d$ real numbers.
- $d$ = the **embedding dimension** ([[embedding-dimension]]).
- $w\mapsto\mathbf{e}_w$ = "the token $w$ is sent to its vector $\mathbf{e}_w$".

**Why a vector space and not just integers?** Text is categorical, so it "isn't compatible with the
mathematical operations used to implement and train neural networks" (book p. 18). Integers like
token ID 3 vs. 4 have no meaningful sum or distance; vectors in $\mathbb{R}^d$ do, and they can be
nudged by [[gradient-descent]].

**Concrete sizes from the book:**

| Setting | $d$ (dimension of the space) | Source |
|---|---|---|
| Fig. 2.2 toy video/audio/text embeddings | 3 | book p. 19 |
| §2.7 toy embedding layer | 3 | book p. 42 |
| §2.8 realistic experiment | 256 | book p. 46 |
| Smallest GPT-2 (117M/125M params) | 768 | book p. 20 |
| Largest GPT-3 (175B params) | 12,288 | book p. 20 |

**Vector addition in action.** The input to GPT is the *sum* of two vectors in the same space:
$$\mathbf{x}_t = E_{\text{tok}}[\text{id}_t] + E_{\text{pos}}[t]$$
($\mathbf{x}_t$ = input vector at position $t$; $E_{\text{tok}}[\text{id}_t]$ = the row of the
token-embedding matrix for the token at position $t$; $E_{\text{pos}}[t]$ = the positional vector
for position $t$; $+$ = component-wise vector addition.) This only works because both vectors live in
the *same* $\mathbb{R}^d$ — closure under addition in action (§2.8, book pp. 45–47; see
[[token-embedding]], [[positional-embedding]]). Fig. 2.18's worked numbers:
$[1,1,1]+[1.1,1.2,1.3]=[2.1,2.2,2.3]$ (book p. 45).

## 🔴 Theory (Expert)
Algebraically, $(V, +)$ is an **abelian group** and scalar multiplication makes $V$ a **module over
the field $\mathbb{R}$** (a module over a field *is* a vector space). The chapter's mind map (Fig. 2.2)
makes this lineage explicit: *Group → Vector space → ...*.

- A **subspace** $U \subseteq V$ is a subset that is itself closed under $+$ and scaling (and contains
  $\mathbf{0}$). Solution sets of *homogeneous* linear systems are subspaces; general solution sets are
  [[affine-space|affine subspaces]] (a subspace shifted off the origin).
- The **span** of a set of vectors is the smallest subspace containing them — the operational meaning
  of closure.
- [[linear-independence]] + spanning ⇒ a [[basis-and-rank|basis]]; the basis size is the **dimension**,
  and it pins down the 1:1 correspondence with $\mathbb{R}^{\dim V}$ for finite-dimensional spaces.

"The concept of a vector space and its properties underlie much of machine learning" (p. 18).

### In LLMs
- **Embedding spaces are $\mathbb{R}^d$ with extra geometry.** The bare axioms give no notion of
  "close"; LLM practice adds an [[inner-product]] (hence [[norm]] and cosine similarity
  $\frac{\mathbf{e}_a^\top\mathbf{e}_b}{\|\mathbf{e}_a\|\|\mathbf{e}_b\|}$) so that "similar words cluster"
  (Fig. 2.3, book p. 20) becomes a measurable statement. The book states the clustering in words only;
  the formula is our link.
- **The lookup is a linear map restricted to a basis.** With $W\in\mathbb{R}^{V\times d}$ and the
  one-hot vector $\mathbf{x}_i$ (the $i$-th standard basis vector of $\mathbb{R}^V$),
  $\mathbf{x}_i^\top W = W_{i,:}$ (book p. 43 states the one-hot equivalence). So the embedding layer
  is the [[linear-mapping]] $\mathbb{R}^V\to\mathbb{R}^d$ evaluated on basis vectors; its image is
  the row space of $W$, a subspace of dimension $\operatorname{rank}(W)\le\min(V,d)$ — for GPT-2,
  at most 768 inside $\mathbb{R}^{768}$ even though $V=50{,}257$.
- **Subspaces in LoRA.** Appendix E says "low-rank" means "limiting model adjustments to a smaller
  dimensional subspace of the total weight parameter space" (book p. 322), with
  $\Delta W\approx AB$, $A\in\mathbb{R}^{d\times r}$, $B\in\mathbb{R}^{r\times d}$ (book p. 323; see
  [[low-rank-adaptation]]).

> ⚠️ Contradiction (terminology, MML vs. Raschka): by the definition on this page (MML Def. 2.9 /
> subspace = closed under $+$), the set of *all* matrices of rank $\le r$ is **not** a subspace —
> e.g. two rank-1 matrices can sum to a rank-2 matrix. Raschka's "smaller dimensional subspace"
> (book p. 322) is precise only for a *fixed* factor: with $A$ fixed, $\{AB : B\in\mathbb{R}^{r\times d}\}$
> is a linear subspace of dimension $rd$, and every update $AB$ has its columns in the
> $r$-dimensional column space of $A$. Read the book's phrase as "a low-dimensional set of
> directions", not as a subspace in the strict axiomatic sense.

## Connections
- Builds on: [[vector]]
- Leads to: [[linear-independence]], [[basis-and-rank]], [[linear-mapping]], [[affine-space]], [[inner-product]], [[norm]]
- Used by: [[system-of-linear-equations]], [[word-embeddings]], [[token-embedding]], [[positional-embedding]], [[embedding-dimension]], [[low-rank-adaptation]], [[dimensionality-reduction]]
- People: —

## Sources
- `raw/mml-book.pdf` §2 intro & §2.4 (**Definition 2.9**), printed pp. 17–18, 35–37 (PDF pp. 22–23, 40–42).
- ✅ Axioms validated verbatim against Definition 2.9 on 2026-06-04.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1 (embedding definition, Figs. 2.2–2.3, GPT-2/GPT-3 dimensions), book pp. 18–20.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7 (one-hot equivalence), book pp. 42–43; §2.8 (token + positional vector addition, Fig. 2.18, $d=256$, chapter summary "continuous vector spaces"), book pp. 43–49.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` App. E §E.1 ("smaller dimensional subspace", $\Delta W\approx AB$), book pp. 322–324.
