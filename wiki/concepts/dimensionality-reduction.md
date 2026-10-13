---
title: Dimensionality Reduction
type: concept
tags: [dimensionality-reduction, unsupervised-learning, compression, embeddings, visualization]
difficulty: beginner
sources: [mml-ch10-dimensionality-reduction-pca, llm-ch02-working-with-text-data]
created: 2026-06-04
updated: 2026-10-13
---

# Dimensionality Reduction

> **One-liner:** Squeezing data with lots of numbers per item down to just a few numbers, keeping almost everything that matters and throwing away the redundant parts.

## 🟢 Intuition (Dummy)
A color photo that's 640×480 pixels is, mathematically, a single point with about a *million*
numbers. That's hard to store, hard to draw, and hard to reason about. But those million numbers
aren't really independent: neighboring pixels look alike, and a photo of a face has far less "real"
freedom than a million random dots. So a lot of the numbers are **redundant** — they can be guessed
from the others.

Dimensionality reduction finds the few numbers that actually carry the information and tosses the
rest. It's exactly like MP3 for music or JPEG for images: a *compression* trick. You lose a little
detail, but you keep the gist and save a ton of space. The catch is choosing *which* directions to
keep — keep the ones where the data spreads out, drop the ones where it barely moves.

### In LLMs: squeezing word vectors onto paper
LLMs give every word a vector with hundreds or thousands of numbers — 768 for the smallest GPT-2,
12,288 for the largest GPT-3 (Raschka, book p. 20). Nobody can *look* at a 768-dimensional point:
"our sensory perception and common graphical representations are inherently limited to three
dimensions or fewer" (book p. 20). So when people want to *see* that "duck", "eagle" and "goose"
sit together, they squash the vectors down to 2-D and draw a scatterplot (Fig. 2.3). That squash is
dimensionality reduction used as a **microscope**, not as compression.

## 🟡 Mechanics (Practitioner)
**Setup.** Data points $\mathbf x_n\in\mathbb R^D$ are assumed to lie (approximately) on a
**lower-dimensional subspace** $U\subseteq\mathbb R^D$ with $\dim(U)=M<D$. We want a
**compressed code** $\mathbf z_n\in\mathbb R^M$ and a **reconstruction** $\tilde{\mathbf x}_n\in U$
that stays close to $\mathbf x_n$.

In the **linear** case (this chapter), we pick an orthonormal $B=[\mathbf b_1,\dots,\mathbf b_M]\in\mathbb R^{D\times M}$ and set
$$ \mathbf z_n = B^\top\mathbf x_n \quad(\text{encode}),\qquad \tilde{\mathbf x}_n = B\mathbf z_n = BB^\top\mathbf x_n \quad(\text{decode}). $$
The quality measure is the **average squared reconstruction error** $\frac1N\sum_n\|\mathbf x_n-\tilde{\mathbf x}_n\|^2$.

**Why it works on real data:** high-dimensional data is usually *overcomplete* (redundant dimensions)
and *correlated* (intrinsic low-dimensional structure). Figure 10.1: a 2-D cloud that barely varies
in $x_2$ is nearly a 1-D line — keep $x_1$, drop $x_2$, lose almost nothing.

**ML usage:** visualization (project to 2-D/3-D), compression/storage, denoising, feature extraction,
and as a preprocessing step before clustering or classification. The flagship linear method is
[[principal-component-analysis]].

### In LLMs: projecting embeddings for visualization
*Build a Large Language Model (From Scratch)* §2.1 says that with Word2Vec-style embeddings,
"when projected into two-dimensional word embeddings for visualization purposes, similar terms are
clustered together" (book p. 19). Fig. 2.3 (book p. 20) shows the result: a bird cluster
(*duck, eagle, goose*), *squirrel* just outside it, country–capital pairs (*Germany–Berlin*,
*England–London*), and word forms (*long, longer, longest*).

The book does not name a projection method. A linear choice is the PCA encoder from above:
$$\mathbf{z}_w = B^\top\mathbf{e}_w,\qquad B\in\mathbb{R}^{d\times 2},\ B^\top B = I_2$$
Symbol by symbol:
- $\mathbf{e}_w\in\mathbb{R}^d$ = the embedding vector of word $w$ (e.g. $d=768$).
- $B$ = a matrix whose 2 columns are orthonormal directions (the top-2 principal components);
  $B^\top$ = its [[matrix-transpose|transpose]].
- $B^\top B=I_2$ = "the two columns have length 1 and are perpendicular".
- $\mathbf{z}_w\in\mathbb{R}^2$ = the 2-D point you actually plot.

**The dimensionality trade-off itself** (book pp. 19–20): embeddings "can have varying dimensions,
from one to thousands. A higher dimensionality might capture more nuanced relationships but at the
cost of computational efficiency." In GPT-2/GPT-3 the embedding size is "a tradeoff between
performance and efficiency" (book p. 20). Choosing $d$ is therefore a design decision made *before*
training — the LLM is built at its chosen $d$ rather than reduced afterwards (see
[[embedding-dimension]]).

## 🔴 Theory (Expert)
- The general framing is an **encoder–decoder** pair (an [[auto-encoder]]); linear PCA is the case
  where both maps are matrices ($B^\top$ and $B$). Replacing them with neural networks gives a
  deep (nonlinear) auto-encoder.
- "Information content" is operationalized as **variance** (spread): retaining the most information
  ⇔ retaining the most variance. This is the bridge to PCA's [[pca-maximum-variance|max-variance]]
  derivation.
- Linear dimensionality reduction is a *family*: [[principal-component-analysis|PCA]],
  factor analysis, and ICA differ only in the assumed noise/latent structure (Cunningham &
  Ghahramani, 2015). Nonlinear variants: kernel PCA, GP-LVM, deep auto-encoders.
- Choosing $M$ (the intrinsic dimensionality) is itself a model-selection problem — eigenvalue
  scree plots, the Gavish–Donoho threshold $4\sigma\sqrt{D}/\sqrt 3$, cross-validation, or Bayesian
  criteria.

### In LLMs
- **A 2-D picture can lie.** Projecting from $d=768$ to $2$ keeps only the two highest-variance
  directions; clusters that look separate in 2-D are suggestive, not proof, of separation in
  $\mathbb{R}^{768}$. Nonlinear methods (t-SNE, UMAP) preserve local neighbourhoods better but distort
  global distances. The book only shows the 2-D plot (Fig. 2.3) and does not discuss these caveats —
  this is our addition.
- **Related idea, different goal:** LoRA (Appendix E) restricts *weight updates* to a low-rank
  factorization $\Delta W\approx AB$ — a low-dimensional parametrization of changes, not a
  projection of data. See [[low-rank-adaptation]] and [[low-rank-approximation]].

## Connections
- Builds on: [[vector-space]], [[basis-and-rank]], [[linear-mapping]], [[embedding-dimension]]
- Leads to: [[principal-component-analysis]], [[probabilistic-pca]], [[auto-encoder]]
- Used by: data visualization (e.g. plotting [[word-embeddings]] from [[word2vec]]), compression, [[feature-extraction]], denoising
- People: [[karl-pearson]], [[harold-hotelling]]

## Sources
- `raw/mml-book.pdf` §10 intro & §10.1, printed pp. 317–319 (PDF pp. 323–325).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1 (2-D projection of word embeddings, Fig. 2.3, dimensionality trade-off, GPT-2 768 / GPT-3 12,288), book pp. 19–20.
