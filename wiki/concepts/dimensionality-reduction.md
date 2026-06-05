---
title: Dimensionality Reduction
type: concept
tags: [dimensionality-reduction, unsupervised-learning, compression]
difficulty: beginner
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
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

## Connections
- Builds on: [[vector-space]], [[basis-and-rank]], [[linear-mapping]]
- Leads to: [[principal-component-analysis]], [[probabilistic-pca]], [[auto-encoder]]
- Used by: data visualization, compression, feature extraction, denoising
- People: [[karl-pearson]], [[harold-hotelling]]

## Sources
- `raw/mml-book.pdf` §10 intro & §10.1, printed pp. 317–319 (PDF pp. 323–325).
