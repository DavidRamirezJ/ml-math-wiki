---
title: Karl Pearson (1857–1936)
type: person
tags: [statistics, pca, history]
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
---

# Karl Pearson (1857–1936)

> **Why they matter to ML:** He invented principal component analysis (1901) and much of the basic vocabulary of statistics — correlation, the standard deviation, the chi-squared test — that ML quietly relies on every day.

## Life (brief)
English mathematician and biometrician, founder of the world's first university statistics department
(University College London, 1911). A central figure in turning statistics into a rigorous mathematical
discipline, working alongside Francis Galton and Walter Weldon on heredity and biometry. The
Pearson correlation coefficient bears his name.

## Key contributions
- **Principal component analysis (1901).** Pearson framed PCA as the problem of fitting the best line or
  plane to a cloud of points by **minimizing the perpendicular (orthogonal) distances** — i.e. minimizing
  reconstruction error. This is exactly the [[pca-reconstruction-error|projection perspective]] of
  [[principal-component-analysis]], and the reconstruction-error objective $J_M=\frac1N\sum_n\|\mathbf x_n-\tilde{\mathbf x}_n\|^2$ traces back to him.
- **Correlation & the standard deviation.** Coined the term "standard deviation" and developed the
  product-moment correlation coefficient — foundations of the [[data-covariance-matrix]].
- **Chi-squared test** and method of moments — staples of statistical inference.

## Connections
- [[principal-component-analysis]] — co-originator (1901).
- [[pca-reconstruction-error]] — his geometric, best-fit formulation.
- [[harold-hotelling]] — later (1933) recast PCA via variance maximization.

## Sources
- `raw/mml-book.pdf` §10 intro, §10.3 (Pearson, 1901), printed pp. 317, 326 (PDF pp. 323, 332).
