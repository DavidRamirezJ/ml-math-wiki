---
title: Harold Hotelling (1895–1973)
type: person
tags: [statistics, pca, history, economics]
sources: [mml-ch10-dimensionality-reduction-pca]
created: 2026-06-04
updated: 2026-06-04
---

# Harold Hotelling (1895–1973)

> **Why they matter to ML:** He gave principal component analysis its modern, variance-maximizing formulation (1933) — the derivation that makes PCA an eigenvalue problem on the covariance matrix.

## Life (brief)
American mathematical statistician and economist, professor at Columbia and later the University of
North Carolina. Influential across statistics (Hotelling's $T^2$ test, canonical correlation analysis)
and mathematical economics (Hotelling's law, the Hotelling rule for exhaustible resources). A key bridge
between rigorous statistical theory and its applications.

## Key contributions
- **Principal component analysis (1933).** Independently of Pearson, Hotelling derived PCA as the problem
  of finding directions that **maximize the variance** of the projected data. This is the
  [[pca-maximum-variance|maximum-variance perspective]] of [[principal-component-analysis]] and the source of
  the result "retaining most information = capturing the largest variance," with variance along a principal
  axis equal to its eigenvalue. He coined the term *principal components*.
- **Canonical correlation analysis (1936)** — finds maximally-correlated linear combinations of two variable
  sets; a relative of PCA still used in multi-view ML.
- **Hotelling's $T^2$** — the multivariate generalization of the $t$-test.

## Connections
- [[principal-component-analysis]] — co-originator (1933), variance-maximization view.
- [[pca-maximum-variance]] — his formulation.
- [[data-covariance-matrix]] — the object whose top eigenvectors are the principal components.
- [[karl-pearson]] — earlier (1901) reconstruction-error formulation of PCA.

## Sources
- `raw/mml-book.pdf` §10 intro, §10.2 (Hotelling, 1933), printed pp. 317, 320 (PDF pp. 323, 326).
