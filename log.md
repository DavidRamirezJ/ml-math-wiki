# 📜 Log

Append-only, chronological. Newest at the bottom. Each entry prefix: `## [YYYY-MM-DD] <op> | <title>`.
Recent activity: `grep "^## \[" log.md | tail -5`.

---

## [2026-06-04] setup | Wiki initialized
- Created schema `CLAUDE.md` (three-tier dummy→expert concept structure, ingest/query/lint workflows).
- Created `index.md` (catalog) and `log.md` (this file).
- Folder layout: `raw/`, `assets/`, `wiki/{maps,concepts,people,sources}`.
- Initialized git repo.
- Moved `mml-book.pdf` → `raw/mml-book.pdf` (Mathematics for ML, Deisenroth/Faisal/Ong 2020, 417 pp).
- Next: first ingest — Chapter 2, Linear Algebra.

## [2026-06-04] ingest | MML Ch.2 — Linear Algebra
- Source: `raw/mml-book.pdf` §2, pp. 17–73. Extracted text via pypdf (poppler also now installed).
- Created source page [[mml-ch02-linear-algebra]].
- Created 3 full three-tier concept pages: [[vector]], [[vector-space]], [[system-of-linear-equations]].
- Created person page [[carl-friedrich-gauss]].
- Seeded 6 concept stubs (matrix, gaussian-elimination, linear-independence, basis-and-rank, linear-mapping, affine-space) — to be filled on next pass; lint will flag.
- Updated [[index]] (counts: 1 source, 3 concepts +6 stubs, 1 person).
- No contradictions (first content ingest).

## [2026-06-04] lint | Validate Ch.2 pages against source
- Re-read §2.1, §2.4 (Def 2.7–2.9) directly from `raw/mml-book.pdf` to fact-check claims.
- ✅ Vector-space axioms in [[vector-space]] match **Definition 2.9** verbatim (Abelian group + 2 distributivity laws + outer associativity + 1·x=x).
- ✅ Trichotomy (no/one/infinite) confirmed printed p.21; ✅ Gauss attribution confirmed (§2.3–2.4).
- ❌ FIXED: source page claimed Ch.2 = "pp. 17–73"; actual = **printed pp. 17–69** (Ch.3 starts p.70). Established page rule: printed = PDF index − 5.
- ➕ Added a **section-coverage table** to [[mml-ch02-linear-algebra]] proving all of §2.1–2.9 are represented (3 full pages ✅, 6 stubs 🌱).
- Open gap: 6 stub concepts (§2.2, 2.3, 2.5, 2.6, 2.7, 2.8) still need full three-tier pages.

## [2026-06-04] ingest | Fill 6 Ch.2 concept stubs (§2.2, 2.3, 2.5, 2.6, 2.7, 2.8)
- Read all six sections directly from `raw/mml-book.pdf` (PDF pp. 27–67) before writing — every formula/definition sourced.
- Promoted to full three-tier pages: [[matrix]] (Defs 2.1–2.5), [[gaussian-elimination]] (Def 2.6, REF/RREF, minus-1 trick), [[linear-independence]] (Defs 2.11–2.12), [[basis-and-rank]] (Defs 2.13–2.14, rank §2.6.2), [[linear-mapping]] (Defs 2.15–2.19, Thm 2.17), [[affine-space]] (Defs 2.25–2.26).
- Updated [[mml-ch02-linear-algebra]] coverage table → all §2.1–2.8 now ✅; corrected internal section page ranges.
- Updated [[index]] (counts: 1 source, 9 concepts, 1 person). Removed stub markers.
- Chapter 2 is now fully fulfilled (only §2.9 further-reading left as pointers). No contradictions.

## [2026-06-04] ingest | Chapters 3–12 (full book, batch via 10-agent workflow)
- Ingested all remaining chapters in parallel (one agent per chapter, each reading its own PDF page range, writing source + concept + person pages per the schema).
- **Ch.3 Analytic Geometry:** norm, inner-product, metric, angle, orthogonality, orthogonal-matrix, orthonormal-basis, orthogonal-complement, orthogonal-projection, gram-schmidt, rotation, SPD-matrix (+ Cauchy, Schwarz, Gram, Schmidt).
- **Ch.4 Matrix Decompositions:** determinant, trace, characteristic-polynomial, eigenvalues-and-eigenvectors, spectral-theorem, cholesky, eigendecomposition, SVD, low-rank-approximation (+ Cholesky, Laplace, Eckart, Young).
- **Ch.5 Vector Calculus:** derivative, partial-derivative, gradient, jacobian, chain-rule, hessian, taylor-series, backpropagation, automatic-differentiation (+ Taylor, Jacobi, Hesse).
- **Ch.6 Probability:** random-variable, probability-distribution, expected-value-and-mean, covariance-and-variance, statistical-independence, bayes-theorem, gaussian-distribution, conjugate-prior, exponential-family, sufficient-statistics, change-of-variables (+ Bayes, Fisher, Jensen).
- **Ch.7 Optimization:** gradient-descent, stochastic-gradient-descent, convex-optimization, convex-conjugate, lagrange-multipliers, linear-programming, quadratic-programming (+ Lagrange, Fenchel, Abel).
- **Ch.8 When Models Meet Data:** empirical-risk-minimization, loss-function, MLE, MAP, overfitting-and-underfitting, regularization, cross-validation, directed-graphical-models, bayesian-model-selection, marginal-likelihood (+ Occam).
- **Ch.9 Linear Regression:** linear-regression, regularized-least-squares, bayesian-linear-regression, generalized-linear-model, overfitting (+ Legendre, Bayes).
- **Ch.10 PCA:** principal-component-analysis, pca-maximum-variance, pca-reconstruction-error, data-covariance-matrix, power-iteration, probabilistic-pca, dimensionality-reduction (+ Pearson, Hotelling).
- **Ch.11 GMM:** gaussian-mixture-model, mixture-model, latent-variable-model, expectation-maximization, responsibility, kernel-density-estimation, histogram (+ Dempster, Parzen).
- **Ch.12 SVM:** support-vector-machine, separating-hyperplane, margin, hard-margin-svm, soft-margin-svm, hinge-loss, dual-support-vector-machine, kernel-trick, support-vector (+ Vapnik).
- Rebuilt [[index]] programmatically from frontmatter: **11 sources, 96 concepts, 25 people.**
- Lint: fixed 1 alias collision (overfitting vs overfitting-and-underfitting — kept both, cross-linked; Ch.8=general, Ch.9=regression case study). Fixed dangling aliases (gram-schmidt→gram-schmidt-orthogonalization, positive-definite-matrix→symmetric-positive-definite-matrix, expected-value→expected-value-and-mean) and 2 trailing-backslash link artifacts. Link check: 0 real danglers (only sanctioned future pages: overview, learning-path, auto-encoder, gaussian-process).
- Whole book (Part I + Part II) now ingested. Ch.1 left as motivation only.
