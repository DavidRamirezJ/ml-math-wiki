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

## [2026-06-04] query | Animation — matrix multiplication (dot-product view)
- Built an animated GIF showing C = A·B as the row-of-A · column-of-B dot product, accumulating term by term into each target cell (uses Example 2.3 from the book).
- Tooling: pure Pillow (no ffmpeg/matplotlib available); 268 frames @ 90ms, 1040×560.
- Files: generator `assets/matrix-multiply-anim.py` (reproducible) → `assets/matrix-multiply.gif`.
- Filed back into the wiki: embedded in [[matrix]] under the 🟡 Mechanics tier.

## [2026-06-11] query | 3-min lesson — the kernel trick (rendered HTML)
Human asked to be taught something from the wiki for ~3 min. Drew from [[kernel-trick]] (§12.4).
Built self-contained HTML artifact `kernel-trick-lesson.html` with MathJax-rendered formulas
(three-tier 🟢🟡🔴 structure + "why it's free" + interactive quiz). No new wiki page created —
the concept page already exists and is complete.

## [2026-06-18] query | Atomic concepts — the irreducible foundations

Human asked which "atomic" concepts underpin everything. Measured in-degree across `wiki/concepts/` to separate *popular* pages (PCA=30, matrix=25) from *atomic* foundations. Organized the true atoms into the four pillars (linear algebra, vector calculus, probability, optimization) and an 11-item ordered shortlist. Filed as the previously-pending `wiki/maps/learning-path.md`; updated index to mark it live.

## [2026-10-13] query | MLP vs CNN
Human asked the difference between an MLP and a CNN. Not covered by the MML book (flagged on the page).
Built `explainers/2026-10-13_mlp-vs-cnn.html` (MathJax formulas with symbol-by-symbol translation, sliding-filter demo,
Toeplitz-matrix view, parameter-count slider, quiz). Filed back as `wiki/maps/mlp-vs-cnn.md`; added to index.

## [2026-10-13] ingest | Build a Large Language Model (From Scratch) — full book
Source: `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` (Raschka, Manning 2025, 370 pp.). Book page N = PDF page N+22.
Method: multi-agent workflow (247 agents). Per chapter, one note-taker per section (50 chapter sections + A.1–A.9, D.1–D.4, E.1–E.4, plus App. B and C),
each followed by an adversarial validator against the PDF text (66/69 section notes needed fixes). One consolidator deduped concepts across the book and
against existing pages and assigned one owner per page. Pages were written in batches of 6, each batch validated against the book; then one source page
and one coverage audit per unit. Staging text and notes: `.staging/llm-book/` (git-ignored).
Result: 10 source pages (`llm-ch01`…`llm-ch07`, `llm-appA`, `llm-appD`, `llm-appE`; App. B references and App. C exercise solutions folded into each),
182 new concept pages, 32 existing pages extended (no content removed), 23 new people pages. 22 mentioned-only terms were dropped by the consolidator.
New map: [[llm-learning-path]]. Index rebuilt: 278 concepts, 48 people, 21 sources.
Lint after ingest: all frontmatter valid, every concept page has all three tiers; only 4 dangling links remain (auto-encoder, gaussian-process, overview,
convolutional-neural-network), all older TODO markers.

## [2026-10-13] query | Foundations for MLP vs CNN
Human asked which foundations are needed to understand MLP vs CNN. Answered with a 7-step prerequisite chain (data as numbers →
dot product/matrix multiplication → neuron & layer → MLP → learning via backprop → overfitting/regularization → CNN-only ideas).
Filed back as a "Foundations" section in [[mlp-vs-cnn]]. Gap noted: no `convolution` / `convolutional-neural-network` concept page yet.

## [2026-10-13] query | Lesson — foundations 1-2-3 for MLP vs CNN
Built `explainers/2026-10-13_foundations-1-2-3.html`: (1) vectors/matrices/tensors with a clickable pixel-grid, (2) dot product
(weight sliders, geometric view), matrix multiplication step-through, linear map canvas, (3) a live neuron (ReLU/sigmoid) and why
activations stop layers collapsing. MathJax formulas read symbol by symbol, 5-question quiz, English vocab helper. Draws on
[[vector]], [[matrix]], [[tensor]], [[inner-product]], [[matrix-multiplication]], [[linear-mapping]], [[activation-function]], [[relu]].

## [2026-10-13] query | Tensor deep-dive + activation-function animation
Human didn't understand tensors and asked why activation functions matter. Extended `explainers/2026-10-13_foundations-1-2-3.html`:
new part 1b (cinema analogy, interactive tensor builder with 0-4 axes and index picking, colour-image stack, reshape/flatten demo,
physics-vs-ML warning) and an animation in step 3 (a 1-hidden-layer ReLU net adds one bend per neuron to fit a curve; with ReLU off,
the sum of neurons stays one straight line). Added 3 quiz questions and vocab entries. Relates to [[tensor]], [[activation-function]], [[relu]].

## [2026-10-13] query | Función de activación — por qué importa, dónde va (HTML animado, español)
Human asked (in Spanish) why activation functions matter and whether they apply per layer or per perceptron. Built
`explainers/2026-10-13_funcion-de-activacion.html`: animated layer (each neuron computes z then σ(z); one σ chosen per layer),
collapse animation W2·W1 = W' with a live "does ReLU break the collapse?" check, XOR plane-folding animation (Goodfellow et al. §6.1
network) with the final decision band, hinge-curve builder, per-neuron vs per-layer cards + softmax exception with live bars,
which-layer table, animated activation gallery, perceptron→MLP history, where it appears (MLP/CNN/GPT), 6-question quiz.
Relates to [[activation-function]], [[relu]], [[gelu]], [[softmax]], [[multilayer-perceptron]], [[feed-forward-network]].

## [2026-10-15] query | Backpropagation explainer (HTML, animated)
Human asked to explain backprop (open item from the 13 Oct session). Built `explainers/2026-10-15_backpropagation.html`: tiny
1-hidden-ReLU net with a 10-step forward/backward/update animation (sliders recompute every value), chain-rule "nudge ripple"
check, reuse-trick comparison, live training loss chart, matrix form δ_prev = (Wᵀδ) ⊙ σ′(z), 5-question quiz, vocab helper.
No new wiki page; content draws on [[backpropagation]], [[chain-rule]], [[relu]], [[gradient-descent]].

## [2026-10-27] query | Atoms, molecules, compounds + 8 first principles
- Asked: what are the atoms and molecules needed to understand everything, so forgotten formulas can be re-derived from principles.
- Method: computed downstream reach over the `Builds on:` graph of all 278 concept pages (vector 276, vector-space 264, matrix 255, … loss-function 165).
- Created [[atoms-and-molecules]] (map): 8 principles with worked re-derivations, 15 atoms, 16 molecules with formulas, 8 compounds. Added to [[index]].
- Explainer: `explainers/2026-10-27_atoms-and-molecules.html` (clickable atom→molecule→compound map with animated bonds, re-derivation drills, checklist, quiz, vocab helper).

## [2026-10-27] query | Validate: are the atoms/principles the most elemental?
- Result: no — 4 of the 8 working principles are derived (chain rule, gradient descent, mean/variance, eigenvectors). The irreducible bottom is 5 axioms: linearity, inner product, local linearity, probability rules (sum + product), learning = optimization. The sum rule and axiom E were missing.
- Updated [[atoms-and-molecules]] with a validation section; explainer `explainers/2026-10-27_atoms-and-molecules.html` §3.
- Graph fix: [[activation-function]] no longer lists [[neural-network]] under `Builds on:` (broke a cycle). TODO: floor-level `logarithm-and-exponential` page.

## [2026-10-27] query | Video series — one video per ML math atom
- Made 16 narrated Manim videos (`explainers/atoms-series/NN-*/final.mp4`, 1080p, subtitles, chapters), 3.2 h total, one per study atom from [[atoms-and-molecules]]; each: hook, ≥2 worked examples, try-it-yourself, real NumPy demo, summary, credits (David R. · davidramirezj.com).
- Pilot 01-vector approved by the user; 02–16 produced in parallel from `series_guide.md`. All finals: 0 layout warnings, no audio gaps; 52 glossary terms, no duplicates.
- Index: `explainers/atoms-series/index.html`. Linked from [[atoms-and-molecules]].
