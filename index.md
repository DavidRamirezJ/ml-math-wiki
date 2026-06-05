---
title: Index
type: map
updated: 2026-06-04
---

# 📇 Wiki Index

The catalog of everything in this wiki. Read this first when answering a query, then drill into the linked pages. Updated on every ingest. See [[CLAUDE]] for the schema.

**Counts:** 11 sources · 96 concepts · 25 people · 1 map (+2 pending)

---

## 🗺️ Maps
- [[index]] — this catalog
- [[overview]] — what this wiki covers and how it's organized *(pending)*
- [[learning-path]] — the dummy → expert route through ML math *(pending)*

## 🟦 Concepts
*One page per mathematical concept — all full three-tier (🟢🟡🔴) pages. 96 total, alphabetical.*

- [[affine-space]] — An affine space is a vector subspace that's been picked up and shifted off the origin — a flat that no longer passes through zero. *(advanced)*
- [[angle-between-vectors]] — The angle between two vectors measures how aligned their directions are, computed from their inner product. *(beginner)*
- [[automatic-differentiation]] — Automatic differentiation computes the exact gradient of any program by breaking it into elementary steps and chaining their kn… *(advanced)*
- [[backpropagation]] — Backpropagation computes a neural network's gradients cheaply by applying the chain rule backward through the network, reusing… *(advanced)*
- [[basis-and-rank]] — A basis is the smallest set of vectors that can build the whole space; its size is the *dimension*, and a matrix's *rank* count… *(intermediate)*
- [[bayes-theorem]] — Bayes' theorem is the rule for updating what you believe after you see new evidence. *(intermediate)*
- [[bayesian-inference]] — Instead of committing to one best setting of the model, keep *every* plausible setting weighted by how well it explains the dat… *(advanced)*
- [[bayesian-linear-regression]] — Instead of choosing one best curve, keep *all* curves that are consistent with the data — weighted by how plausible each is — a… *(advanced)*
- [[bayesian-model-selection]] — Choose between competing models by how well each one *predicted the data you actually got* — a rule that automatically prefers… *(advanced)*
- [[chain-rule]] — The chain rule says: to differentiate a function inside a function, multiply the outer slope by the inner slope. *(intermediate)*
- [[change-of-variables-for-distributions]] — When you transform a random variable, this is the rule that tells you the shape of the new distribution. *(advanced)*
- [[characteristic-polynomial]] — It's a special polynomial built from a square matrix whose roots are exactly the matrix's eigenvalues. *(intermediate)*
- [[cholesky-decomposition]] — It's the "square root" of a nice symmetric matrix — it splits the matrix into a triangular factor times its mirror image. *(intermediate)*
- [[conjugate-prior]] — A conjugate prior is a prior chosen so that the posterior comes out as the same kind of distribution — so updating beliefs is j… *(intermediate)*
- [[convex-conjugate]] — Describe a bowl-shaped curve not by its points but by all the slanted lines that just touch it from below — then "slope" become… *(advanced)*
- [[convex-optimization]] — If your landscape is a single smooth bowl, then any valley you find is *the* lowest point — so optimization becomes safe and th… *(advanced)*
- [[covariance-and-variance]] — Variance measures how spread out a random quantity is; covariance measures whether two quantities move together. *(intermediate)*
- [[cross-validation]] — Cross-validation estimates how well a model will do on new data by repeatedly hiding part of your data, training on the rest, a… *(beginner)*
- [[data-covariance-matrix]] — A square table that records, for a dataset, how much each pair of features spreads and moves together — the object whose bigges… *(intermediate)*
- [[derivative]] — The derivative tells you how fast a function is changing at a point — the steepness of its graph right there. *(beginner)*
- [[determinant]] — The determinant is a single number that tells you how much a matrix stretches or shrinks volume — and if it's zero, the matrix… *(intermediate)*
- [[dimensionality-reduction]] — Squeezing data with lots of numbers per item down to just a few numbers, keeping almost everything that matters and throwing aw… *(beginner)*
- [[directed-graphical-models]] — A picture of arrows-between-circles that tells you how a big probability distribution breaks into a product of small ones — and… *(advanced)*
- [[dual-support-vector-machine]] — The dual SVM solves the same problem from the data's point of view — one weight per training example instead of one per feature… *(advanced)*
- [[eigendecomposition]] — If you describe a matrix using its own special stretch-directions as the coordinate axes, the matrix becomes a simple diagonal… *(advanced)*
- [[eigenvalues-and-eigenvectors]] — An eigenvector is a special direction that a matrix only stretches or shrinks (never rotates off course), and its eigenvalue is… *(intermediate)*
- [[empirical-risk-minimization]] — Pick a family of guess-functions, score how badly each one does on your examples, and keep the one with the smallest average mi… *(intermediate)*
- [[expectation-maximization]] — EM fits models with hidden groupings by guessing the hidden labels, updating the model to match, and repeating until it stops i… *(advanced)*
- [[expected-value-and-mean]] — The expected value is the long-run average outcome you'd get if you repeated a random experiment forever. *(beginner)*
- [[exponential-family]] — The exponential family is the single template that most "named" distributions fit into — and the reason they're so easy to comp… *(advanced)*
- [[gaussian-distribution]] — The Gaussian (bell curve) is the distribution that shows up whenever many small random effects add together. *(intermediate)*
- [[gaussian-elimination]] — A bookkeeping recipe of row operations that grinds any linear system into a "staircase" form you can read the answer straight o… *(intermediate)*
- [[gaussian-mixture-model]] — A Gaussian mixture model fits data by stacking several bell curves on top of each other, and learns where each bell goes by rep… *(intermediate)*
- [[generalized-linear-model]] — Take a linear regression and squash its output through a fixed nonlinear function, so you can predict things like probabilities… *(intermediate)*
- [[gradient]] — The gradient is the list of all a function's partial slopes, bundled into one vector that points in the direction the function… *(intermediate)*
- [[gradient-descent]] — To find the bottom of a valley you can't see, keep taking small steps in whatever direction is most steeply downhill from where… *(beginner)*
- [[gram-schmidt-orthogonalization]] — Gram–Schmidt is a step-by-step recipe that straightens any set of basis vectors into mutually perpendicular ones. *(intermediate)*
- [[hard-margin-svm]] — The hard-margin SVM is the strictest version of the max-margin classifier: every single point must stay outside the buffer zone… *(advanced)*
- [[hessian]] — The Hessian is the table of a function's second slopes — it measures how the surface curves, i.e. how bowl-shaped or saddle-sha… *(intermediate)*
- [[hinge-loss]] — The hinge loss charges nothing once a point is comfortably on the correct side of the line, and a growing fine the closer it ge… *(intermediate)*
- [[histogram]] — A histogram chops your data's range into bins and draws a bar for each bin as tall as the number of points that fell inside it. *(beginner)*
- [[inner-product]] — An inner product is a way to multiply two vectors into a single number that tells you how much they point in the same direction. *(intermediate)*
- [[jacobian]] — The Jacobian is the full grid of partial slopes for a function that outputs many numbers — the matrix version of a derivative. *(intermediate)*
- [[kernel-density-estimation]] — Kernel density estimation draws a small smooth bump on top of every data point and adds them all up to get a smooth picture of… *(intermediate)*
- [[kernel-trick]] — The kernel trick lets a straight-line method draw curved boundaries by secretly measuring similarity in a richer space — withou… *(advanced)*
- [[lagrange-multipliers]] — When you must minimize something while obeying rules, attach a price tag to breaking each rule and fold those prices into the o… *(advanced)*
- [[latent-variable-model]] — A latent-variable model imagines an unseen "which-group" switch behind each data point, and recovers the data's distribution by… *(advanced)*
- [[linear-independence]] — A set of vectors is linearly independent when none of them is redundant — you can't build any one out of the others. *(intermediate)*
- [[linear-mapping]] — A linear mapping is a function between vector spaces that respects adding and scaling — and every such function is just a matri… *(advanced)*
- [[linear-programming]] — Push a straight cost as low as possible while staying inside a region carved out by straight walls — the best answer always sit… *(intermediate)*
- [[linear-regression]] — Linear regression draws the best line (or curve) through a cloud of noisy data points so you can predict the value at a new point. *(beginner)*
- [[loss-function]] — A loss function is the rule that turns "how wrong was this one prediction?" into a single non-negative number you can try to ma… *(beginner)*
- [[low-rank-approximation]] — Keep only a matrix's few biggest "stretch" pieces and you get the best possible simplified version of it — a compressed copy th… *(advanced)*
- [[margin]] — The margin is the width of the empty buffer zone between the dividing line and the nearest data point on either side. *(intermediate)*
- [[marginal-likelihood]] — The probability that a whole model — averaged over every parameter setting it could take — would have produced your data; it's… *(advanced)*
- [[matrix]] — A matrix is a rectangular grid of numbers — and that grid can both *store a system of equations* and *act as a function that tr… *(beginner)*
- [[maximum-a-posteriori-estimation]] — Like maximum likelihood, but you also bring in a prior belief about which parameter values are reasonable, then pick the parame… *(intermediate)*
- [[maximum-likelihood-estimation]] — Pick the parameter values that make the data you actually observed as unsurprising as possible. *(intermediate)*
- [[metric]] — A metric is a rule for measuring how far apart two points are, obeying the common-sense laws of distance. *(beginner)*
- [[mixture-model]] — A mixture model builds a complicated bumpy probability curve by blending several simple ones together, each weighted by how muc… *(beginner)*
- [[norm]] — A norm is a rule that measures how "long" a vector is — a single non-negative number standing in for its size. *(beginner)*
- [[orthogonal-complement]] — The orthogonal complement of a subspace is everything that is perpendicular to all of it — together they fill the whole space w… *(intermediate)*
- [[orthogonal-matrix]] — An orthogonal matrix is a square matrix whose columns are mutually perpendicular unit vectors — it rotates or reflects space wi… *(intermediate)*
- [[orthogonal-projection]] — An orthogonal projection drops a vector straight down onto a subspace, finding the closest point in that subspace. *(advanced)*
- [[orthogonality]] — Two vectors are orthogonal when their inner product is zero — the precise, general meaning of "at right angles." *(beginner)*
- [[orthonormal-basis]] — An orthonormal basis is a set of coordinate directions that are all mutually perpendicular and exactly one unit long — the clea… *(intermediate)*
- [[overfitting]] — A model overfits when it memorizes the noise in the training data instead of learning the real pattern, so it predicts badly on… *(beginner)*
- [[overfitting-and-underfitting]] — Overfitting is memorizing the practice questions instead of learning the subject; underfitting is being too simple to learn it… *(beginner)*
- [[partial-derivative]] — A partial derivative measures how a multi-input function changes when you nudge just one input and freeze all the others. *(beginner)*
- [[pca-maximum-variance]] — One way to derive PCA: pick the directions that make the compressed data spread out as much as possible, because spread = infor… *(advanced)*
- [[pca-reconstruction-error]] — A second way to derive PCA: find the low-dimensional subspace whose shadows of the data points are, on average, as close to the… *(advanced)*
- [[power-iteration]] — A dead-simple loop — multiply by the matrix, normalize, repeat — that homes in on the single most dominant direction of a matrix. *(intermediate)*
- [[principal-component-analysis]] — PCA finds the few directions in which your data spreads out the most and describes every data point using only its position alo… *(intermediate)*
- [[probabilistic-pca]] — PCA rebuilt as a probability story — each data point is born from a few hidden random numbers, stretched by a matrix and sprink… *(advanced)*
- [[probability-distribution]] — A probability distribution is the complete bookkeeping of how likely each possible value of a random variable is. *(beginner)*
- [[quadratic-programming]] — Minimize a bowl-shaped cost while staying inside straight walls — the same as a linear program, but now the cost itself curves… *(intermediate)*
- [[random-variable]] — A random variable is a rule that turns the messy outcome of a random experiment into a number you actually care about. *(beginner)*
- [[regularization]] — Regularization adds a "keep it simple" penalty to training so the model fits the data without contorting itself into something… *(intermediate)*
- [[regularized-least-squares]] — Plain least squares but with an extra penalty that keeps the parameters small, so the fit stays smooth and doesn't chase noise. *(intermediate)*
- [[responsibility]] — A responsibility is the share of "credit" each cluster gets for producing a given data point — like guessing how likely it is t… *(intermediate)*
- [[rotation]] — A rotation spins vectors around a fixed point by an angle, changing their direction but never their length. *(intermediate)*
- [[separating-hyperplane]] — A separating hyperplane is a flat dividing wall that puts one class on one side and the other class on the other. *(intermediate)*
- [[singular-value-decomposition]] — Any matrix at all — square or not — is really just "rotate, then stretch each axis by a positive amount, then rotate again." *(advanced)*
- [[soft-margin-svm]] — The soft-margin SVM is the practical SVM that still wants a wide buffer zone but lets a few points sneak inside or onto the wro… *(advanced)*
- [[spectral-theorem]] — Every symmetric matrix can be fully described by a set of perpendicular stretch-directions with plain real stretch-factors — no… *(advanced)*
- [[statistical-independence]] — Two things are independent if learning one tells you nothing new about the other. *(intermediate)*
- [[stochastic-gradient-descent]] — Instead of measuring the whole hillside before each step, you sample a few spots, guess the downhill direction, and step — much… *(intermediate)*
- [[sufficient-statistics]] — A sufficient statistic is a summary of your data that captures everything the data can tell you about an unknown parameter — so… *(advanced)*
- [[support-vector]] — Support vectors are the handful of training points sitting right on the edge of the buffer zone — they alone determine the divi… *(intermediate)*
- [[support-vector-machine]] — A support vector machine draws the straightest dividing line between two groups of dots so that the line sits as far as possibl… *(intermediate)*
- [[symmetric-positive-definite-matrix]] — A symmetric positive-definite matrix is a square matrix that always returns a positive number when you sandwich any nonzero vec… *(intermediate)*
- [[system-of-linear-equations]] — Several straight-line equations that must all be true at once — and you hunt for the values that satisfy every one of them. *(beginner)*
- [[taylor-series]] — A Taylor series rebuilds a smooth function as an infinite sum of polynomial pieces, using the function's slopes (derivatives) a… *(intermediate)*
- [[trace]] — The trace is just the sum of the numbers running down a square matrix's main diagonal — a tiny summary that doesn't care which… *(beginner)*
- [[vector]] — A vector is anything you can *add to another of its kind* and *stretch by a number* and still get the same kind of thing back. *(beginner)*
- [[vector-space]] — A vector space is the complete collection of every vector you could ever reach by adding and scaling a starting set — nothing e… *(intermediate)*

## 👤 People
*Mathematicians behind the math. 25 total, alphabetical by first name.*

- [[adrien-marie-legendre]] — The Legendre transform re-describes a convex function by its slopes — the basis of the convex conjugate and a key dua…
- [[andre-louis-cholesky]] — his matrix factorization is the standard, numerically stable way to handle covariance matrices — used everywhere from…
- [[arthur-dempster]] — Co-author of the 1977 paper that named and unified the **EM algorithm**, the workhorse for fitting latent-variable an…
- [[augustin-louis-cauchy]] — The Cauchy–Schwarz inequality is what makes "the angle between two vectors" — and thus cosine similarity — well-defined.
- [[brook-taylor]] — His series expansion is how we replace any smooth function by a polynomial — the basis of linearization, Newton's met…
- [[carl-eckart]] — with Gale Young he proved that the truncated SVD is the optimal low-rank approximation of a matrix — the theorem behi…
- [[carl-friedrich-gauss]] — his name is on two pillars of the field — **Gaussian elimination** (solving linear systems) and the **Gaussian distri…
- [[carl-gustav-jacob-jacobi]] — The matrix of first-order partial derivatives — the Jacobian — is named for him, and it is the central object for dif…
- [[emanuel-parzen]] — He formalized kernel density estimation — the "Parzen window" — a foundational nonparametric method for estimating a…
- [[erhard-schmidt]] — Co-creator of Gram–Schmidt orthogonalization and a founder of Hilbert-space theory — the infinite-dimensional inner-p…
- [[gale-young]] — co-author of the theorem proving that the truncated SVD gives the best low-rank approximation — the optimality guaran…
- [[harold-hotelling]] — He gave principal component analysis its modern, variance-maximizing formulation (1933) — the derivation that makes P…
- [[hermann-schwarz]] — His half of the Cauchy–Schwarz inequality extends the angle/similarity bound to function spaces — the setting kernels…
- [[johan-jensen]] — Jensen's inequality is the defining property of convex functions and appears throughout ML — from convex losses to th…
- [[jorgen-pedersen-gram]] — The Gram–Schmidt process and the Gram matrix bear his name — orthonormalization and the kernel/covariance matrices ce…
- [[joseph-louis-lagrange]] — His multipliers are the standard tool for optimizing under constraints — the foundation of SVM duals and constrained…
- [[karl-pearson]] — He invented principal component analysis (1901) and much of the basic vocabulary of statistics — correlation, the sta…
- [[ludwig-otto-hesse]] — The matrix of second-order partial derivatives — the Hessian — is named after him; it encodes curvature and powers se…
- [[niels-henrik-abel]] — His impossibility theorem explains why we *must* optimize numerically — there's no algebraic formula for the roots of…
- [[pierre-simon-laplace]] — his cofactor expansion is the textbook recursive way to compute determinants, and his broader work on probability and…
- [[ronald-fisher]] — he formalized **maximum likelihood estimation**, the principle behind how most probabilistic ML models are trained.
- [[thomas-bayes]] — His theorem turns a prior belief plus observed data into a posterior belief — the engine of probabilistic inference,…
- [[vladimir-vapnik]] — co-founder of **statistical learning theory** and the **support vector machine**, and the person who popularized **em…
- [[werner-fenchel]] — He turned the Legendre transform into the general convex conjugate that works even for non-smooth losses — the modern…
- [[william-of-occam]] — his namesake principle — prefer the simplest explanation that fits — is the guiding intuition of **model selection**.

## 📄 Sources
*One summary page per ingested chapter. 11 of 12 chapters (Ch.1 is motivation only).*

- [[mml-ch02-linear-algebra]] — MML Ch.2 — Linear Algebra
- [[mml-ch03-analytic-geometry]] — MML Ch. 3 — Analytic Geometry
- [[mml-ch04-matrix-decompositions]] — MML Ch.4 — Matrix Decompositions
- [[mml-ch05-vector-calculus]] — MML Ch. 5 — Vector Calculus
- [[mml-ch06-probability-and-distributions]] — MML Ch.6 — Probability and Distributions
- [[mml-ch07-continuous-optimization]] — MML Ch. 7 — Continuous Optimization
- [[mml-ch08-when-models-meet-data]] — MML Ch.8 — When Models Meet Data
- [[mml-ch09-linear-regression]] — MML Ch. 9 — Linear Regression
- [[mml-ch10-dimensionality-reduction-pca]] — MML Ch.10 — Dimensionality Reduction with Principal Component Analysis
- [[mml-ch11-density-estimation-gaussian-mixture-models]] — MML Ch. 11 — Density Estimation with Gaussian Mixture Models
- [[mml-ch12-classification-with-support-vector-machines]] — MML Ch. 12 — Classification with Support Vector Machines

---

> At ~130 pages this index + `grep` is still adequate. If it grows much further, add a search tool (e.g. `qmd`).