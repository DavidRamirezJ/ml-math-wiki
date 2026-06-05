---
title: "MML Ch.8 — When Models Meet Data"
type: source
tags: [machine-learning, foundations, probabilistic-modeling, optimization]
sources: []
created: 2026-06-04
updated: 2026-06-04
---

# When Models Meet Data (MML Chapter 8)

- **Raw file:** `raw/mml-book.pdf`, Chapter 8, printed pp. 251–288 (PDF pp. 257–294).
- **Type:** book chapter
- **Ingested:** 2026-06-04

This is the opening chapter of **Part II** of *Mathematics for Machine Learning*. Part I built the
math (linear algebra, geometry, calculus, probability, optimization); Part II spends it on the
"four pillars": regression (Ch. 9), dimensionality reduction (Ch. 10), density estimation (Ch. 11),
classification (Ch. 12). Chapter 8 sets up the vocabulary and the three "flavors of learning" that
the rest of Part II instantiates.

## Section coverage

| Section | Title | Wiki page(s) |
|---|---|---|
| 8.1 | Data, Models, and Learning | data as [[vector]]s; predictor = [[linear-mapping|affine function]]; three phases (predict / train / model-select) |
| 8.1.1 | Data as Vectors | [[vector]] (linked, owned by Ch. 2) |
| 8.1.2 | Models as Functions | predictor $f(\mathbf x)=\boldsymbol\theta^\top\mathbf x+\theta_0$ → [[empirical-risk-minimization]] |
| 8.1.3 | Models as Probability Distributions | → [[maximum-likelihood-estimation]] |
| 8.1.4 | Learning is Finding Parameters | overview of ERM / MLE / Bayesian inference |
| 8.2 | Empirical Risk Minimization | [[empirical-risk-minimization]] |
| 8.2.1 | Hypothesis Class of Functions | [[empirical-risk-minimization]] |
| 8.2.2 | Loss Function for Training | [[loss-function]] |
| 8.2.3 | Regularization to Reduce Overfitting | [[regularization]], [[overfitting-and-underfitting]] |
| 8.2.4 | Cross-Validation | [[cross-validation]] |
| 8.3 | Parameter Estimation | [[maximum-likelihood-estimation]], [[maximum-a-posteriori-estimation]] |
| 8.3.1 | Maximum Likelihood Estimation | [[maximum-likelihood-estimation]] |
| 8.3.2 | Maximum A Posteriori Estimation | [[maximum-a-posteriori-estimation]] |
| 8.3.3 | Model Fitting | [[overfitting-and-underfitting]] |
| 8.4 | Probabilistic Modeling and Inference | [[bayesian-inference]], probabilistic models |
| 8.4.1 | Probabilistic Models | joint distribution as the central object → [[bayesian-inference]] |
| 8.4.2 | Bayesian Inference | [[bayesian-inference]] |
| 8.4.3 | Latent-Variable Models | [[bayesian-inference]] (latent-variable section) |
| 8.5 | Directed Graphical Models | [[directed-graphical-models]] |
| 8.5.1 | Graph Semantics | [[directed-graphical-models]] |
| 8.5.2 | Conditional Independence and d-Separation | [[directed-graphical-models]] |
| 8.6 | Model Selection | [[bayesian-model-selection]] |
| 8.6.1 | Nested Cross-Validation | [[cross-validation]] (nested section), [[bayesian-model-selection]] |
| 8.6.2 | Bayesian Model Selection | [[bayesian-model-selection]] |
| 8.6.3 | Bayes Factors for Model Comparison | [[bayesian-model-selection]] |

## TL;DR (5 bullets)
- **Data become vectors**, models become **functions** $f(\mathbf x)=\boldsymbol\theta^\top\mathbf x+\theta_0$ or **probability distributions**, and **learning = finding parameters $\boldsymbol\theta$** that generalize to unseen data. Three algorithmic phases: prediction/inference, training, and model selection.
- **Empirical risk minimization (ERM)** is the "probability-free" recipe: pick a hypothesis class, pick a [[loss-function]], and minimize the **average training loss** $R_{\text{emp}}=\frac1N\sum_n\ell(y_n,\hat y_n)$ — but the real target is the **expected risk** $R_{\text{true}}=\mathbb E[\ell(y,f(\mathbf x))]$, which forces [[regularization]] and [[cross-validation]].
- **Maximum likelihood estimation (MLE)** is the probabilistic twin of ERM: minimize the **negative log-likelihood** $-\sum_n\log p(y_n\mid\mathbf x_n,\boldsymbol\theta)$. A **Gaussian likelihood gives exactly least squares.** Adding a parameter prior $p(\boldsymbol\theta)$ gives **MAP estimation**, the probabilistic twin of regularization.
- **Bayesian inference** goes further: don't pick one $\boldsymbol\theta^*$, keep the whole **posterior** $p(\boldsymbol\theta\mid\mathcal X)$ and **integrate** parameters out when predicting. Estimation becomes *optimization*; Bayesian inference becomes *integration*.
- **Directed graphical models** (Bayesian networks) draw the factorization of the joint distribution; **model selection** trades fit against complexity, formalized by **cross-validation** and by the **marginal likelihood / evidence** (an automatic Occam's razor).

## Key concepts introduced
- [[empirical-risk-minimization]] — minimize average training loss over a hypothesis class
- [[loss-function]] — how a single prediction's error is scored (e.g. squared loss → least squares)
- [[regularization]] — penalty term that biases toward simpler parameters to fight overfitting
- [[overfitting-and-underfitting]] — the generalization gap between training and test risk
- [[cross-validation]] — estimate expected risk by held-out validation; nested version selects models
- [[maximum-likelihood-estimation]] — minimize negative log-likelihood; Gaussian case = least squares
- [[maximum-a-posteriori-estimation]] — MLE plus a parameter prior; the probabilistic regularizer
- [[bayesian-inference]] — keep the full posterior and marginalize, instead of a point estimate
- [[directed-graphical-models]] — graphs encoding the factorization and (conditional) independence of a joint distribution
- [[bayesian-model-selection]] — Occam's razor via the model evidence / marginal likelihood and Bayes factors

## Notable results / theorems
- **Eq. (8.6) empirical risk:** $R_{\text{emp}}(f,X,\mathbf y)=\frac1N\sum_{n=1}^N \ell(y_n,\hat y_n)$.
- **Eq. (8.9) least-squares problem:** $\min_{\boldsymbol\theta}\frac1N\lVert \mathbf y - X\boldsymbol\theta\rVert^2$.
- **Eq. (8.12) regularized least squares:** $\min_{\boldsymbol\theta}\frac1N\lVert \mathbf y - X\boldsymbol\theta\rVert^2 + \lambda\lVert\boldsymbol\theta\rVert^2$ (Tikhonov regularization).
- **Eq. (8.17) negative log-likelihood:** $\mathcal L(\boldsymbol\theta)=-\sum_{n=1}^N\log p(y_n\mid\mathbf x_n,\boldsymbol\theta)$; for a Gaussian likelihood this reduces to least squares (Example 8.5).
- **Eq. (8.19)/(8.20) MAP via Bayes:** $p(\boldsymbol\theta\mid\mathbf x)\propto p(\mathbf x\mid\boldsymbol\theta)\,p(\boldsymbol\theta)$.
- **Eq. (8.22)/(8.23) Bayesian posterior & predictive:** $p(\boldsymbol\theta\mid\mathcal X)=\frac{p(\mathcal X\mid\boldsymbol\theta)p(\boldsymbol\theta)}{p(\mathcal X)}$, $\;p(\mathbf x)=\int p(\mathbf x\mid\boldsymbol\theta)p(\boldsymbol\theta)\,d\boldsymbol\theta$.
- **Eq. (8.31) DGM factorization:** $p(\mathbf x)=\prod_{k=1}^K p(x_k\mid \mathrm{Pa}_k)$.
- **Eq. (8.44)/(8.46) marginal likelihood & Bayes factor:** evidence $p(\mathcal D\mid M_k)=\int p(\mathcal D\mid\boldsymbol\theta_k)p(\boldsymbol\theta_k\mid M_k)d\boldsymbol\theta_k$; Bayes factor $=\frac{p(\mathcal D\mid M_1)}{p(\mathcal D\mid M_2)}$.
- **Information criteria:** AIC $=\log p(\mathbf x\mid\boldsymbol\theta)-M$ (8.48); BIC $\approx\log p(\mathbf x\mid\boldsymbol\theta)-\tfrac12 M\log N$ (8.49).

## What this changes in the wiki
- First **Part II** source page: pivots the wiki from pure math toward ML methods. The earlier Ch. 2 pages ([[vector]], [[linear-mapping]], [[affine-space]]) are now *used*: a predictor is literally an affine map, and the least-squares objective is a [[system-of-linear-equations]] in disguise (normal equations, Ch. 9).
- Establishes the **ERM ↔ MLE ↔ Bayes** triangle and the **loss ↔ likelihood** / **regularizer ↔ prior** correspondences that Chapters 9–12 reuse.
- Introduces person pages: [[thomas-bayes]], [[ronald-fisher]], [[vladimir-vapnik]], [[william-of-occam]]. [[carl-friedrich-gauss]] (already present) reappears via least squares / the Gaussian likelihood.
- Many forward TODO links left for later chapters: [[linear-regression]], [[gaussian-distribution]], [[principal-component-analysis]], [[gaussian-mixture-model]], [[bayes-theorem]], [[expectation-maximization]], [[conjugate-prior]], [[support-vector-machine]].
