---
title: Vladimir Vapnik
type: person
tags: [statistician, machine-learning, learning-theory]
sources: [mml-ch08-when-models-meet-data, mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Vladimir Naumovich Vapnik (b. 1936)

> **Why they matter to ML:** co-founder of **statistical learning theory** and the **support vector machine**, and the person who popularized **empirical risk minimization**.

## Life (brief)
Soviet-born (and later US-based) mathematician and computer scientist. Developed much of his work at
the Institute of Control Sciences in Moscow, then at AT&T Bell Labs in the 1990s, where the support
vector machine became practical. A central figure in giving machine learning rigorous theoretical
foundations.

## Key contributions
- **Empirical risk minimization & statistical learning theory** (Vapnik 1998, 1999) — the framework
  that asks "what does it mean to learn?" by minimizing average training loss while controlling the
  gap to the expected risk. → [[empirical-risk-minimization]]
- **VC (Vapnik–Chervonenkis) dimension** — a measure of the capacity of a hypothesis class that
  bounds the generalization gap, formalizing [[overfitting-and-underfitting]].
- **Support Vector Machine** — the margin-based classifier that popularized ERM in practice; its
  large-margin objective is equivalent to a regularizer (MML Ch. 12). With Chervonenkis he showed a
  large [[margin]] means low function-class complexity, hence good generalization. Co-developed the
  [[soft-margin-svm|soft-margin]] formulation and the [[kernel-trick|kernel]] approach. → [[support-vector-machine]]

## Connections
- Concepts: [[empirical-risk-minimization]], [[overfitting-and-underfitting]], [[regularization]], [[support-vector-machine]], [[margin]], [[soft-margin-svm]], [[hinge-loss]], [[kernel-trick]]
- Appears in sources: [[mml-ch08-when-models-meet-data]], [[mml-ch12-classification-with-support-vector-machines]]

## Sources
- `raw/mml-book.pdf` §8.2 & §8.2.5 (ERM popularized by the SVM; statistical learning theory, Vapnik 1998/1999), printed pp. 258, 265 (PDF pp. 264, 271).
- `raw/mml-book.pdf` §12.2.1 remark (margin ⇒ low complexity, Vapnik & Chervonenkis), §12.6, printed pp. 376, 392–394 (PDF pp. 382, 398–400).
- Biographical detail is general knowledge; to be expanded/cited when a biographical source is ingested.
