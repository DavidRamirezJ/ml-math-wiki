---
title: Friedrich Wilhelm Bessel
type: person
tags: [mathematician, astronomer, statistics, variance]
sources: [llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Friedrich Wilhelm Bessel (1784–1846)

> **Why they matter to ML:** his name is on **Bessel's correction**, the $n-1$ (instead of $n$) in the sample-variance formula. GPT-style [[layer-normalization]] deliberately *skips* it.

## Life (brief)
Bessel was a German astronomer and mathematician. He spent most of his career running the
Königsberg Observatory. He was obsessed with measurement error. If you point a telescope at the
same star many times, you get slightly different numbers each time, so he needed solid ways to
estimate how spread out those errors really were. In 1838 he made the first reliable measurement
of a star's parallax (61 Cygni). That gave the first real distance to a star other than the Sun.

## Key contributions
- **Bessel's correction** is what we call it today. You divide by $n-1$ instead of $n$ when you
  estimate variance from a sample, so the estimate isn't biased toward being too small:
  $$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2 \quad\text{vs.}\quad \sigma^2_{\text{biased}} = \frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2$$
  Here $x_i$ is the $i$-th value, $\bar{x}$ is the sample mean (the "$n-1$" is needed exactly because $\bar{x}$ is itself estimated from the same data), $n$ is how many values you have, and
  $\sum_{i=1}^{n}$ means "add up over every value." → [[bessels-correction-biased-variance]],
  [[covariance-and-variance]]
  - *In LLMs:* Raschka's `LayerNorm` class calls `x.var(dim=-1, keepdim=True, unbiased=False)`.
    That means it divides by $n$, with **no** Bessel's correction, which gives a "biased estimate
    of the variance." The book gives three reasons: (1) it stays compatible with GPT-2's
    normalization layers, (2) it matches TensorFlow's default, which the original GPT-2 used, and
    (3) it lets the pretrained OpenAI weights load correctly. Because $n$ here is the
    [[embedding-dimension]] (768 for GPT-2 small), the gap between the two is tiny:
    $\tfrac{n-1}{n} = \tfrac{767}{768} \approx 0.9987$ (book p. 103). The book itself only says the difference is "practically negligible" because $n$ is large; the 767/768 figure is our arithmetic.
  - > ⚠️ Book cross-reference note: the sidebar says the pretrained weights get loaded "in
    > chapter 6". They actually get loaded in §5.5 (section starts book p. 160; `load_weights_into_gpt` is called on p. 166). Chapter 6 then reuses them.
  - *History note:* Gauss already used the $n-1$ form in his work on least squares in the 1820s,
    so Bessel's name on it is partly a naming convention. → [[carl-friedrich-gauss]]
- **Bessel functions** solve Bessel's differential equation. They show up wherever you get waves
  or heat in circular or cylindrical shapes. *(No wiki page yet. They aren't used in this book.)*
- **Precision astrometry** and the first stellar parallax (1838). This is a classic early example
  of carefully modelling measurement error, the same thinking that underlies
  [[standardization-z-score]] and [[probability-distribution]]-based estimation.

## Connections
- Concepts: [[bessels-correction-biased-variance]], [[covariance-and-variance]], [[layer-normalization]], [[standardization-z-score]], [[expected-value-and-mean]], [[embedding-dimension]]
- People: [[carl-friedrich-gauss]]
- Appears in sources: [[llm-ch04-implementing-a-gpt-model]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2 "Normalizing activations with layer normalization", sidebar "Biased variance", book p. 103 (PDF p. 125).
- The biographical details (dates, Königsberg, the 61 Cygni parallax, Bessel functions, Gauss's earlier $n-1$ usage) are general knowledge. The book doesn't state them. Cite them properly once a biographical source has been ingested.
