---
title: Machine Learning
type: concept
tags: [machine-learning, ai, supervised-learning, foundations]
difficulty: beginner
aliases: [ML]
sources: [llm-ch01-understanding-large-language-models, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Machine Learning

> **One-liner:** Machine learning means letting a computer learn the rules from examples, instead of a programmer writing the rules by hand.

## 🟢 Intuition (Dummy)
Say you want a spam filter. The old way: sit down and write rules — "if the
email says *FREE PRIZE*, it's spam". You'll miss tons of cases and spend forever
updating rules.

The machine-learning way: collect a big pile of emails that people already
marked "spam" or "not spam". Show them to an algorithm. It adjusts itself to make
fewer mistakes on that pile, and in doing so it **figures out the patterns on
its own**. Then it can judge brand-new emails it has never seen.

So ML is "learning from data": the more examples (and feedback) it gets, the
better it becomes. It powers recommendation systems, spam filters, voice
assistants, self-driving cars — and LLMs.

## 🟡 Mechanics (Practitioner)
**Book definitions.**
- ML develops algorithms that "learn from and make predictions or decisions
  based on data without being explicitly programmed" (§1.1, p. 3).
- Fig. 1.1 label: "Algorithms that learn rules automatically from data."
- App. A: a subfield of AI focused on developing and improving learning
  algorithms; they identify patterns, learn from historical data, and improve
  with more data and feedback (A.1.2, p. 253).

**The supervised workflow (Fig. A.3, pp. 253–254):**
1. **Training:** a learning algorithm fits a model on a training dataset of
   examples + labels.
2. **Evaluation:** check the model meets performance criteria before real use.
3. **Inference:** the trained model predicts labels for new observations.

**"Minimizing the error on a training dataset", in math** (the book says it in
words, p. 3; this is the standard formalization —
[[empirical-risk-minimization]]):

$$\theta^{*} = \arg\min_{\theta}\ \frac{1}{N}\sum_{n=1}^{N} \ell\big(f_\theta(\mathbf x_n),\, y_n\big)$$

Symbol by symbol:
- $\mathbf x_n$ = the $n$-th training example (e.g. an email turned into numbers).
- $y_n$ = its label (spam / not spam).
- $f_\theta$ = the model, with adjustable parameters $\theta$.
- $\ell(\cdot,\cdot)$ = the [[loss-function]]: how wrong the prediction $f_\theta(\mathbf x_n)$ is compared with $y_n$.
- $\frac{1}{N}\sum_{n=1}^{N}$ = average over all $N$ training examples.
- $\arg\min_\theta$ = "the $\theta$ that makes this as small as possible"; $\theta^{*}$ = that best $\theta$.

Then inference is just $\hat y = f_{\theta^{*}}(\mathbf x_{\text{new}})$ — the hat
means "predicted".

**Classic ML often needs hand-made features** ([[feature-extraction]]). For spam
the book lists: counts of trigger words ("prize", "win", "free"), number of
exclamation marks, all-uppercase words, suspicious links (p. 4).
[[deep-learning]] removes that manual step.

**How LLMs fit.** Training an LLM to classify text follows Fig. A.3 directly
([[classification-fine-tuning]]). Training it to generate text also fits: in
[[pretraining]] the labels are the next words of the text itself
([[next-token-prediction]]); at inference the LLM writes new text from a prompt
instead of predicting a label (A.1.2, p. 254).

## 🔴 Theory (Expert)
- **ERM and generalization.** Minimizing the training average is a proxy for the
  true goal, the expected loss $\mathbb E_{(\mathbf x,y)}[\ell(f_\theta(\mathbf
  x),y)]$ on unseen data. The gap between them is what
  [[overfitting-and-underfitting]], [[regularization]] and held-out evaluation
  ([[train-validation-test-split]], [[cross-validation]]) are about — hence the
  book's insistence on an evaluation step between training and inference.
- **Probabilistic reading.** With $\ell$ = negative log-likelihood, ERM becomes
  [[maximum-likelihood-estimation]]; this is exactly the LLM pretraining
  objective (cross-entropy on next tokens).
- **Optimization.** For differentiable models, $\theta^*$ is approached
  iteratively with [[gradient-descent]] / [[stochastic-gradient-descent]], using
  gradients from [[backpropagation]].
- **Beyond rules.** ML lets AI move past strict rule-based systems and adapt to
  new inputs or changing environments (A.1.2, p. 253). AI also includes non-ML
  approaches: rule-based systems, genetic algorithms, expert systems, fuzzy
  logic, symbolic reasoning (§1.1, p. 4).
- **Learning paradigms.** The book's examples are [[supervised-learning]]
  (labeled spam data) and [[self-supervised-learning]] (labels derived from the
  data); fine-tuning an LLM is supervised learning on top of a self-supervised
  [[foundation-model]].

## Connections
- Builds on: [[artificial-intelligence]], [[loss-function]], [[probability-distribution]], [[vector]]
- Leads to: [[deep-learning]], [[supervised-learning]], [[empirical-risk-minimization]], [[large-language-model]]
- Used by: [[text-classification]], [[classification-fine-tuning]], [[linear-regression]], [[support-vector-machine]]
- Related: [[feature-extraction]], [[overfitting-and-underfitting]], [[gradient-descent]], [[maximum-likelihood-estimation]], [[cross-validation]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1, book pp. 3–4 (PDF pp. 25–26) — definition, spam example, handcrafted features, non-ML AI approaches, Fig. 1.1.
- Same file §A.1.2, book pp. 252–254 (PDF pp. 274–276) — ML definition, applications, supervised workflow (Fig. A.3), how LLMs fit.
