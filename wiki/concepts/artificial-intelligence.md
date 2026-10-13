---
title: Artificial Intelligence
type: concept
tags: [foundations, ai, machine-learning, deep-learning, llm]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Artificial Intelligence

> **One-liner:** Artificial intelligence (AI) is the big umbrella field of building computers that can do things we normally think need a human brain, like understanding language, spotting patterns, and making decisions.

## 🟢 Intuition (Dummy)
Think of a set of Russian nesting dolls. The biggest doll is **AI**: "any computer system that does
something that looks smart." Open it and you find a smaller doll, [[machine-learning]]: "smart
behaviour that the computer *learns from examples* instead of being told rule by rule." Inside that
is [[deep-learning]]: "machine learning done with big, many-layered [[neural-network|neural networks]]."
And inside that is the doll this whole book is about: [[large-language-model|large language models]]
(LLMs), deep networks that read and write human-like text.

So when the news calls ChatGPT "an AI", that is true, but it is a bit like calling a poodle "an
animal". It is correct, just very broad. And AI is not *only* machine learning: older "smart"
systems were hand-written rule books, and those count as AI too.

One honest caveat from the book: AI is still far from **general intelligence**, the flexible,
do-anything kind of smart that people have (book p. 252).

## 🟡 Mechanics (Practitioner)

**Definitions used in the book.**
- **AI**: "the broader field of creating machines that can perform tasks requiring human-like
  intelligence, including understanding language, recognizing patterns, and making decisions"
  (book p. 3; similar wording on p. 252).
- **[[machine-learning|Machine learning (ML)]]**: the subfield of AI that develops algorithms that
  "learn from and make predictions or decisions based on data without being explicitly programmed"
  (book p. 3, p. 253).
- **[[deep-learning|Deep learning (DL)]]**: the subset of ML that uses neural networks with **three or
  more layers** (deep neural networks) to model complex patterns (book p. 3). Appendix A adds: these
  networks were originally inspired by interconnected neurons in the brain; "deep" refers to the
  multiple **hidden layers** that model complex, nonlinear relationships; and DL is especially good
  with **unstructured data** (images, audio, text), which is why it suits LLMs (book p. 253).
- **[[large-language-model|LLMs]]**: a specific application of deep learning: "deep neural network for
  parsing and generating human-like text" (Figure 1.1 label, book p. 3).
- **[[generative-ai|GenAI]]**: the use of deep neural networks to create new content (text, images,
  other media). In Figure 1.1 the GenAI box sits inside deep learning and partly overlaps the LLM box
  (book p. 3).

**The nesting, written as sets.** You can read Figures 1.1 and A.2 as a chain of subsets:
$$\text{LLMs}\;\subset\;\text{DL}\;\subset\;\text{ML}\;\subset\;\text{AI}.$$
Symbol by symbol: $\subset$ = "is a subset of" (every member of the left set is also in the right
set). So "LLMs $\subset$ DL" reads "every LLM is a deep-learning model", and the whole chain says
every LLM is, in the end, an AI system, but not every AI system is an LLM.

**What else lives inside AI (but outside ML).** The book lists **rule-based systems, genetic
algorithms, expert systems, fuzzy logic, and symbolic reasoning** (book p. 4). Today the field is
dominated by ML and DL, but these other approaches still belong to AI.

**Rules vs. learning: the spam-filter example.** A rule-based AI spam filter is a list of hand-written
`if` statements ("if the email contains *prize* and three `!`, mark as spam"). An ML spam filter is
instead shown emails labelled *spam* / *not spam* and learns the patterns itself by "minimizing the
error in its predictions on a training dataset" (book p. 3), see [[empirical-risk-minimization]].
ML lets AI move beyond strict rule-based systems and adapt to new inputs (book p. 253).

**Where ML-powered AI shows up** (book p. 253): LLMs, recommendation systems (online shops,
streaming), email spam filtering, voice recognition in virtual assistants, and self-driving cars.

## 🔴 Theory (Expert)
- **"Understanding" is a functional claim, not a mental one.** The book is careful: when we say LLMs
  "understand", we mean they process and generate text that *appears* coherent and contextually
  relevant, not that they have human-like consciousness or comprehension (book p. 2). This is the
  behavioural (Turing-style) view of intelligence: judge by outputs, stay agnostic about inner states.
- **Narrow vs. general AI.** Every system in the book is *narrow* in the technical sense: it optimizes
  one objective (next-token prediction, spam classification). LLMs look "broad" because one objective,
  [[next-token-prediction]], turns out to need many skills at once, but the book still says AI is far
  from general intelligence (book p. 252).
- **Symbolic vs. statistical AI.** Rule-based and expert systems encode knowledge explicitly as
  symbols and logic; ML encodes it implicitly in [[model-parameters|numerical parameters]] fitted to
  data. The modern trend (which LLMs embody) is to replace hand-written knowledge with learned
  parameters plus very large datasets. The cost is interpretability: a rule can be read, a 124-million
  number weight vector cannot.
- **Why the nesting matters for the math.** Each step inward adds structure: ML adds "fit a function
  to data by minimizing a [[loss-function]]"; DL adds "the function is a deep composition of
  [[linear-mapping|linear maps]] and nonlinearities"; LLMs add "the data is text and the target is
  the next token". The math in the rest of this wiki enters at the ML level.

## Connections
- Builds on: — (this is the root of the hierarchy, so it has no prerequisite page)
- Leads to: [[machine-learning]], [[deep-learning]], [[large-language-model]], [[generative-ai]]
- Related: [[neural-network]], [[feature-extraction]], [[natural-language-processing]],
  [[empirical-risk-minimization]], [[supervised-learning]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1 (Figure 1.1), book pp. 1–4 (PDF pp. 23–26).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1.2 "Defining deep learning" (Figure A.2), book pp. 252–254 (PDF pp. 274–276).
