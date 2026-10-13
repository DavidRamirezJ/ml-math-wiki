---
title: Retrieval-Augmented Generation (RAG)
type: concept
tags: [llm, nlp, embeddings, applications]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data]
created: 2026-10-13
updated: 2026-10-13
---

# Retrieval-Augmented Generation (RAG)

> **One-liner:** RAG lets an LLM look things up first — it searches a knowledge base for relevant passages and then writes its answer using them.

## 🟢 Intuition (Dummy)
Imagine an open-book exam. A student who memorised the textbook (a plain LLM)
can answer many questions, but may misremember details or not know anything
newer than the book. A student who is allowed to **search the library first**
and then write the answer, with the right pages open, does much better on
facts.

RAG is that open-book student. Step 1: **retrieve** — find the few documents
most related to the question. Step 2: **generate** — give those documents to
the LLM together with the question, and let it write the answer.

How do we "find related documents" quickly? We turn each document (and the
question) into an [[word-embeddings|embedding]] — a list of numbers — and pick
the documents whose numbers are closest to the question's.

## 🟡 Mechanics (Practitioner)
**What the book says** (§2.1, book p. 19), and it is all it says:
- Besides word embeddings, there are **sentence, paragraph and document
  embeddings**, and sentence/paragraph embeddings are popular for RAG.
- RAG "combines *generation* (like producing text) with *retrieval* (like
  searching an external knowledge base) to pull relevant information when
  generating text".
- It is **beyond the scope of the book**.
- Ch. 1 also lists knowledge retrieval from large volumes of specialised text
  (medicine, law) as an LLM application (§1.2, book p. 4).

**The standard recipe** (general background, not from the book):
1. Split the knowledge base into chunks $c_1,\dots,c_N$ and embed each:
   $\mathbf{d}_i = f(c_i)$.
2. Embed the user question: $\mathbf{q} = f(\text{question})$.
3. Score every chunk by cosine similarity and keep the top $k$:

$$s_i = \frac{\mathbf{q}^{\top}\mathbf{d}_i}{\lVert\mathbf{q}\rVert\,\lVert\mathbf{d}_i\rVert},\qquad
\mathcal{R} = \operatorname{top-}k\{\, s_1,\dots,s_N \,\}$$

Symbol by symbol:
- $f$ = a sentence/paragraph embedding model (text → [[vector]] in $\mathbb{R}^d$).
- $\mathbf{q}$ = the question's embedding; $\mathbf{d}_i$ = chunk $i$'s embedding.
- $\mathbf{q}^{\top}\mathbf{d}_i$ = their dot product ([[inner-product]]).
- $\lVert\cdot\rVert$ = vector length ([[norm]]); dividing by both lengths gives the cosine of the angle between them ([[angle-between-vectors]]), a number in $[-1, 1]$.
- $s_i$ = the relevance score of chunk $i$; bigger = more related.
- $\mathcal{R}$ = the set of the $k$ best-scoring chunks.

4. Build a prompt "context: $\mathcal{R}$ … question: …" and let the LLM
   generate with [[autoregressive-text-generation]].

**Tiny example** (2-D toy vectors): $\mathbf{q}=[1,0]$,
$\mathbf{d}_1=[0.9,0.1]$, $\mathbf{d}_2=[0,1]$. Then
$s_1 = 0.9/\sqrt{0.82}\approx 0.99$ and $s_2 = 0$, so chunk 1 is retrieved.

## 🔴 Theory (Expert)
- **Why it helps.** A pretrained LLM stores knowledge only in its weights,
  fixed at training time. Appendix B points to Gekhman (2024), "Does Fine-Tuning LLMs on New
  Knowledge Encourage Hallucinations?": LLMs
  acquire factual knowledge mainly during [[pretraining]], and teaching *new*
  facts by [[fine-tuning]] is slow and increases hallucination (book
  pp. 297–298). RAG side-steps this by supplying facts at inference time
  instead of baking them into weights (our link; the book does not connect
  these two points).
- **Probabilistic view** (background): treat the retrieved passage $z$ as a
  latent variable and generate with
  $p(y\mid x)=\sum_{z\in\mathcal{R}} p(z\mid x)\,p(y\mid x,z)$ — a
  [[mixture-model]] over retrieved documents.
- **Limits.** Retrieved text has to fit inside the model's
  [[context-length]] (1,024 tokens for GPT-2; book p. 95), so chunk size and
  $k$ trade off against prompt space. Retrieval quality is bounded by how well
  the embedding geometry reflects relevance.
- **Search at scale** uses approximate nearest-neighbour indexes rather than
  scoring all $N$ chunks (background).

## Connections
- Builds on: [[word-embeddings]], [[inner-product]], [[norm]], [[angle-between-vectors]], [[large-language-model]]
- Leads to: [[llm-applications]]
- Related: [[autoregressive-text-generation]], [[context-length]], [[fine-tuning]], [[pretraining]], [[mixture-model]], [[embedding-dimension]]
- Used by: question-answering and chatbot systems ([[llm-applications]])
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book p. 19 (definition of RAG; sentence/paragraph embeddings; out of scope).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.2, book p. 4 (knowledge retrieval as an application).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 297–298 (Gekhman et al. 2024 on knowledge in pretraining vs fine-tuning).
