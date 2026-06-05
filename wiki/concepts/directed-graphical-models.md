---
title: Directed Graphical Models
type: concept
tags: [machine-learning, probabilistic-modeling, graphical-models]
difficulty: advanced
sources: [mml-ch08-when-models-meet-data]
created: 2026-06-04
updated: 2026-06-04
---

# Directed Graphical Models

> **One-liner:** A picture of arrows-between-circles that tells you how a big probability distribution breaks into a product of small ones — and which variables ignore which.

## 🟢 Intuition (Dummy)
A complicated joint probability over many variables is hard to read. A directed graphical model
draws each variable as a circle and each "depends-directly-on" relationship as an arrow. Now you can
*see* the structure: which variables influence which, and which ones don't care about each other.
It's like a wiring diagram for a probability model — and from the diagram alone you can read off
which quantities are independent without doing any algebra.

## 🟡 Mechanics (Practitioner)
A **directed graphical model** (DGM), a.k.a. **Bayesian network**, has nodes = random variables and
directed edges (arrows) = conditional dependencies (MML §8.5). The arrow $a\to b$ means the factor
$p(b\mid a)$ appears.

**Joint from graph (Eq. 8.31).** The joint factorizes as a product, one factor per node conditioned
on its **parents**:
$$ p(\mathbf x)=\prod_{k=1}^{K} p\big(x_k\mid \mathrm{Pa}_k\big), $$
where $\mathrm{Pa}_k$ = the nodes with arrows into $x_k$.

**Graph from joint.** Given a factorization, create a node per variable and draw an arrow into $x_k$
from each variable it is conditioned on.

**Worked example (Example 8.8, Fig. 8.9b).** The graph
$$ p(x_1,x_2,x_3,x_4,x_5)=p(x_1)\,p(x_5)\,p(x_2\mid x_5)\,p(x_3\mid x_1,x_2)\,p(x_4\mid x_2). $$
Five nodes, one conditional each; e.g. $x_4$ has only $x_2$ as a parent.

**Plate notation (Fig. 8.10).** A repeated structure — $N$ i.i.d. observations $x_n$ all sharing
parameter $\mu$ — is drawn once inside a box (a **plate**) labelled $n=1,\dots,N$ instead of $N$
copies. **Shaded** nodes are observed; unshaded are latent/parameters. A **hyperprior** (e.g.
$\mathrm{Beta}(\alpha,\beta)$ on $\mu$) is a prior on a prior's parameters.

**ML usage.** DGMs specify and communicate the models throughout Part II — naive Bayes, mixtures,
PCA, HMMs, topic models — and let inference/learning be expressed as graph operations (message
passing).

## 🔴 Theory (Expert)
- **Conditional independence by d-separation (§8.5.2, Pearl).** For disjoint node sets $A,B,C$, read
  off $A\perp\!\!\!\perp B\mid C$ by checking every undirected **trail** from $A$ to $B$. A trail is
  *blocked* if it passes a node where either (i) arrows meet **head-to-tail** or **tail-to-tail** and
  that node $\in C$, or (ii) arrows meet **head-to-head** (a collider) and neither it nor any
  descendant $\in C$. If *all* trails are blocked, $A\perp\!\!\!\perp B\mid C$ holds in the joint.
  The collider rule is the counter-intuitive "explaining away" effect.
- DGMs require a **directed acyclic graph**; the factorization (8.31) is exactly the chain rule
  applied in a topological order, which is why it always defines a valid joint.
- Two cousins (§8.5.3): **undirected graphical models** (Markov random fields, symmetric
  dependencies, no causal direction) and **factor graphs** (make the factors explicit, ideal for
  message passing). With extra assumptions the arrows can carry **causal** meaning (Pearl 2009).

## Connections
- Builds on: [[statistical-independence]], [[bayes-theorem]], [[probability-distribution]]
- Leads to: [[bayesian-inference]], [[bayesian-model-selection]]
- Used by: [[gaussian-mixture-model]], [[principal-component-analysis]], naive Bayes
- People: [[thomas-bayes]]

## Sources
- `raw/mml-book.pdf` §8.5 (Eqs. 8.29–8.34, Figs. 8.9–8.12, Examples 8.7–8.9), printed pp. 278–282 (PDF pp. 284–288).
