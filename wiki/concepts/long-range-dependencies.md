---
title: Long-Range Dependencies (RNN Bottleneck)
type: concept
tags: [deep-learning, sequence-models, rnn, attention, nlp]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Long-Range Dependencies (RNN Bottleneck)

> **One-liner:** A long-range dependency is when a word needs information from a word far away in the sentence — and older recurrent models were bad at it because everything had to squeeze through one memory vector.

## 🟢 Intuition (Dummy)
Take the German sentence "Kannst du mir helfen diesen Satz zu uebersetzen". Translate
it word by word and you get "Can you me help this sentence to translate" — wrong.
The correct English is "Can you help me to translate this sentence". To write
"translate" in the right place, you need the German word at the very *end*; to place
"me", you need to know a verb is coming. Words depend on other words, sometimes far
apart. That is a **long-range dependency**.

Old encoder–decoder RNNs read the whole sentence and squeezed it into one
[[hidden-state]] — one fixed-size sticky note. The decoder could only look at that
note, never back at the original words. For short sentences that is fine. For long
ones, early details fade or get overwritten. It's like trying to retell a long story
when you're only allowed to keep a one-line summary.

The fix was to let the decoder **look back at every input word** whenever it needs
to — that's [[attention-mechanism|attention]].

## 🟡 Mechanics (Practitioner)
**What the book says.** Translation can't be done word by word because languages
have different grammatical structures; it needs contextual understanding and
grammatical alignment (book p. 52, Fig. 3.3). Encoder–decoder RNNs process the input
sequentially and try to capture its whole meaning in the final hidden state (book
p. 53). The big limitation: during decoding, the RNN **cannot directly access
earlier hidden states** of the encoder; it relies only on the *current* hidden state,
"which encapsulates all relevant information". This leads to **loss of context,
especially in complex sentences where dependencies might span long distances**
(book p. 53). §3.2 restates it: RNNs work for short sentences but fail on longer
texts because they have no direct access to earlier words; the encoder must compress
the *whole* input into one hidden state (book p. 54). That shortcoming is what
"motivated the design of attention mechanisms" (book p. 54).

**Two ways to see the bottleneck** (background math, not printed in the book):

1. *Capacity.* The decoder sees only $\mathbf h_T\in\mathbb R^{d}$.
   - $\mathbf h_T$ = the encoder's last hidden state.
   - $\mathbb R^{d}$ = "a list of $d$ real numbers" — the size is fixed no matter how long the input $T$ is.

   A 5-word sentence and a 500-word paragraph both get $d$ numbers.

2. *Path length.* Information from word 1 reaches the decoder only after passing
   through every update $\mathbf h_1\to\mathbf h_2\to\cdots\to\mathbf h_T$. The
   gradient for training follows the same path backward:
   $$\frac{\partial \mathbf h_T}{\partial \mathbf h_1}=\prod_{t=2}^{T}\frac{\partial \mathbf h_t}{\partial \mathbf h_{t-1}}$$
   - $\frac{\partial \mathbf h_T}{\partial \mathbf h_1}$ = "how much does the final memory change if the first memory changes?" (a Jacobian [[matrix]]).
   - $\prod_{t=2}^{T}$ = multiply together the factors for $t=2,3,\dots,T$ ([[chain-rule]]).
   - Each factor $\frac{\partial \mathbf h_t}{\partial \mathbf h_{t-1}}$ = one step's sensitivity.

   Multiply many factors smaller than 1 and you get ~0 ([[vanishing-and-exploding-gradients|vanishing gradients]]); larger than 1 and it explodes.

**How attention fixes it.** With attention, the decoder can access *all* input tokens
selectively at each step; some inputs matter more than others, and the importance is
set by attention weights (book p. 54, Fig. 3.5 — in the figure, line width is
proportional to importance). The book prints no formula here; the weighted sum below
is formalized in §3.3 (book p. 60):
$$\mathbf z=\sum_{j}\alpha_j\,\mathbf x_j$$
- $\mathbf z$ = the [[context-vector]] for the token being produced.
- $\mathbf x_j$ = the representation of input token $j$.
- $\alpha_j$ = how important input $j$ is right now ([[attention-weights]]; $\alpha_j\ge 0$, $\sum_j\alpha_j=1$).

Now word 1 is *one hop* away from every output, whatever $T$ is.

## 🔴 Theory (Expert)
- **Maximum path length.** In a recurrent net the signal path between positions $i$
  and $j$ has length $O(|i-j|)$; in self-attention it is $O(1)$ — every pair is
  connected directly by a score $\mathbf q_i^\top\mathbf k_j$. This is the
  argument given by Vaswani et al. (2017) for preferring attention ([[ashish-vaswani]]).
  The cost is $O(T^2)$ pairwise scores per layer, which motivates work like
  [[flashattention]].
- **Spectral view of vanishing gradients.** For a linearised RNN
  $\mathbf h_t=W\mathbf h_{t-1}$, $\partial\mathbf h_T/\partial\mathbf h_1=W^{T-1}$,
  whose norm behaves like $\rho(W)^{T-1}$ (spectral radius). $\rho<1$ ⇒ exponential
  decay of long-range signal; $\rho>1$ ⇒ explosion. Nonlinearities with
  $|f'|\le 1$ (e.g. $\tanh$) only worsen decay.
- **Information-theoretic view.** A fixed-dimensional finite-precision state has
  bounded capacity; mutual information between distant inputs and outputs is capped
  by what fits in $\mathbf h_T$. Attention sidesteps this by keeping all $T$
  states and selecting among them per step.
- **Historical arc.** RNN encoder–decoder → [[bahdanau-attention]] (2014: keep all
  encoder states, attend over them) → transformer (2017: drop recurrence; only
  attention) — exactly the sequence the book traces in §§3.1–3.2 (book pp. 52–55).
- Causal LLMs still have a hard limit: the [[context-length]]. Dependencies longer
  than the context window cannot be modelled at all.

## Connections
- Builds on: [[hidden-state]], [[recurrent-neural-network]], [[encoder-decoder-architecture]], [[chain-rule]]
- Leads to: [[bahdanau-attention]], [[attention-mechanism]], [[self-attention]], [[context-length]]
- Used by: [[machine-translation]], [[natural-language-processing]]
- Related math: [[vanishing-and-exploding-gradients]], [[dimensionality-reduction]], [[convex-combination]]
- People: [[dzmitry-bahdanau]], [[ashish-vaswani]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1, book pp. 52–54 (Figs. 3.3–3.4); §3.2, book pp. 54–55 (Fig. 3.5). The Jacobian-product, spectral-radius, and path-length arguments are standard background, not printed in the book.
