---
title: Machine Translation
type: concept
tags: [nlp, llm, sequence-to-sequence, applications]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Machine Translation

> **One-liner:** Machine translation is getting a computer to turn a sentence in one language into the same meaning in another language.

## 🟢 Intuition (Dummy)
You can't translate by swapping words one at a time with a dictionary. Languages put
words in different orders and glue meaning together differently. The book's example
(Fig. 3.3, book p. 52): the German sentence

> *Kannst du mir helfen diesen Satz zu uebersetzen*

translated word by word becomes "Can you me help this sentence to translate" — every
word is "right", but the sentence is broken. The correct answer is "Can you help me to
translate this sentence". To produce a good word, the translator often needs to look at
words that come **earlier or later** in the source. So translation needs *contextual
understanding* and *grammatical alignment*, not a lookup table.

Translation matters for this wiki for a historical reason: it is the task that **gave
birth to attention and the transformer**. The models that power today's
[[large-language-model|LLMs]] were first invented to translate.

## 🟡 Mechanics (Practitioner)

### The encoder–decoder setup
The usual design is an [[encoder-decoder-architecture|encoder–decoder]] network
(book p. 52):
- the **encoder** reads and processes the *whole* source text;
- the **decoder** then writes the target text, **one word at a time**.

Writing one word at a time is a conditional prediction (this formula is a standard
formalization, not printed in the book):

$$p(y_t \mid y_1, \dots, y_{t-1},\ \mathbf x)$$

Symbol by symbol:
- $\mathbf x$ = the whole source sentence (e.g. the German input);
- $y_t$ = the target word at step $t$;
- $y_1, \dots, y_{t-1}$ = the target words already written;
- $p(\cdot \mid \cdot)$ = "probability of the left side **given** the right side". The vertical bar "$\mid$" reads "given".

So at each step the decoder asks: "Given the source and what I've written so far, what
is the most likely next word?"

### Three generations of translation models in the book
1. **Encoder–decoder RNNs** (before transformers). The encoder updates a
   [[hidden-state]] word by word and tries to pack the meaning of the entire sentence
   into the **final** hidden state; the decoder translates from that single vector
   (Fig. 3.4: "Kannst du mir" → "Can you help", book p. 53). Weakness: the decoder
   "cannot directly access earlier hidden states", so long sentences lose context
   (book p. 53). See [[recurrent-neural-network]].
2. **RNN + Bahdanau attention (2014).** At each decoding step the decoder may look at
   *all* encoder positions and weight them by importance (Fig. 3.5, book p. 54). See
   [[attention-mechanism]].
3. **The original transformer (2017).** "Attention Is All You Need" dropped the RNN
   entirely. It was built for **English→German and English→French** translation
   (book p. 7). Its encoder sees the complete input and turns it into contextual
   embedding vectors; its decoder generates the translation one word at a time,
   using [[self-attention]] in both parts (book pp. 7–8).

### The book's transformer translation example (Fig. 1.4, book p. 8)
Input "This is an example". The decoder has already written "Das ist ein" and now only
needs the last word, **"Beispiel"**, giving "Das ist ein Beispiel". The eight numbered
steps: input text → preprocessing → encoder (sees full input) → embedding vectors to
the decoder → partial output → preprocessing → decoder generates one word → complete
translation.

### Translation without being trained for it
GPT models are *decoder-only* and trained only on [[next-token-prediction]], yet they
can translate. The book calls this unexpected and an example of
[[emergent-behavior]] (book pp. 13–14). In zero-shot form: input
`Translate English to German: breakfast =>` gives **"Frühstück"** (Fig. 1.6, book
p. 10). GPT, which keeps only the decoder of the original transformer, is designed for
text-generation tasks, and the book lists machine translation first among them (book
p. 9). Machine translation is also first on the book's list of LLM applications (book
p. 4); see [[llm-applications]]. Translation can also be taught directly: an
instruction fine-tuning dataset can pair "a query to translate a text" with "the
correctly translated text" (§1.3, book p. 7; see [[instruction-fine-tuning]]).

## 🔴 Theory (Expert)
- **Sequence-to-sequence as conditional language modeling.** By the chain rule of
  probability, $p(y_1,\dots,y_m\mid\mathbf x)=\prod_{t=1}^{m}p(y_t\mid y_{<t},\mathbf x)$.
  Training maximizes this likelihood ([[maximum-likelihood-estimation]]), i.e. it
  minimizes the token-level [[cross-entropy-loss]]. Decoding then picks words with
  [[greedy-decoding]], [[beam-search]], or sampling (background; not formalized in the
  book's translation sections).
- **Alignment.** Fig. 3.3's "words need access to earlier or later words" is the
  *alignment* problem: target position $t$ depends on a soft subset of source
  positions. Bahdanau et al.'s paper title — "Neural Machine Translation by Jointly
  Learning to Align and Translate" (book p. 291) — names it directly; attention
  weights *are* a learned soft alignment.
- **The bottleneck argument.** A fixed-size final hidden state $\mathbf h_T\in\mathbb R^d$
  must encode a sentence of any length. Capacity is fixed while input grows, so
  information is necessarily lost for long inputs; attention removes the single-vector
  bottleneck (book pp. 53–54).
- **Encoder vs. decoder masking.** In the transformer, the encoder can attend to the
  whole source (no mask), while the decoder must not see target words it has not
  written yet — the job of the [[causal-attention-mask]].
- **Subword vocabularies** came out of translation research too: Appendix B cites
  Sennrich et al. (2015), "Neural Machine Translation of Rare Words with Subword
  Units", for [[byte-pair-encoding]] (book p. 290).

## Connections
- Builds on: [[natural-language-processing]], [[encoder-decoder-architecture]], [[recurrent-neural-network]], [[probability-distribution]]
- Leads to: [[attention-mechanism]], [[bahdanau-attention]], [[self-attention]], [[transformer-architecture]]
- Related: [[llm-applications]], [[emergent-behavior]], [[zero-shot-learning]], [[autoregressive-text-generation]], [[long-range-dependencies]], [[byte-pair-encoding]]
- Used by: [[large-language-model]], [[gpt-architecture]]
- People: [[dzmitry-bahdanau]], [[yoshua-bengio]], [[ashish-vaswani]], [[rico-sennrich]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.2, book p. 4 (machine translation as an LLM application)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book p. 7 (translation pairs in instruction fine-tuning data)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 7–10 (GPT's generative tasks incl. machine translation, p. 9; original transformer for EN→DE/FR; Fig. 1.4 "Das ist ein Beispiel"; Fig. 1.6 zero-shot "Frühstück", p. 10)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book pp. 13–14 (translation as emergent behavior)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1, book pp. 52–54 (word-by-word failure, Fig. 3.3; encoder–decoder RNN, Fig. 3.4)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.2, book p. 54 (Bahdanau attention, Fig. 3.5)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book pp. 290–291 (Sennrich 2015; Bahdanau 2014)
