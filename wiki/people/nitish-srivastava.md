---
title: Nitish Srivastava
type: person
tags: [deep-learning, regularization, dropout, neural-networks]
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Nitish Srivastava (active 2010s–present)

> **Why they matter to ML:** he is the first author of the 2014 **Dropout** paper. Dropout is the "randomly switch off some units while training" trick. GPT-style LLMs use it in the attention weights, after the embeddings and inside every transformer block.

## Life (brief)
Nitish Srivastava did his PhD in [[geoffrey-hinton]]'s group at the University of Toronto. The dropout work came out of that group: Hinton and colleagues first proposed the idea in 2012, and Srivastava led the full journal paper in 2014. (These biographical details are general knowledge. The book names him only once, as "Srivastava et al." in the appendix B references for chapter 3.)

## Key contributions
- **Dropout** (Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", *JMLR* 15, 2014). The book describes it as "a regularization technique used in neural networks to prevent overfitting by randomly dropping units (along with their connections) from the neural network during training" (book p. 292). Today it is just called **dropout**. → [[dropout]], [[regularization]], [[overfitting]]
  - **The intuition (book p. 78):** during training you pick some hidden units at random and ignore them. Then the network can't lean too hard on any one specific set of units, so it has to spread what it learns around. Dropout is **only for training**. You switch it off afterward, when you use the model (see [[train-vs-eval-mode]]).
  - **Where the book uses it:** in chapter 3 the book applies dropout to the **[[attention-weights]]**, right after the softmax. The book says this is the more common of the two choices (the other choice is after multiplying by the values) (book p. 78). The book says it uses a lower rate later, "such as 0.1 or 0.2", when it trains the GPT model (book p. 78). In chapter 4 the GPT-2 config sets `"drop_rate": 0.1`, which means 10% of units are dropped at random. → [[causal-attention]], [[gpt-model-configuration]]
  - **The formula ("inverted dropout", what PyTorch's `nn.Dropout` does):**
    $$\tilde{a}_{ij} = \frac{m_{ij}}{1-p}\, a_{ij}, \qquad m_{ij}\sim\text{Bernoulli}(1-p), \qquad \mathbb{E}[\tilde a_{ij}] = a_{ij}$$
    Symbol by symbol: $a_{ij}$ = one entry of the matrix before dropout (for example, how much token $i$ attends to token $j$); $p$ = the dropout rate (0.5 in the demo, 0.1 in GPT-2); $m_{ij}$ = a random coin flip that is 1 ("keep") with probability $1-p$ and 0 ("drop") with probability $p$; $\sim\text{Bernoulli}(1-p)$ = "is drawn from that coin-flip distribution" (see [[random-variable]]); $\frac{1}{1-p}$ = the boost we give to the survivors; $\tilde a_{ij}$ = the entry after dropout; $\mathbb{E}[\cdot]$ = the average over many coin flips (see [[expected-value-and-mean]]). The last equation says that, on average, the value stays the same. The book puts it this way: the "average influence" of attention stays consistent between training and inference (book pp. 79–80).
  - **The book's worked numbers (book pp. 78–80):** with $p = 0.5$, the boost is $1/(1-0.5) = 2$. A $6\times 6$ matrix of ones comes out with about half the entries at 0 and the rest at 2. If you apply it to the causal attention weights, row 3 `[0.38, 0.30, 0.31, 0, 0, 0]` becomes `[0.7599, 0.6194, 0.6206, 0, 0, 0]`, which is each kept weight times 2. Row 2 got dropped completely, so it is all zeros. That means the rows no longer sum to 1.
    ```python
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    print(dropout(example))   # ~half 0., the rest 2.
    ```
    The book adds a caveat: the exact pattern can look different on a different operating system (PyTorch issue #121595, book p. 80).
- **Why it works, roughly (from the paper, not the book):** every training step trains a different random "thinned" sub-network. Using the full network with the rescaling at test time is like cheaply averaging all of those sub-networks. That is a form of ensembling, and that is why it reduces overfitting.

## Connections
- Concepts: [[dropout]], [[regularization]], [[overfitting]], [[overfitting-and-underfitting]], [[attention-weights]], [[causal-attention]], [[train-vs-eval-mode]], [[random-variable]], [[expected-value-and-mean]], [[neural-network]], [[deep-learning]]
- People: [[geoffrey-hinton]] (PhD advisor and co-author of the dropout paper)
- Appears in sources: [[llm-ch03-coding-attention-mechanisms]] (dropout in attention), [[llm-ch04-implementing-a-gpt-model]] (`drop_rate` in GPT-2)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 3 references, book p. 292 (Srivastava et al. 2014, dropout definition, jmlr.org/papers/v15/srivastava14a.html).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5.2, book pp. 78–80 (dropout definition, training-only use, 50% demo, 1/(1−p) = 2 scaling, dropout on attention weights, OS caveat).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book p. 95 (`drop_rate: 0.1` in the GPT-2 124M config).
- The biography, the full co-author list, the Bernoulli/expectation formula, and the ensemble explanation are general knowledge from the original paper. They are not printed in the book.
