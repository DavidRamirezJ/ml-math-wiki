# Storyboard — Inner product (ML Math Atoms · Video 5 of 16)

Source: wiki `inner-product` (MML §3.2, Defs 3.2–3.3), `angle-between-vectors` (§3.4), `orthogonality` (Def 3.7);
Raschka §3.3.1 pp. 58–59 (attention score 0.9544). Axiom: **B — Inner product**.

## Hook
How does a model know cat is closer to dog than to car? One number: the inner product — the engine of attention.

## Prerequisites
Video 1 (vector), video 2 (matrix, transpose).

## Chapters
| # | Scene | Learning goal | Main visual | Keys |
|---|---|---|---|---|
| 0 | C0_Intro | why | toy word arrows | intro1-2 |
| 1 | C1_Dot | recipe, Σ x_i y_i, xᵀy; (1,2,3)·(4,−5,6)=12; (1,1)·(1,2)=3 | matching components | dot1-4 |
| 2 | C2_Geometry | shadow (projection); ‖x‖‖y‖cos θ; θ≈18.4° | shadow on y; angle arc | geo1-5 |
| 3 | C3_Sign | sign; orthogonal; (1,1)·(−1,1)=0 | rotating vector with live dot | sign1-4 |
| 4 | C4_Cosine | cosine similarity; toy cat/dog/car | [−1,1] scale; bars | cos1-4 |
| 5 | C5_AxiomB | symmetric, linear, positive; geometry | rule list | ax1-4 |
| 6 | C6_ML | attention score (book), similarity search, neuron | Your·journey = 0.9544 | ml1-3 |
| 7 | C7_Practice | (2,−1,3)·(1,4,2)=4; (3,6)⟂(−2,1) | | prac1-4 |
| 8 | C7_Code | NumPy | code + terminal | code1-2 |
| 9 | C8_Summary | summary + next (norm) | bullets | sum1-2 |
| 10 | C9_Credits | | | |

## Terms (owner = this video)
- dot product — Multiply matching components and add: x·y = Σ x_i y_i.
- inner product — A rule ⟨x, y⟩ that turns two vectors into one number; symmetric, linear in each input, positive.
- orthogonal — Two vectors are orthogonal when their inner product is zero (at a right angle).
- cosine similarity — x·y divided by the two lengths: the cosine of the angle, from −1 to 1.
Reminder: transpose (video 2). "Length" used informally; norm = video 6.

## Colour ledger
x, u, first vector → BLUE · y, v, second vector → GREEN · results, dot values → YELLOW · weights, i, k → PURPLE ·
shadow guide lines → ORANGE · negative / not orthogonal → RED · labels → MUTED.
