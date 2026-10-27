# Storyboard — Basis & rank (ML Math Atoms · Video 3 of 16)

Source: wiki `linear-independence`, `basis-and-rank` (MML §2.5 Defs 2.11–2.12 pp. 40–44; §2.6 Defs 2.13–2.14
pp. 44–48; Raschka App. E pp. 322–324 for LoRA). Axiom: **A — Linearity**.

## Hook
A table with 100 columns may hold far fewer real directions (meters vs centimeters). Rank counts them.

## Prerequisites
Video 1 (vector, span), video 2 (matrix).

## Chapters
| # | Scene | Learning goal | Main visual | Keys |
|---|---|---|---|---|
| 0 | C0_Intro | why count directions | toy m / cm / kg table | intro1-2 |
| 1 | C1_Redundant | hidden redundancy c = a + b | plane, tip-to-tail | red1-4 |
| 2 | C2_Independence | definition + zero test, 2 worked examples | formula + legend; closed loop a+b−c | ind1-7 |
| 3 | C3_Basis | basis, coordinates in two bases, dimension | e-grid vs b-grid, p=(1,3) → (2,1) | basis1-6, dim1-2 |
| 4 | C4_Rank | rank = # independent columns; line vs plane | pseudo-3D column spaces, rank 1/2/3 | rank1-5 |
| 5 | C5_ML | redundant features, LoRA mention, axiom A | toy table, rank 2 | ml1-3, axiom |
| 6 | C7_Practice | (1,2),(3,6) dependent; rank of [[1,2,3],[2,4,6]] = 1 | | prac1-4 |
| 7 | C7_Code | NumPy matrix_rank, solve | code + terminal | code1-2 |
| 8 | C8_Summary | summary + next (linear mapping) | bullets | sum1-2 |
| 9 | C9_Credits | credits | | |

## Terms (owner = this video)
- linear independence — No vector in the set is a combination of the others; only all-zero weights give 0.
- basis — A linearly independent set of vectors that spans the whole space.
- dimension — The number of vectors in any basis of a space.
- rank — The number of linearly independent columns of a matrix (= independent rows).
Reminders: span (video 1), matrix (video 2). Mentioned only: image (video 4).

## Colour ledger
a / e1 / b1 → BLUE · b / e2 / b2 → GREEN · c, result, point p → YELLOW · weights λ, c1, c2 → PURPLE ·
guide paths → ORANGE · dependent / redundant / warning → RED · labels → MUTED.
