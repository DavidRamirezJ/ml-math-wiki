# Storyboard — Matrix (ML Math Atoms · Video 2 of 16)

Source: wiki `matrix`, `matrix-multiplication`, `matrix-transpose` (MML §2.2, Defs 2.1–2.5, pp. 22–26;
Raschka §A.2.3 pp. 260–261, §4.1 p. 94). Axiom: **A — Linearity**.

## Hook (first 30 s)
A model never sees one vector; it sees a whole table of them. Every layer stores what it learned
in a grid of numbers: a matrix. Learn its two faces — table and machine.

## Prerequisites
Video 1 (vector, adding, scaling, linear combination).

## Chapters
| # | Scene | Learning goal | Main visual | Keys |
|---|---|---|---|---|
| 0 | C0_Intro | why matrices; plan | title, table + machine | intro1-2 |
| 1 | C1_Table | matrix as data table, m×n, a_ij, rows/columns | 3 houses × 3 features | table1-4 |
| 2 | C2_Machine | A x = linear combination of columns; row view | [[2,1],[1,3]]·(1,2)=(4,7) | mach1-5 |
| 3 | C3_Grid | columns = where e1, e2 land; stretch & rotate | NumberPlane transforms | grid1-5 |
| 4 | C4_Multiply | product = one machine after another; entry rule; shapes; AB≠BA | [[1,2],[3,4]][[5,6],[7,8]]; RS vs SR | mult1-7 |
| 5 | C5_TransposeIdentity | transpose, identity; axiom A | 2×3 → 3×2; I | tr1-3, id1-2, axiom |
| 6 | C6_ML | batch × weights; layer weight matrix; size | X w = (255,150,335) toy; 2,048×2,048 | ml1-4 |
| 7 | C7_Practice | 2 questions | [[1,2],[0,1]](3,1); shapes 2×3, 3×4 | prac1-4 |
| 8 | C7_Code | NumPy demo, real output | code + terminal | code1-2 |
| 9 | C8_Summary | summary + next (basis & rank) | bullets | sum1-2 |
| 10 | C9_Credits | credits (common.py) | — | — |

## Terms (owner = this video)
- matrix — A rectangular grid of numbers, arranged in rows and columns.
- matrix multiplication — Doing one matrix after another; entry (i, j) = row i of A times column j of B, multiplied entry by entry and added.
- transpose — Flip a matrix over its diagonal: rows become columns.
- identity matrix — Ones on the diagonal, zeros elsewhere; it changes nothing.
Reminders: vector, linear combination (video 1). Mentioned only: inner product (video 5), linear map (video 4).

## Symbol & colour ledger
first column / input → BLUE · second column → GREEN · result → YELLOW · weights / entries of x → PURPLE ·
guide lines / one column highlight → ORANGE · not allowed → RED · labels → MUTED.
