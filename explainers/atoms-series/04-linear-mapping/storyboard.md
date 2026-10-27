# Storyboard — Linear mapping (ML Math Atoms · Video 4 of 16)

Source: wiki `linear-mapping` (MML §2.7, Defs 2.15, 2.19), `affine-space` (MML §2.8, Def 2.26); Raschka §4.3 p. 108
(768 → 3072 → 768 feed-forward), App. A (Linear = weight matrix + bias). Axiom: **A — Linearity**.

## Hook
Every layer starts with a map that respects add and scale. Knowing linear maps = knowing what a layer can and cannot do.

## Prerequisites
Videos 1 (vector), 2 (matrix), 3 (basis).

## Chapters
| # | Scene | Learning goal | Main visual | Keys |
|---|---|---|---|---|
| 0 | C0_Intro | why | x → layer → y | intro1-2 |
| 1 | C1_Definition | f(au+bv)=af(u)+bf(v); f(0)=0 | formula + legend; f(x)=3x check | def1-5 |
| 2 | C2_Pictures | grid stays straight/parallel/even, origin fixed | generic map; stretch, rotation, shear, projection | pic1-3 |
| 3 | C3_NonExamples | x+1 (affine), x² fail | number line shift; checks; graphs | non1-4 |
| 4 | C4_Matrix | columns = f(e1), f(e2); worked (3,2) → (4,5); rotation | transformed grid, column path | mat1-5 |
| 5 | C5_Compose | composition = product; kernel, image, dim balance | shear∘projection; projection drops | comp1-2, ker1-3 |
| 6 | C6_ML | Wx + b is affine; layers collapse; need activation | chain → one matrix; 768→3072→768 | ml1-3, axiom |
| 7 | C7_Practice | (2x, x+y) matrix; 5x−3 not linear | | prac1-4 |
| 8 | C7_Code | NumPy checks | code + terminal | code1-2 |
| 9 | C8_Summary | summary + next (inner product) | bullets | sum1-2 |
| 10 | C9_Credits | | | |

## Terms (owner = this video)
- linear mapping — A function f with f(a·u + b·v) = a·f(u) + b·f(v): it respects adding and scaling.
- kernel (null space) — All inputs that the map sends to the zero vector.
- image (range) — All outputs the map can produce.
Reminders: matrix (02), basis (03). Mentioned only: activation function (video 16). "affine" explained in words.

## Colour ledger
u, e1, f(e1) → BLUE · v, e2, f(e2) → GREEN · outputs / results → YELLOW · scalars a, b, c, weights → PURPLE ·
guide paths, kernel → ORANGE · not linear → RED · labels → MUTED.
