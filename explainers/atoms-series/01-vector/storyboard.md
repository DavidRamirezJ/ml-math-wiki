# Storyboard — Vector (ML Math Atoms · Video 1 of 16)

Source: `raw/mml-book.pdf` §2.1 (pp. 17–19), §2.4 (vector spaces); wiki `vector`, `vector-space`.
Axiom: **A — Linearity** (add and scale).

## Hook (first 30 s)
Every photo, word, song and customer that a machine-learning model sees becomes a list of
numbers first. That list is a vector. Learn two moves — add and scale — and you hold the
atom that every other atom in this series is made of.

## Prerequisites
Arithmetic only (adding and multiplying numbers). No algebra needed.

## Chapters
| # | Scene | Learning goal | Main visual | Keys |
|---|---|---|---|---|
| 0 | C0_Intro | Why vectors; plan | title, 3 icons → lists of numbers | intro1-2 |
| 1 | C1_ThreeFaces | Arrow, list, and "anything you can add and scale" | NumberPlane arrow (3,2) + components; house row; | faces1-4 |
| 2 | C2_Add | Add = tip-to-tail = add entries | (3,1)+(1,2)=(4,3), parallelogram | add1-4 |
| 3 | C3_Scale | Scale = stretch/flip; scalar | live slider c ∈ [−1.5, 2] on (2,1) | scale1-3 |
| 4 | C4_Combine | Linear combination; span | 2·(2,1)+1·(1,2)=(5,4); parallel → only a line | comb1-4 |
| 5 | C5_Rules | The real definition: vector space rules; functions are vectors | rule list + two waves adding | rules1-4 |
| 6 | C6_ML | Vectors in ML: data rows, averaging images, weighted sum | 2×2 toy images blend; neuron sum | ml1-3 |
| 7 | C7_Code | NumPy demo with real output | code + terminal | code1-2 |
| 8 | C8_Summary | Summary + next video (Matrix) | bullets | sum1-2 |

## Demo
numpy: u+v, 2u, −u, 2u+v, solve a,b for target (5,4), average two toy 2×2 images. Real output.

## Terms (owner = this video)
- vector — Anything you can add to another of its kind and scale by a number, and still get the same kind of thing. Often: a list of numbers.
- component — One entry of a vector: the number in one position of the list.
- scalar — An ordinary single number, used to stretch, shrink or flip a vector.
- linear combination — Scale some vectors, then add them: a·u + b·v.
- span — All the vectors you can reach with linear combinations of a set of vectors.
- vector space — A collection where adding and scaling never take you outside it, and the usual rules hold.

## Symbol & colour ledger (series-wide)
| Symbol | Meaning | Colour |
|---|---|---|
| u | first vector | BLUE |
| v | second vector | GREEN |
| u+v, result | result vector | YELLOW |
| a, b, c (scalars) | numbers that scale | PURPLE |
| x-component / y-component | entries | ORANGE (guide lines) |
| warning / impossible | — | RED |
