# Storyboard — Loss function (ML Math Atoms · Video 15 of 16)

Source: `raw/mml-book.pdf` §8.2 (Eq 8.6, Examples 8.1–8.2); wiki `loss-function`,
`empirical-risk-minimization` (GPT loss numbers: Raschka pp. 145, 149). Axiom: **E**.

## Hook
A model needs one number for "how wrong am I". Training = make it smaller. GPT in the book: ~11 → 0.39.

## Prerequisites
Video 13 (mean); video 7 (derivative) for chapter 4.

## Chapters
| # | Scene | Learning goal | Main visual | Keys |
|---|---|---|---|---|
| 0 | C0_Intro | why a loss | model → loss counter | intro1-2 |
| 1 | C1_OnePrediction | loss of one guess: squared, absolute, 0–1 | number line 170 vs 180, curves | one1-5 |
| 2 | C2_Risk | mean loss = empirical risk | 3 toy examples, table, formula | risk1-4 |
| 3 | C3_Fit | parameter w; grid search L(w) | line + loss curve side by side | fit1-5 |
| 4 | C4_Smooth | smooth vs staircase 0–1 loss | tangent on bowl; threshold; staircase | smooth1-4 |
| 5 | C5_Outlier | squared vs absolute with an outlier | two fitted lines | out1-4 |
| 6 | C6_Optimize | axiom E; GD + cross-entropy previews | argmin; real GD steps | opt1-4 |
| 7 | C7_Practice / C7_Code / C8_Summary / C9_Credits | | | |

## Demo
Per-prediction errors for y=(2,4,6), ŷ=(1,4,8); grid search 401 slopes on (1,2),(2,5),(3,6),(4,8) → w=2.07,
L=0.217; 0–1 loss over 701 thresholds → 3 values; outlier fit: squared w=3.82, absolute w=2.00.

## Terms (owner)
loss function, parameter, optimization. Reminders: mean (13), derivative (07).

## Colour ledger
x BLUE · label y GREEN · prediction ŷ YELLOW · parameter w PURPLE · errors/absolute ORANGE · loss RED · labels MUTED.
