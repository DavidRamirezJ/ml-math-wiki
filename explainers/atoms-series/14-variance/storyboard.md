# Storyboard — Variance & covariance (ML Math Atoms · Video 14 of 16)

Source: `raw/mml-book.pdf` §6.4.1–6.4.4 (Defs 6.5–6.9, Eq 6.44); wiki `covariance-and-variance`
(LayerNorm numbers from Raschka §4.2). Axioms: **A + B + D**.

## Hook (first 30 s)
Two classes, same mean quiz score 5, very different spread. The mean hides it; ML (standardization,
LayerNorm in every GPT block) needs a number for spread.

## Prerequisites
Video 13 (expected value / mean); video 5 (dot product) helps in chapter 5.

## Chapters
| # | Scene | Learning goal | Main visual | Keys |
|---|---|---|---|---|
| 0 | C0_Intro | why spread matters | two dot plots, plan | intro1-2 |
| 1 | C1_Spread | same mean, different spread; deviations cancel | dot plots, deviation arrows, sum = 0 | spread1-4 |
| 2 | C2_Variance | Var = E[(X−μ)²], worked on A and B | squares of deviations, table | var1-5 |
| 3 | C3_StdDev | σ = √Var; shortcut E[X²]−μ² | area-4 square with side 2; check 29−25 | sd1-4 |
| 4 | C4_Covariance | sign of Cov; worked example | 3 scatter clouds, quadrants, table | cov1-6 |
| 5 | C5_Matrix | 2×2 covariance matrix; dot-product view; axioms; Bessel | matrices, deviation vectors | mat1-5 |
| 6 | C6_ML | z-score, LayerNorm, PCA preview | number lines, formula, PCA arrow | ml1-4 |
| 7 | C7_Practice | 2 questions | | prac1-4 |
| 8 | C7_Code | NumPy demo | code + terminal | code1-2 |
| 9 | C8_Summary / C9_Credits | | | sum1-2 |

## Demo
Class A [4,5,5,5,6], class B [2,4,5,6,8]: deviations, Var 0.4 / 4, σ 2, shortcut 4; hours x=[1,3,5,7],
scores y=[2,3,6,5]: Cov 3, matrix [[5,3],[3,2.5]] (1/n) vs np.cov default (n−1); seeded scatter covs
0.59 / −0.65 / 0.10; z-scores of B.

## Terms (owner = this video)
variance, standard deviation, covariance. Reminders: expected value (13), dot product (05).

## Colour ledger
x / class A BLUE · y / class B GREEN · mean, results YELLOW · deviations ORANGE · probabilities,
E[·] PURPLE · warnings RED · labels MUTED.
