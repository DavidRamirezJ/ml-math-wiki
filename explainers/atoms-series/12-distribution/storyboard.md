# Storyboard — Probability distribution (ML Math Atoms · Video 12 of 16)
Source: wiki `probability-distribution`, `statistical-independence` (MML §6.2–6.4). Axiom: **D — sum rule + product rule**.
## Hook
A classifier answers "cat 0.9, dog 0.08, fox 0.02" — a distribution; strict rules behind it.
## Chapters
| # | Scene | Goal | Visual |
|---|---|---|---|
| 0 | C0_Intro | hook + plan | class bars; plan list |
| 1 | C1_Table | dice-sum distribution by counting; P(2)=1/36, P(7)=6/36; ≥0, sums to 1; vs simulation | table, BarChart |
| 2 | C2_Density | density, probability = area (uniform bus wait: 3×0.1 = 0.3; total 1); bell slice; exact value → 0 | axes, shaded areas |
| 3 | C3_Joint | toy joint table rain × umbrella (0.25, 0.05, 0.10, 0.60) | 2×2 table |
| 4 | C4_Rules | sum rule (marginals 0.30/0.70, 0.35/0.65), conditional 0.833, product rule check | table + formulas |
| 5 | C5_IndepBayes | independence (coin+die 1/12); table NOT independent (0.105 ≠ 0.25); Bayes in 3 lines → 0.714 | formulas |
| 6 | C6_ML | classifier p(class|photo); LM chain of conditionals; i.i.d. data | bars, formulas |
| 7 | C7_Practice | p(no umbrella|rain)=0.167; P(X≥11)=3/36 | worked answers |
| 8 | C7_Code | NumPy demo | code + terminal |
| 9 | C8_Summary | summary + next (expected value) | bullets |
## Terms owned: probability distribution, sum rule, product rule, conditional probability, independence. Bayes: derived, no card (molecule).
## Colour ledger: weather rows BLUE · umbrella columns GREEN · probabilities YELLOW · marginals ORANGE · conditionals PURPLE · not-equal RED.
