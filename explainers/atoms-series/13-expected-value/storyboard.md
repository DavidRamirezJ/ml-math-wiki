# Storyboard — Expected value (ML Math Atoms · Video 13 of 16)
Source: wiki `expected-value-and-mean` (MML §6.4.1, Defs 6.3, 6.4, 6.9). Axioms: **A (linearity) + D (probability rules)**.
## Hook
A casino cannot predict one game but can predict a million; a model's error on one example vs its average error.
## Chapters
| # | Scene | Goal | Visual |
|---|---|---|---|
| 0 | C0_Intro | hook + plan | one game vs many |
| 1 | C1_Weighted | grade weighted average 0.3·80+0.7·90 = 87; probabilities as weights | bars, formula |
| 2 | C2_Definition | E[X]=Σ p(x)x with legend; die 21/6 = 3.5 (never rollable, balance point); toy lottery −1 € | bars, ruler + fulcrum, table |
| 3 | C3_Linearity | E[aX+bY]=aE[X]+bE[Y]; why (sum rule); two dice → 7; E[2X+1]=8; E[X²]=91/6 ≠ 12.25 | formulas |
| 4 | C4_LongRun | seeded running average: 2.9 → 3.46 → 3.453 → 3.4883; law of large numbers | live log-scale plot |
| 5 | C5_DataMean | data 2,4,4,5,10 → mean 5 = E under pick-a-row; E[X] vs x̄ | dot plot |
| 6 | C6_ML | average loss (toy 0.2,0.5,0.1,1.2 → 0.5) ≈ expected loss; batches, derivatives | bars |
| 7 | C7_Practice | E = 2.6; three dice 10.5; E[3X−2] = 8.5 | worked answers |
| 8 | C7_Code | NumPy demo | code + terminal |
| 9 | C8_Summary | summary + next (variance & covariance) | bullets |
## Terms owned: expected value, mean. Reminder: probability distribution (12). Mentioned only: loss (video 15).
## Colour ledger: values x BLUE · probabilities/weights PURPLE · expected value/result YELLOW · losses/warnings RED · Σ ORANGE.
