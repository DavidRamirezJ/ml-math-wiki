# Storyboard — exp & log (ML Math Atoms · Video 10 of 16)

Source: school math (the "floor"); wiki `log-probability` (Raschka §5.1.2, book pp. 136–138) for the ML part.
Axiom: **floor** (school math) — not one of the five axioms A–E.

## Hook (first 30 s)
Multiply 1000 probabilities of 0.01 (a language model scoring a text) and the computer answers exactly 0.0.
exp and log are the pair that fixes this.

## Prerequisites
Powers (school arithmetic); derivative (video 07) for chapter 5; chain rule (video 09) mentioned.

## Chapters
| # | Scene | Learning goal | Main visual | Keys |
|---|---|---|---|---|
| 0 | C0_Intro | Hook: product underflows to 0.0 | 0.01×…×0.01 = 0.0; exp ⇄ log | intro1-2 |
| 1 | C1_Exp | Repeated growth, e, e^x positive, e^(a+b)=e^a e^b | doubling dots, (1+1/n)^n table, plot | exp1-4 |
| 2 | C2_Log | log = what power?; inverse; values; domain | e^? = 7.389; machines; mirror graph | log1-3 |
| 3 | C3_Products | log(ab)=log a+log b; check ln2+ln3=ln6; power/division rules; log tables | derivation, flow | prod1-4 |
| 4 | C4_Tiny | tiny p → friendly negatives; ÷10 → −2.30 steps; underflow at 162 factors; −4605.17 | vertical ln-p ruler, live counter | tiny1-4 |
| 5 | C5_Slopes | (e^x)'=e^x, (ln x)'=1/x; floor vs axioms | live tangents; pillars on floor | slope1-3 |
| 6 | C6_ML | softmax positivity (preview), log-likelihood, cross-entropy, stability | bars scores→exp→probs; 0.9·0.8·0.5 | ml1-4 |
| 7 | C7_Practice | ln 10 from ln 2, ln 5; four probabilities ½ | worked answers | prac1-4 |
| 8 | C7_Code | NumPy demo with real output | code + terminal | code1-2 |
| 9 | C8_Summary | Summary + next (Random variable) | bullets | sum1-2 |

## Demo
exp rule, log rule, ln 0.0001 = −9.21, prod of 1000×0.01 = 0.0, sum of logs = −4605.17, zero after 162 factors, softmax([2,1,−1]) = [0.705 0.259 0.035].

## Terms (owner = this video)
- exponential — The function e to the power x, with e ≈ 2.718: repeated growth that is always positive.
- logarithm — The power you must put on e to get a number: ln y = x means e^x = y.
Reminder: derivative (07). Mentioned only: softmax (molecule), cross-entropy loss (video 15 area), chain rule (09).

## Symbol & colour ledger
x input BLUE · e^x curve GREEN · ln / powers / logs PURPLE · results YELLOW · ÷10 steps ORANGE · underflow / undefined RED.
