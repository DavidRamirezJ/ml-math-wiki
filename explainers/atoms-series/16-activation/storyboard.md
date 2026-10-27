# Storyboard — Activation function (ML Math Atoms · Video 16 of 16)

Source: wiki `activation-function`, `relu` (Raschka §4.2–4.3, §A.5). Axiom: **A** + one deliberate nonlinearity.

## Hook
Without a bend, 100 linear layers = 1 layer. One small bend makes depth useful.

## Prerequisites
Video 2 (matrix multiplication), video 7 (derivative).

## Chapters
| # | Scene | Learning goal | Main visual | Keys |
|---|---|---|---|---|
| 0 | C0_Intro | why bends | 100 layers collapse to one | intro1-2 |
| 1 | C1_Collapse | W2(W1x) = (W2W1)x with real 2×2 numbers | equations | col1-4 |
| 2 | C2_ReLU | nonlinearity, activation, ReLU values, breaks the collapse | plot, chain | relu1-4 |
| 3 | C3_Shapes | sigmoid (0,1), tanh (−1,1), GELU = zΦ(z) | two plots | shape1-4 |
| 4 | C4_Bends | sum of shifted ReLUs ≈ x² | pieces added one by one | bend1-4 |
| 5 | C5_Slope | slope 0/1; dead ReLU | ReLU and its slope; dead unit | slope1-4 |
| 6 | C6_ML | MLP (book A.5), GPT-2 FFN with GELU, axiom | block diagrams | ml1-3 |
| 7 | C7_Practice / C7_Code / C8_Summary / C9_Credits | | | |

## Demo
W1=[[1,2],[0,1]], W2=[[2,0],[1,1]], x=(3,−1): both give (2,0), with ReLU (2,1); ReLU values; ReLU weights
[−3,2,2,2] for x² with max error 0.25 (8 pieces 0.0625); numeric slopes 0/1; dead unit 0 of 11; GELU(±1).

## Terms (owner)
activation function, ReLU, nonlinearity. Reminders: matrix multiplication (02), derivative (07).

## Colour ledger
x BLUE · weights/matrices PURPLE · outputs, ReLU YELLOW · sigmoid GREEN · tanh ORANGE · GELU BLUE (own panel) · warnings RED.
