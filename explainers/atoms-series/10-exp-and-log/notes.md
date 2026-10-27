# Notes — 10 exp & log
- Duration: 12.3 min (final.mp4, 1080p30, subtitles + chapters). Narration 1768 words / 32 blocks; measure.py OK (11.84 min narration).
- Coverage: e and repeated growth ((1+1/n)^n table), e^x > 0, e^(a+b)=e^a e^b (7.389 × 20.086 = 148.41), ln as inverse,
  ln values / domain, log(ab)=log a+log b (ln2+ln3 = ln6 = 1.792), power and division rules, log tables,
  tiny probabilities (0.0001 → −9.21, ÷10 → −2.30), float64 underflow (product 0.0 after 162 factors), sum of logs −4605.17,
  slopes (e^x)'=e^x, (ln x)'=1/x, floor vs axioms, softmax positivity preview, log-likelihood (0.9·0.8·0.5), cross-entropy mention, stability.
- Wiki: log-probability (ML part). Axiom: floor (school math).
- Terms owned: exponential, logarithm. Reminder: derivative (07). Mentioned only: softmax (molecule), cross-entropy (video 15), chain rule (09).
- Demo: real NumPy output; numbers in narration match (148.4132, 1.7918, −9.21, 0.0, −4605.17, 162, [0.705 0.259 0.035]).
- QA log: draft 1 → 0 layout warnings; contact sheets: −4605.17 cut at right edge (re-centred), ÷10 glyph enlarged,
  practice question appeared after narration ended (moved to "Without"). Draft 2 clean. Final: no AUDIO PROBLEM.
- Simplification: "float64 cannot store positive numbers smaller than about 10^-324" (smallest subnormal ≈ 4.9e-324).
