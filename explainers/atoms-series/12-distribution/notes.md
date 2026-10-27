# Notes — 12 Probability distribution
- Duration: 11.2 min (final.mp4, 1080p30, subtitles + chapters). Narration 1610 words / 30 blocks; measure.py OK (10.79 min narration).
- Coverage: dice-sum distribution by counting (P(2)=1/36≈0.028, P(7)=6/36≈0.167), p ≥ 0 and sum to 1, histogram vs distribution
  (1680/10000 from video 11), density as area (uniform 0–10 min: P(2≤T≤5)=0.3, total 1; bell slice; exact value → 0),
  toy joint table rain × umbrella (0.25/0.05/0.10/0.60), sum rule → marginals (0.30/0.70; 0.35/0.65), conditional 0.25/0.30≈0.833,
  product rule check, independence (coin+die 1/12; table NOT independent: 0.105 ≠ 0.25), Bayes derived in 3 lines → p(rain|umbrella)≈0.714,
  ML: classifier p(class|photo), LM chain of next-token conditionals, i.i.d. data.
- Terms owned: probability distribution, sum rule, product rule, conditional probability, independence. Bayes derived, no card.
- Demo: real NumPy output (counts, marginals, conditional, product rule, outer-product independence check, Bayes).
- QA log: draft 1 → 0 layout warnings; contact sheets: table label crowding, density formula at right edge, joint-table total written
  after narration ended, small column headers, Bayes label placement, tiny code (shortened comments) — all fixed, re-drafted. Final: no AUDIO PROBLEM.
- Simplification: the bell curve is labelled "illustrative" (no parameters claimed); density can exceed 1 not mentioned.
