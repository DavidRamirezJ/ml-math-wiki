# Notes — 06 Norm
- Duration: 13.7 min, 1080p30, subtitles + chapters. Narration 1958 words / 38 blocks (measure.py 13.26 min, OK).
- Coverage: wiki `norm`, `metric`; MML §3.1, §3.3; LLM book §D.3 (clipping). Axiom B.
- New terms (owner): norm, unit vector, distance. Reminders: dot product, cosine similarity (video 5).
  "metric" is defined in speech + on-screen title only (not in term_plan).
- Honest note: L1 and L∞ are norms not induced by any inner product (parallelogram law fails) — stated as a known fact.
- Demo: np.linalg.norm with ord=1/2/inf, normalize, distance √22, triangle check, clipping, nearest neighbour — real output.
- QA log: draft 1 → 2 layout warnings (reminder cards over headings) → moved to DR; fixed Rule-2 label touching
  heading, clip circle leaving the frame (now a quarter arc), "‖" glyph in Menlo readout, code panel too small
  (switched to `from numpy.linalg import norm`). Draft 2: 0 warnings. Final: 0 warnings, no AUDIO PROBLEM.
