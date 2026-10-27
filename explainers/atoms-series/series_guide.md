# ML Math Atoms — production guide (read fully before starting)

Series: one 10–15 min narrated Manim video per math atom. Goal: the viewer truly UNDERSTANDS
the atom and can rebuild it from first principles. Video 01 (`01-vector/`) is the APPROVED PILOT —
copy its structure, pacing, helpers and style exactly.

## Paths and commands
- Series root: `/Users/drjx/Documents/ai-ml-math/ai-ml/explainers/atoms-series/`
- `PY=~/.venvs/explainer-video/bin/python` · `SK=~/.claude/skills/explainer-video/scripts`
- Scaffold: `$PY -I $SK/new_video.py <root>/<folder> --title "<Title>" --series "ML Math Atoms · Video N of 16"`
- Measure: `$PY -I $SK/measure.py <dir> --min 10 --max 14.5` (must print OK)
- Draft: `$PY -I $SK/build.py <dir> --draft > /tmp/<folder>_draft.log 2>&1; tail -15 /tmp/<folder>_draft.log`
- Contact sheets: `$PY -I $SK/contact_sheet.py <dir> --blocks` then Read each `media/qa/sheet_*.png` ONE AT A TIME
- Fix + partial redraft: `build.py <dir> --draft --only C3_X C5_Y`
- Final: `$PY -I $SK/build.py <dir> > /tmp/<folder>_final.log 2>&1; tail -6 ...` → final.mp4 (1080p)
- Rendering is slow: run long builds with a generous timeout (up to 600000 ms) or in the background.
- Skill rules: `~/.claude/skills/explainer-video/reference/style-guide.md` and `manim-cheatsheet.md`
  (Manim Community 0.22 — only cheat-sheet APIs). Read both once.

## Files you may touch
ONLY your own video folders. Never edit `common.py`, `curriculum.py`, `term_plan.py`, this guide,
the skill toolkit, `01-vector/`, or other workers' folders. Read 01-vector files for reference.

## Required structure (same as pilot)
1. `C0_Intro` — series line + title; hook in first 30 s (why it matters in ML, concrete); what
   the viewer will be able to do; prerequisites ("you need video NN").
2. 4–6 teaching chapters, each ONE learning goal, concrete before abstract:
   intuition with a picture/analogy → the formal definition / formula as `MathTex` split into
   coloured parts + a legend that reads every symbol in words → **at least 2 fully worked numeric
   examples** across the video → edge cases / common mistake → where it comes from (which
   **axiom**: A linearity, B inner product, C local linearity, D probability rules,
   E learning = optimization; "floor" = school math) → where it is used in ML.
3. `C7_Practice` "Try it yourself": 2 questions, "pause and try", then worked answers.
4. Python demo chapter: `demo.py` with NumPy (offline, < 20 s, ≤ 16 output lines), real output
   via `run_python()` + `self.terminal()`; say "This output is real". Numbers shown in animations
   come from `demo` (import it like the pilot).
5. `C8_Summary` — heading "In summary" in YELLOW, bullets, then "Next · Video N+1: <title>" teaser
   (video 16: "That completes the atoms — next: the molecules map"). Ends with the FadeOut of all.
6. `C9_Credits` — ALWAYS last, exactly:
   ```python
   sys.path.insert(0, str(HERE.parent))          # right after `from narration import N`
   from common import Credits  # noqa: E402
   class C9_Credits(Credits):
       pass
   ```
   and add `"C9_Credits"` to SCENES and `"C9_Credits": "Credits"` to CHAPTER_TITLES.
   Do NOT put a credits block in narration.py — common.py has it.

## Narration
- Natural friendly English for a Spanish speaker learning English: sentences ≤ ~25 words, one
  idea each, define every term at first use, same word for same thing, no idioms/slang/filler.
- Blocks ≤ ~75 words; 1,550–2,000 words total; measure.py must print OK.
- Write math as speakable words ("x squared", "the sum over i"). Numbers spelled as words in
  narration are fine; on screen use digits/MathTex.
- Every `self.wait_until(v, "word", nth)` word MUST occur that many times in that block
  (matching is case-insensitive, punctuation stripped) — count carefully; this was the main pilot bug.

## Terms
`term_plan.py` OWNER maps each term → owning video. Your video shows `self.term_card` (yellow)
ONLY for terms it owns; for terms owned by earlier videos use `self.reminder_card` once; never
card a term owned by a later video (say one sentence + "we will study it in video NN").

## Colour ledger (series-wide, keep identical)
first vector/input u, x → BLUE · second vector v → GREEN · result / output / highlight → YELLOW ·
scalars, weights, parameters → PURPLE · components / guide lines → ORANGE · warning, impossible,
error → RED · labels → MUTED. Background/fonts come from studio.py.

## Accuracy
Content source of truth: the wiki concept pages listed for your video in
`/Users/drjx/Documents/ai-ml-math/ai-ml/wiki/concepts/<page>.md` (read them — they cite
`raw/mml-book.pdf` sections) and `wiki/maps/atoms-and-molecules.md` (axioms). Never invent facts,
numbers, dates or APIs. Toy data is labelled "toy example". Every number on screen must be correct —
re-check arithmetic in demo.py.

## QA (mandatory — you must LOOK)
Draft → read LAYOUT WARNINGS → fix real ones → contact sheets → read EVERY sheet image → fix overlaps,
off-frame text (keep x ∈ [−7.1, 7.1], y ∈ [−4, 4]), term cards covering content, tiny text, wrong
numbers, leftover Circumscribe outlines (prefer Indicate) → re-draft changed chapters → final.
Check the final log has no "AUDIO PROBLEM". Write `notes.md` (duration, coverage, terms, QA log).

## Context budget (workers crashed before when ignoring this)
Print only the last ~15 lines of build logs (redirect to /tmp files). Read one contact sheet at a time.
Never cat big files; never re-read files you wrote. Finish one video completely (final.mp4) before
starting the next.

## Report back (short)
Per video: folder, final duration, number of layout warnings left (should be 0), AUDIO PROBLEM yes/no,
anything simplified or uncertain.
