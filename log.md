# 📜 Log

Append-only, chronological. Newest at the bottom. Each entry prefix: `## [YYYY-MM-DD] <op> | <title>`.
Recent activity: `grep "^## \[" log.md | tail -5`.

---

## [2026-06-04] setup | Wiki initialized
- Created schema `CLAUDE.md` (three-tier dummy→expert concept structure, ingest/query/lint workflows).
- Created `index.md` (catalog) and `log.md` (this file).
- Folder layout: `raw/`, `assets/`, `wiki/{maps,concepts,people,sources}`.
- Initialized git repo.
- Moved `mml-book.pdf` → `raw/mml-book.pdf` (Mathematics for ML, Deisenroth/Faisal/Ong 2020, 417 pp).
- Next: first ingest — Chapter 2, Linear Algebra.

## [2026-06-04] ingest | MML Ch.2 — Linear Algebra
- Source: `raw/mml-book.pdf` §2, pp. 17–73. Extracted text via pypdf (poppler also now installed).
- Created source page [[mml-ch02-linear-algebra]].
- Created 3 full three-tier concept pages: [[vector]], [[vector-space]], [[system-of-linear-equations]].
- Created person page [[carl-friedrich-gauss]].
- Seeded 6 concept stubs (matrix, gaussian-elimination, linear-independence, basis-and-rank, linear-mapping, affine-space) — to be filled on next pass; lint will flag.
- Updated [[index]] (counts: 1 source, 3 concepts +6 stubs, 1 person).
- No contradictions (first content ingest).
