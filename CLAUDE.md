# CLAUDE.md — ML Mathematics Wiki Schema

> This file is the **operating contract** for this knowledge base. Every session, read this
> file first, then `index.md`, then `log.md` (last 5 entries). Follow these rules exactly.
> The human curates sources and asks questions. **You** (the LLM) write and maintain the
> entire wiki — every page, link, summary, and bookkeeping entry.

---

## 0. Mission

Build a persistent, interlinked wiki that teaches **the mathematics of machine learning,
from dummy to expert**. Every concept must be explained at three depths so the same page
serves a beginner and an expert. The wiki compounds: each source read makes it richer, each
question answered gets filed back as a page. Never re-derive knowledge that already lives here.

Primary source of truth: `raw/mml-book.pdf` — *Mathematics for Machine Learning*, Deisenroth,
Faisal & Ong (2020), Cambridge University Press. Additional sources welcome.

Second source: `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` — *Build a Large Language Model
(From Scratch)*, Raschka (2025), Manning. Book page N = PDF page N+22. Source pages are prefixed `llm-`.

---

## 1. The three layers

1. **`raw/`** — immutable sources. PDFs, articles, images, datasets. **Never edit these.**
   Read-only source of truth. Cite by filename + page/section.
2. **`wiki/`** — LLM-owned markdown. You create, update, cross-link, and keep consistent.
   The human reads it; you write it.
3. **`CLAUDE.md` (this file)** — the schema. Co-evolve it with the human as conventions improve.

---

## 2. Directory layout

```
ai-ml/
├── CLAUDE.md            # this schema
├── index.md            # content catalog — updated every ingest
├── log.md              # chronological append-only record
├── raw/                # immutable sources (the book lives here)
├── assets/             # downloaded figures/images, named <source>-figN.png
├── study-log/          # personal study journal — one file per study session (NOT in graph)
└── wiki/
    ├── maps/           # big-picture pages: overview, learning-path, synthesis, MOCs
    ├── concepts/       # ONE page per math concept (eigenvalue, gradient, Bayes-theorem…)
    ├── people/         # ONE page per mathematician (Gauss, Bayes, Cholesky…)
    └── sources/        # ONE summary page per ingested source or book chapter
```

**Filenames:** lowercase-kebab-case, `.md`. Concept files match the term: `singular-value-decomposition.md`,
`gradient-descent.md`. People files: `carl-friedrich-gauss.md`. Source files: `mml-ch02-linear-algebra.md`.

---

## 3. Page conventions

### 3.1 Frontmatter (every wiki page)
```yaml
---
title: Eigenvalues and Eigenvectors
type: concept            # concept | person | source | map
tags: [linear-algebra, matrix-decomposition]
difficulty: intermediate # beginner | intermediate | advanced  (concepts only)
sources: [mml-ch04-matrix-decompositions]   # source pages this draws from
created: 2026-06-04
updated: 2026-06-04
---
```
Frontmatter is for Obsidian Dataview. Keep `updated` current. `sources` lists the source
pages (not raw files) the page is built from.

### 3.2 Concept page — MANDATORY three-tier structure
Every concept page MUST contain these sections, in this order. This is the "dummy → expert"
backbone — do not skip a tier.

```
# <Concept Name>

> **One-liner:** <single plain sentence a 12-year-old could follow.>

## 🟢 Intuition (Dummy)
Plain-language analogy, no jargon, no formulas. Why should anyone care? A picture in words.

## 🟡 Mechanics (Practitioner)
The actual definition, the key formula(s) in LaTeX, a small worked numeric example,
and how/where it's used in ML. This is the "I need to use this" tier.

## 🔴 Theory (Expert)
Formal definition, proofs or proof sketches, edge cases, connections to deeper structure,
generalizations. Cite theorems/sources precisely.

## Connections
- Builds on: [[prerequisite-concept]]
- Leads to: [[downstream-concept]]
- Used by: [[ml-method-that-uses-this]]
- People: [[mathematician]]

## Sources
- `raw/mml-book.pdf` §4.2, pp. 105–114
```

Use `$...$` for inline LaTeX and `$$...$$` for display math (Obsidian renders these).

### 3.3 Person page
```
# <Full Name> (years)

> **Why they matter to ML:** <one line>

## Life (brief)
## Key contributions
For each: what it is, what it's called today, and the [[concept]] page it maps to.
## Connections
## Sources
```

### 3.4 Source page (one per ingested source/chapter)
```
# <Source Title>

- **Raw file:** `raw/<file>` (§ / pages)
- **Type:** book chapter | paper | article
- **Ingested:** YYYY-MM-DD

## TL;DR (5 bullets)
## Key concepts introduced
Bulleted list, each linking to its [[concept]] page.
## Notable results / theorems
## What this changes in the wiki
Note any contradictions with, or refinements to, existing pages.
```

---

## 4. Linking rules

- Use `[[wikilinks]]` for every concept, person, and source you mention that has (or should
  have) a page. Liberal linking is good — a link to a page that doesn't exist yet is a valid
  TODO marker, not an error.
- Every concept page must link **at least one** prerequisite and (where applicable) one
  downstream concept. No orphans.
- When you name a mathematician, link their `[[person-page]]` and create a stub if missing.

---

## 5. Operations (workflows)

### 5.1 INGEST — when the human adds/points to a source
1. **Read** the raw source (the relevant section). For PDFs, use `pdftotext`/`pdftoppm`
   (poppler) to extract text/pages. Read text first; view figures separately if needed.
2. **Discuss** key takeaways with the human before writing. Confirm scope (which chapter/section).
3. **Write a source page** in `wiki/sources/` using the §3.4 template.
4. **Create/update concept pages** in `wiki/concepts/` — one per concept, three-tier structure.
   A single chapter typically touches 8–15 concept pages.
5. **Create/update person pages** for any mathematician named.
6. **Update `index.md`** — add new pages under the right category with one-line summaries.
7. **Update relevant map pages** (overview, learning-path) so the big picture stays current.
8. **Append to `log.md`** using the exact prefix format (§6).
9. **Flag contradictions:** if a new source disagrees with an existing page, note it explicitly
   on both pages under a `> ⚠️ Contradiction:` callout — never silently overwrite.

### 5.2 QUERY — when the human asks a question
1. Read `index.md` to locate relevant pages. Drill into them. Search `wiki/` if needed.
2. Synthesize an answer **with citations** to wiki pages and raw sources.
3. **File valuable answers back into the wiki** — a comparison, a derivation, a worked
   example becomes a new `concept` or `map` page. Good explorations should not vanish into chat.
4. Append a `query` entry to `log.md` noting what was asked and any page created.

### 5.3 LINT — when the human asks for a health check
Scan the wiki and report:
- Contradictions between pages
- Stale claims superseded by newer sources
- Orphan pages (no inbound links) — `grep -L` style check
- Concepts mentioned but lacking their own page (dangling `[[links]]`)
- Missing prerequisite/downstream cross-references
- Difficulty tiers that are incomplete (a concept missing its 🔴 Theory section, etc.)
- Data gaps a web search or new source could fill
Then suggest next sources to ingest and next questions to investigate.
Append a `lint` entry to `log.md`.

### 5.4 STUDY — when a study skill runs in this folder
Whenever a teaching/study skill (e.g. `/teach`) is invoked here, you MUST keep a traceability
record of what the human studied. This is for the **learner**, not the wiki.
1. **At session start**, create a session file `study-log/YYYY-MM-DD-<topic>.md` using the
   template in `study-log/README.md`. Record date, skill used, source material, and the goal.
2. **During/after the session**, fill in what was covered, the takeaways that stuck, a mastery
   checklist, and open questions to revisit. Append — never rewrite a prior session.
3. **Add a row** to the index table in `study-log/README.md`.
4. **Keep it off the graph:** session files use **plain backtick references** to wiki pages
   (e.g. `singular-value-decomposition`), NEVER `[[wikilinks]]`. The folder is also excluded by
   the graph filter in `.obsidian/graph.json` (`-path:study-log/`).
5. The study log is independent of `log.md` — do not duplicate study sessions there (a `log.md`
   `query` entry is still fine if you also created a wiki page).

---

## 6. Log format (append-only)

Every entry starts with this exact prefix so it's greppable:
```
## [YYYY-MM-DD] <op> | <short title>
```
where `<op>` ∈ `setup | ingest | query | lint | refactor`.
Quick recent history: `grep "^## \[" log.md | tail -5`.
Newest entries go at the **bottom** of the file. Never rewrite past entries.

---

## 7. Index format

`index.md` is the catalog. Organized by category (Maps, Concepts, People, Sources). Each line:
```
- [[page-name]] — one-line summary (difficulty, N sources)
```
Update it on **every** ingest and whenever you create a page. Keep entries alphabetized within
each category.

---

## 8. House style

- **LaTeX** for all math. `$inline$`, `$$display$$`.
- **Define before use.** If a page uses a term, link it or define it.
- **Be precise with citations** — page numbers and section numbers, not vibes.
- **Three tiers always.** Never write a concept page that only an expert could read.
- **Concise prose.** No filler. The human is here to learn, not to read padding.
- **You do the bookkeeping.** Updating links, index, and log is your job, every time —
  near-zero cost for you, the whole point of the system.

---

## 9. Session bootstrap checklist

At the start of every session, before doing anything else:
1. Read this file (`CLAUDE.md`).
2. Read `index.md` to know what already exists.
3. `grep "^## \[" log.md | tail -5` to see recent activity.
4. Then act on the human's request following the workflows above.
