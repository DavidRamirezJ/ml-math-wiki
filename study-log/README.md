# Study Log

This folder is the **personal study journal** for this vault. It is separate from the
curated `wiki/` (the polished knowledge base) and from `log.md` (the wiki-maintenance record).

**Purpose:** traceability of *what the human actually studied* — which topics, when, with which
skill, how it went, and what to revisit. The wiki teaches the subject; the study log tracks the
learner.

## Rules

- **One file per study session**, named `YYYY-MM-DD-<topic>.md` (e.g. `2026-06-18-svd.md`).
  If a second session happens the same day on the same topic, append `-2`.
- **Kept off the Obsidian graph.** The graph filter excludes `study-log/`
  (`.obsidian/graph.json` → `search: "... -path:study-log/"`). Also, session files use **plain
  text references** to wiki pages, *not* `[[wikilinks]]`, so they create no graph edges.
  To reference a wiki page, write its name in backticks: `singular-value-decomposition`.
- **Append, never rewrite.** Past sessions are a record; don't edit them after the fact.
- The LLM (Claude) writes these automatically whenever a study skill (e.g. `/teach`) runs in
  this folder.

## Session file template

```markdown
# Study Session — <Topic>

- **Date:** YYYY-MM-DD
- **Skill:** /teach (or other)
- **Source material:** wiki page(s) / raw source studied (plain names, no wikilinks)
- **Goal:** what the human wanted to understand

## What we covered
Bulleted summary of the concepts walked through.

## Key takeaways (in the human's words)
The mental models / explanations that stuck.

## Mastery check
- [x] Could explain X back
- [ ] Still shaky on Y

## Open questions / revisit next
- ...

## Artifacts
- Any HTML lesson / file generated this session.
```

## Index

| Date | Topic | Skill | Status |
|------|-------|-------|--------|
| 2026-06-18 | Vector | /teach | ✅ mastered |
| 2026-06-22 | Matrix | /teach | ✅ mastered |
| 2026-06-25 | Vector Space | /teach | ✅ mastered |
