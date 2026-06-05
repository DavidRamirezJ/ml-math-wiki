# ML-Mathematics Wiki

An LLM-maintained personal knowledge base for studying the mathematics of machine learning,
**dummy → expert**. Built on the [LLM Wiki pattern](https://en.wikipedia.org/wiki/Memex): the human
curates sources and asks questions; the LLM (Claude Code) writes and maintains every page.

Primary source: *Mathematics for Machine Learning* — Deisenroth, Faisal & Ong (2020), in `raw/`.

## Layout
```
CLAUDE.md          # the schema — rules the LLM follows every session
index.md           # catalog of every page
log.md             # chronological record of ingests / queries / lints
raw/               # immutable source documents (read-only)
assets/            # generated figures + their generator scripts
wiki/
  ├── concepts/    # one page per math concept (3-tier: 🟢 intuition / 🟡 mechanics / 🔴 theory)
  ├── people/      # the mathematicians behind the math
  ├── sources/     # one summary page per ingested chapter
  └── maps/        # overview / learning-path (orientation pages)
```
Current: 11 sources · 96 concepts · 25 people (full book, Ch. 2–12).

## Run it on another computer
1. **Clone:** `git clone <this-repo-url> && cd ai-ml`
2. **Obsidian:** install from https://obsidian.md → *Open folder as vault* → pick this folder.
3. **Claude Code:** install (https://docs.claude.com/claude-code), then `cd` into this folder and
   run `claude`. It auto-reads `CLAUDE.md` and continues maintaining the wiki under the same rules.
4. **Python tooling** (only for new ingests / regenerating animations):
   `pip3 install -r requirements.txt`
   PDF page rendering also needs poppler: `brew install poppler` (macOS).

## How to use it
- **Ingest** a new source: drop it in `raw/`, tell Claude "ingest it."
- **Query:** ask a question; good answers get filed back as new pages.
- **Lint:** ask Claude to health-check for contradictions, orphans, and gaps.

See `CLAUDE.md` for the full conventions.
