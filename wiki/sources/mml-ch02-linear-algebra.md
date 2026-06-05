---
title: "MML Ch.2 — Linear Algebra"
type: source
tags: [linear-algebra, source]
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch.2 — Linear Algebra

- **Raw file:** `raw/mml-book.pdf` §2, **printed pp. 17–69** (PDF pages 22–74)
- **Type:** book chapter — *Mathematics for Machine Learning*, Deisenroth, Faisal & Ong (2020)
- **Ingested:** 2026-06-04 · **Validated against source:** 2026-06-04
- *Page rule:* printed page = PDF page index − 5.

## Section coverage (all of §2.1–2.9 accounted for)
| § | Section title | Printed pp. | Wiki coverage |
|---|---|---|---|
| 2.1 | Systems of Linear Equations | 19–21 | ✅ [[system-of-linear-equations]] |
| 2.2 | Matrices | 22–26 | ✅ [[matrix]] |
| 2.3 | Solving Systems of Linear Equations | 27–35 | ✅ [[gaussian-elimination]] |
| 2.4 | Vector Spaces | 35–40 | ✅ [[vector-space]] |
| 2.5 | Linear Independence | 40–44 | ✅ [[linear-independence]] |
| 2.6 | Basis and Rank | 44–48 | ✅ [[basis-and-rank]] |
| 2.7 | Linear Mappings | 48–60 | ✅ [[linear-mapping]] |
| 2.8 | Affine Spaces | 61–62 | ✅ [[affine-space]] |
| 2.9 | Further Reading | 63–64 | — (pointers only) |
| — | Exercises | 64–69 | — |

> ✅ **All of §2.1–2.8 now have full three-tier pages.** §2.9 is further-reading pointers only.

## TL;DR (5 bullets)
- **Linear algebra is the study of vectors** and the rules for adding them and scaling them. A "vector" is *any* object closed under addition and scalar multiplication — geometric arrows, polynomials, audio signals, and tuples in $\mathbb{R}^n$ all qualify.
- The chapter centers on **$\mathbb{R}^n$** because most ML algorithms are formulated there and data is represented as vectors in $\mathbb{R}^n$ (see Ch. 8).
- **Systems of linear equations** are the workhorse problem: they have either *no*, *exactly one*, or *infinitely many* solutions, and are solved compactly with **matrices** via **Gaussian elimination**.
- Key structural ideas build in a chain: **vector → vector space → linear independence → basis & rank → linear mappings → affine spaces.**
- The concept of **closure** ("what set results from my operations?") is the thread that leads to the definition of a [[vector-space]], which "underlies much of machine learning."

## Key concepts introduced
- [[vector]] — objects you can add and scale
- [[vector-space]] — the set closed under those two operations
- [[system-of-linear-equations]] — central problem; no / one / infinite solutions
- [[matrix]] — compact notation for systems and linear mappings
- [[gaussian-elimination]] — algorithm for solving systems (REF/RREF, minus-1 trick)
- [[linear-independence]] — when no vector is a combination of the others
- [[basis-and-rank]] — minimal spanning set; dimension and rank
- [[linear-mapping]] — structure-preserving maps between vector spaces
- [[affine-space]] — vector space shifted off the origin

## Notable results / theorems
- A real system of linear equations has **no solution, exactly one, or infinitely many** (p. 21). Never "exactly two."
- **Geometric interpretation:** with 2 variables each equation is a line; the solution set is their intersection (a point, a line, or empty). With 3 variables, each equation is a plane (pp. 20–21).
- Linear regression (Ch. 9) solves a least-squares version of a system that cannot be solved exactly.

## What this changes in the wiki
Establishes the entire linear-algebra foundation. **All 8 concept pages (§2.1–2.8) are now full
three-tier pages**, plus [[carl-friedrich-gauss]]. Chapter coverage is complete except §2.9
(further-reading pointers). No contradictions. Next chapters to ingest: Ch. 3 (Analytic Geometry)
and Ch. 4 (Matrix Decompositions) build directly on these pages.
