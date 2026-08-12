# About

Check section [README#About](./README#about).

# Contributing
* Do not git commit nor stage changes unless the user explicitly commands it.

# Documentation rules

These apply to all code and docs Claude generates or touches.

## Docs workflow

Avoid rewriting doc files multiple times per session. While the code is still undergoing reviews and changes,
do not touch the docs. Just after reaching a stable system design and having it validated by the user, update
doc files.

## In-code documentation
* No "what" or "how" comments inline. Comments explain why — non-obvious constraints, tradeoffs, workarounds.
* Docstrings only on: callable scripts (with parameter descriptions), public class interfaces, and module-level __init__.py / top-of-file summaries only for major subsystems.
* No docstrings on private helpers, simple getters/setters, or anything self-evident from the signature.

## Markdown documentation
* No prose explanations of what a function does or how the system works, unless in a designated start guide or architecture doc.
* README.md scope: repo/folder responsibility (one paragraph max), setup steps, and required external structures (see below). Nothing else.
* If a README documents a config file, a template file (e.g. config.template.yaml) must exist alongside it. Docs and templates are updated together when the code changes.
* If a README documents a required directory structure, represent it as a tree block — not as prose.

# What Claude must NOT generate unless explicitly asked
* Section headers like "Overview", "How it works", "Usage" in module docstrings
* Inline comments restating what the next line of code does
* README sections titled "Features", "Contributing", "License" in internal repos
* Any comment block describing the purpose of a variable
