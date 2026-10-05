# Writing documentation

The README is the repository and package-index landing page. The MkDocs site is
the canonical location for procedures, concepts, troubleshooting, and reference
material.

## Preview locally

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

English is built at `/`; Japanese is built at `/ja/`. Detailed pages currently
fall back to English when no Japanese source file exists.

## Validate before review

```bash
uv run --group docs mkdocs build --strict
```

Strict mode rejects missing pages, invalid anchors, omitted navigation pages, and
other MkDocs warnings. The same command runs in the documentation CI workflow.
Generated output belongs in `site/` and is not committed.

## Add or move a page

1. Add the English source below `docs/en/`.
2. Add the page to `nav` in `mkdocs.yml`.
3. Use relative links to Markdown source paths.
4. Add a Japanese file at the same path below `docs/ja/` only when it is being
   maintained as a translation.
5. Run the strict build.

## Keep one source of truth

| Information | Canonical source |
| --- | --- |
| Python versions and package extras | `pyproject.toml` |
| CLI arguments and defaults | Typer definitions in `ghand.cli` |
| MuJoCo physics defaults | `MujocoPhysicsProfile` |
| Python signatures and object docs | source types and docstrings |
| Procedures and explanations | MkDocs pages |
| Project summary and first safe command | root README files |

Do not manually copy a complete CLI option table into a guide. Link the generated
reference and explain only the workflow-specific choices. When a default value is
important to a concept, point to the defining object or show code that reads it.

## Generated references

`mkdocs-typer2` imports `ghand.cli` and renders the command tree in
`reference/cli.md`. `mkdocstrings` reads the selected package surfaces from
`src/ghand` for the Python API pages.

Adding a command, changing a help string, or changing a published subpackage
object is therefore visible in the next docs build. The surrounding guide still
needs an update when the user workflow or meaning changes.

Publishing and versioned documentation are intentionally outside this build-only
workflow until their repository policy is decided.
