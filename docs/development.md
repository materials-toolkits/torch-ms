# Development

To use TorchMS in your own project, see [Installation](guide/installation.md). This page is
for working on the library itself.

The project uses [uv](https://docs.astral.sh/uv/) to manage its environment:

```bash
git clone https://gitlab.univ-artois.fr/astrid.klipfel/torch-ms.git
cd torch-ms
uv sync
```

`uv sync` installs the library in editable mode with the development group (tests, type
checker, ASE, pymatgen, pyarrow, and RDKit for the script building ZINC250k). Always run Python through `uv run`, which uses the
environment of the project.

| Need | Command | What it adds |
|---|---|---|
| Loading datasets and the estimator from Hugging Face | `uv sync --extra hub` | `huggingface_hub` |
| Building this documentation | `uv sync --group docs` | MkDocs Material, mkdocstrings |

## Checks

```bash
uv run pytest          # tests, including the examples of the docstrings
uv run pyright src     # type checking
uv run black .         # formatting
uv run pylint src      # linter
```

## Building the documentation

```bash
uv run poe docs        # preview on http://127.0.0.1:8000
uv run poe docs-build  # static site in site/
```

## Building the package

```bash
uv build               # source distribution and wheel in dist/
```
