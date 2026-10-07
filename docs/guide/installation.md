# Installation

TorchMS needs Python 3.10 or later and uses [uv](https://docs.astral.sh/uv/) to manage its
environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv, once

git clone https://gitlab.univ-artois.fr/astrid.klipfel/pycrystalgen.git
cd pycrystalgen
uv sync
```

Always run Python through `uv run`, which uses the environment of the project:

```bash
uv run python my_script.py
uv run pytest
```

## Optional parts

| Need | Command | What it adds |
|---|---|---|
| Loading datasets and the estimator from Hugging Face | `uv sync --extra hub` | `huggingface_hub` |
| Building this documentation | `uv sync --group docs` | MkDocs Material, mkdocstrings |

The development group (tests, type checker, ASE, pymatgen, pyarrow) is installed by default by
`uv sync`.

## GPU

PyTorch is installed with CUDA support. The energy estimator runs on CPU but is much faster on
a GPU: about 600 structures per second on a laptop GPU with 8 GB of memory.

## Checks

```bash
uv run pytest          # tests, including the examples of the docstrings
uv run pyright src     # type checking
uv run black .         # formatting
```

## Building the documentation

```bash
uv run --group docs mkdocs serve          # preview on http://127.0.0.1:8000
uv run --group docs mkdocs build --strict # static site in site/
```
