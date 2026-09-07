# cpsc330-slides
CPSC 330 lecture slides 

This repository contains the slides for CPSC 330, Section 102, for 2024W1. The instructor for this section is Varada Kolhatkar.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
[Quarto](https://quarto.org/docs/get-started/), and the
[Graphviz executable](https://graphviz.org/download/). On macOS, Graphviz can
be installed with `brew install graphviz`; check it with `dot -V`.
The Python `graphviz` package does not include this executable.

From the repository root, deactivate any active Conda environment and run:

```bash
uv sync --locked
uv run python -m ipykernel install --sys-prefix --name python3 --display-name "CPSC 330 slides (.venv)"
```

uv installs Python 3.12 if needed and creates `.venv` using the versions in
`uv.lock`. Conda is no longer needed for this repository.

### Preview and render slides

Run these commands from the repository root on macOS/Linux:

```bash
export QUARTO_PYTHON="$PWD/.venv/bin/python"
export JUPYTER_PREFER_ENV_PATH=1
uv run quarto preview website
# Or render without starting a preview server:
uv run quarto render website
```

In PowerShell, set `$env:QUARTO_PYTHON = "$PWD/.venv/Scripts/python.exe"`
and `$env:JUPYTER_PREFER_ENV_PATH = "1"`
before running the same `uv run quarto` commands.

The kernel is registered inside `.venv`, and the environment preference prevents
an old user-level Conda `python3` kernel from taking precedence. Repeat the
kernel installation if you recreate `.venv`.

The website uses `freeze: auto`. To check Python execution after changing
dependencies, render an individual slide file explicitly, for example:

```bash
uv run quarto render website/slides/slides-02-terminology-decision-trees.qmd --execute
```

### Notebooks

Use the same `JUPYTER_PREFER_ENV_PATH` setting as above, then launch:

```bash
uv run jupyter lab
```

In VS Code, select `.venv/bin/python` (Windows: `.venv/Scripts/python.exe`)
as the notebook kernel. NLP datasets and pretrained model downloads used by
individual lectures may require additional network access and disk space.

### Manage dependencies

Use `uv add PACKAGE` and `uv remove PACKAGE`, and commit both `pyproject.toml`
and `uv.lock`. Use `uv sync --locked` after pulling changes. To deliberately
upgrade the locked versions, run `uv lock --upgrade` followed by `uv sync`
and check the affected lectures.

PyTorch uses the default PyPI distributions. For a specific CUDA version or
CPU-only Linux installation, follow the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
