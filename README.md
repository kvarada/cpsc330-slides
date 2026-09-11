# cpsc330-slides
CPSC 330 lecture slides 

This repository contains the slides for CPSC 330, Section 102, for 2026W1. The instructor for this section is Varada Kolhatkar.

## Automatic publishing

The [Publish slides workflow](.github/workflows/publish.yml) renders the Quarto
website and deploys it to <https://kvarada.github.io/cpsc330-slides/> on every
push to `main`. You can also start it from **Actions → Publish slides → Run
workflow**. No manual `quarto publish` command is needed.

### Downloadable slide PDFs

PDFs are opt-in. When a lecture is ready, add this to the YAML frontmatter
of its **lecture page** (for example, `website/lecture-02.qmd`):

```yaml
publish-pdf: true
```

Lecture 1 is enabled initially. Other lectures default to disabled. This flag
shows the download button and enables PDF export during publishing. Remove it
or set it to `false` to remove the button and PDF on the next publication.
Changing only this flag does not require re-executing the slide source.

Actions restores cached PDFs and exports only enabled decks whose rendered
HTML or assets have changed. A homepage-only edit reuses existing PDFs. Each
deck's figure directory is tracked separately; shared assets under `slides/`
and `site_libs/` conservatively invalidate all enabled PDFs. Export script and
workflow changes also invalidate PDFs. Remote asset changes are not detected;
store assets locally for reliable invalidation. GitHub may evict caches, in
which case enabled PDFs are regenerated automatically.

DeckTape and Chrome installation is skipped when all enabled PDFs are cached
or no lectures are enabled. Missing PDFs export three decks at a time, and any
export failure prevents deployment. Generated PDFs are not committed.

After changing slide sources, continue rendering locally and committing the
updated `website/_freeze/` files. PDF export uses rendered HTML, not Python.
To generate PDFs locally after `quarto render website`, install DeckTape 3.16.1
and run `python3 scripts/slide_pdfs.py plan` followed by
`python3 scripts/slide_pdfs.py export`. A plain local preview can show an enabled
button before its PDF has been generated. PDFs are static; interactive content
remains available in the HTML slides.

### One-time GitHub setup

In the repository's [Settings → Pages](https://github.com/kvarada/cpsc330-slides/settings/pages),
set **Build and deployment → Source** to **GitHub Actions**. Commit and push the
workflow, website sources and assets, and the complete `website/_freeze/`
directory. The workflow uses GitHub's built-in token; no personal access token
or publishing secret is needed.

Also open **Settings → Environments → github-pages**. Under **Deployment
branches and tags**, choose **Selected branches and tags** and add a **Branch**
rule for `main`. An older Pages setup may only allow `gh-pages`; this workflow
deploys from `main`. Leave any other required environment protections in place.
If a run was rejected by this branch rule, correct the setting and use
**Re-run failed jobs** on that Actions run.

### Updating slides

Preview or render changed slides locally using the setup below, then commit
the source changes **and their updated `website/_freeze/` files** and push to
`main`. Commit any new images or data needed by the slides too.

Actions uses Quarto 1.10.18 and `freeze: auto` to reuse saved Python results.
It builds the HTML and publishes the site without installing the ML environment
or downloading pretrained models. This follows Quarto's
[local execution with CI rendering approach](https://quarto.org/docs/publishing/ci.html).
Keep `_freeze/` in version control; keep `website/_site/` ignored.

If a slide containing executable code changes (including its prose), render
it locally before pushing. Missing or outdated frozen results make Quarto try
to execute Python in Actions, where the ML dependencies are not installed.
Render the affected slide and commit the updated `_freeze/` directory to fix
that build failure. Changes to data, imported Python helpers, or dependencies
also require explicitly re-rendering the affected slides, since `freeze: auto`
does not detect those changes.

Check **Actions → Publish slides** for build errors or deployment status.
The live site is updated only after a successful build and deployment.

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
