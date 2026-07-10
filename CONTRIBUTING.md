# Contributing

Thanks for your interest in improving this customer segmentation project.

## What this repo is

Tutorial-style **unsupervised** clustering notebooks on three UCI datasets (RFM + wholesale spend). PyCaret is used only for algorithm survey; sklearn re-implementation is the real deliverable. There are **no ground-truth segment labels** and no classification accuracy metrics.

## Ways to contribute

- Bug fixes in notebooks, env setup, or docs
- Clearer explanations for learners (markdown in notebooks or README)
- Reproducibility improvements (`uv` pins, kernel notes, data loaders)
- Small, focused feature work (e.g. extra plots, export of segment tables)

Please open an **issue** before large refactors or new datasets so scope stays clear.

## Development setup

```bash
git clone https://github.com/pypi-ahmad/customer-segmentation.git
cd customer-segmentation
uv sync
uv run python -m ipykernel install --user \
  --name customer-segmentation-project \
  --display-name "Python (customer-segmentation-project)"
```

- Python: **3.13.13** (see `.python-version`)
- Kernel name: `customer-segmentation-project`
- Prefer `uv add` for dependencies — not bare `pip install`

### Data notes

| Dataset | How notebooks load it |
|---------|------------------------|
| UCI 292, 352 | `ucimlrepo` (network required) |
| UCI 502 (Online Retail II) | Official UCI zip under `data/` if `ucimlrepo` import fails |

Large `data/*.zip` / `data/*.xlsx` files are gitignored. Download instructions are in the README.

## Workflow

1. Fork the repo (or branch from `main` if you have write access).
2. Create a branch: `git checkout -b fix/short-description`.
3. Make a focused change. Prefer editing jupytext `.py` sources, then reconvert/execute notebooks when behavior or outputs change.
4. Re-run the affected notebook(s) when you change analysis code:

```bash
uv run jupytext --to ipynb notebooks/0X_name.py
uv run jupyter nbconvert --to notebook --execute notebooks/0X_name.ipynb \
  --output 0X_name.ipynb \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=customer-segmentation-project
```

5. Do **not** invent supervised metrics (accuracy, F1) for clustering quality.
6. Open a pull request against `main` with a short summary of *what* and *why*.

## Code style

- Python with type hints where public helpers are added
- Keep notebooks self-contained and tutorial-readable
- Seeds: keep `SEED = 42` unless you document a reason to change it
- No secrets, tokens, or large raw datasets in commits

## Reporting issues

Use the issue templates under **New issue**:

- **Bug report** — broken runs, import errors, wrong columns
- **Feature request** — extensions, docs, new analyses

Include OS, Python version (`uv run python -V`), and the exact command/error when possible.

## Code of conduct

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
