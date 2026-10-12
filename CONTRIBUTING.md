# Contributing to Agima

Report bugs or propose research changes in an [issue](https://github.com/Agentic-Imagination/Agima/issues). Describe the affected task and workflow so others can reproduce the problem or discuss the change.

## Development checks

Run from the repository root with Python 3.10 or later:

```bash
python3 tools/check_repo.py
python3 -m unittest discover -s tests -v
git diff --check
```

These checks use the standard library and do not require model weights, datasets, PyTorch, or CUDA. They cover source syntax, local imports and assets, documentation links, and launcher behavior. See [installation](docs/installation.md) for model dependencies.

For the critical lint checks run in CI, install `requirements/dev.txt` and run
`python -m ruff check --select E9,F821,F822,F823 .`. Separate simulator smoke checks
exercise reset, rendering, actions and state replay; see [validation](docs/validation.md)
for their CPU dependencies and strict mode. Dependency-free tests skip these simulators
when their packages are absent.

Keep changes focused. Update the relevant guide when changing a command, dependency, path, or supported workflow. Include a regression test for a behavioral bug; do not import the full model stack in CPU-only launcher tests.

## Preserve experiment contracts

Changes to prompts, image encoding, observation order, action parsing, stop conditions, loss masks, or environment dynamics can change results. Treat them as research changes: describe the before/after behavior and provide task-specific evidence. Keep maintenance changes separate so their effect can be reviewed.

Prompt builders under `scripts/data/` are shared with evaluation. Update their data and evaluation consumers together. Public examples must use explicit resource arguments or documented configuration; keep downloaded assets, generated outputs, credentials, and personal machine paths out of commits.

Preserve copyright and license notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before adding or redistributing third-party code, data, or assets.

## Report a reproducible problem

Include the following information in an issue or pull request:

- Code commit, task, variant, chunk size, and decoder.
- The exact command and complete traceback, with private paths or credentials removed.
- Python, PyTorch, CUDA, FlashAttention, GPU model, and visible GPU count when relevant.
- Model and dataset identifiers/revisions, and whether the input is a held-out file or directory.
- Expected behavior, observed behavior, and the smallest run that reproduces it.
- Checks performed and any GPU validation that remains outstanding.

A launcher dry run checks command construction. For model or environment changes, include an end-to-end run with episode-level results. Keep the episode pool, initial states, preprocessing, decoder, and action budget fixed when comparing results; use the complete evaluation pool for benchmark claims.
