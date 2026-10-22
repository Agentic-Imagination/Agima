# Contributing to Agima

Open an [issue](https://github.com/Agentic-Imagination/Agima/issues) or pull request with the task, exact command, expected behavior and observed result. For runtime bugs, include the code commit, model/data revisions, Python/PyTorch/CUDA versions, GPU and traceback. Remove credentials and private paths.

Run these checks from the repository root with Python 3.10+:

```bash
python3 tools/check_repo.py
python3 -m unittest discover -s tests -v
git diff --check
```

These checks run without model weights or CUDA; tests that need optional packages skip when they are absent. CI also runs:

```bash
python -m pip install -r requirements/dev.txt
python -m ruff check --select E9,F821,F822,F823 .
```

Keep changes focused and update affected commands in docs/usage.md. Add a regression test for behavioral fixes. For model or simulator changes, report an end-to-end check and any remaining GPU validation.

Changes to prompts, preprocessing, action parsing, loss masks or environment dynamics can change experimental results. Describe those changes and compare using the same episode pool, initial states, decoder and action budget. Update shared data/evaluation consumers together.

Keep downloaded assets, generated outputs and credentials out of commits. Preserve copyright and [third-party license notices](THIRD_PARTY_NOTICES.md).
