# Contributing

Thanks for wanting to improve this project. Small, focused changes are the easiest to review.

## How to contribute

1. Fork the repository and create a branch for your change.
2. Set up the project as described in [docs/SETUP.md](docs/SETUP.md) and confirm the app still runs.
3. Make your change. Keep it focused: one fix or one feature per pull request.
4. If you change generation behaviour, note the prompt, seed, steps, and size you tested with.
5. Open a pull request and describe what changed, why, and how you tested it.

## Code style

* Python: keep functions short, use clear names, and add a comment only where the reason is not obvious from the code.
* Do not hardcode paths, ports, or credentials. Use environment variables, with a sensible default.
* Do not commit generated images, model weights, or `.env` files. They are gitignored for a reason.
* Keep the web UI in a single self contained HTML file per mode, with no external CDN dependencies, so offline mode stays offline.

## Prompts and documentation

New example prompts are welcome in [examples/PROMPTS.md](examples/PROMPTS.md). Please test a prompt before adding it, and note the seed and settings if a particular combination matters.

Documentation fixes are welcome too. If a setup step did not work for you, say which OS, Python version, and GPU you used, so the fix can be specific instead of a guess.

## Reporting issues

Open an issue with:

* What you tried to do, and what happened instead
* The exact error message from the terminal, if there is one
* Your OS, Python version, PyTorch version, and GPU (or that you are on CPU)
* Which mode you were running: online (`s4`) or offline (`s5 (offline)`)
