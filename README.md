# TTTD

Refactored implementation of [TTT-Discover](https://github.com/test-time-training/discover) for the Erdős minimum overlap problem.

Uses [tinker-cookbook](https://github.com/anthropics/tinker-cookbook) as a library instead of embedding the training infrastructure.

Just much cleaner.

## Paper

https://arxiv.org/abs/2505.14644

## Usage

```bash
# Training
uv run python src/tttd/main.py

# Evaluation
uv run python src/tttd/eval.py model=openai/gpt-oss-20b
```
