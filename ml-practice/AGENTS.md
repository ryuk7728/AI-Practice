# AGENTS.md

Guidance for Codex or any coding agent working on this project.

## Project Purpose

This repository is a local LeetCode-style practice platform for machine learning concepts.

It is not an interview prep platform. It is a personal ML knowledge gym for long-term retention of useful ML techniques, implementations, patterns, and engineering tricks encountered during real projects.

Each exercise should focus on one concept and be solvable in roughly 5-20 minutes.

Good question topics include:

- ReLU
- Softmax
- Cross entropy
- IoU
- Train/test split
- Gradient descent
- Batch normalization
- PyTorch Dataset concepts, if implemented without adding PyTorch
- Confusion matrix
- Non-max suppression

## Tech Stack

Use:

- Python 3.11+
- NumPy
- PyTest
- Python standard library

Do not add unnecessary dependencies. Avoid adding ML frameworks unless the user explicitly asks and approves a broader project direction.

## Repository Structure

```text
ml-practice/
  questions/             Markdown problem statements
  solutions/             Starter function files
  tests/                 PyTest tests for each question
  data/                  Optional data files for future questions
  metadata/
    questions.json       Question index and status metadata
  cli.py                 Command line interface
  requirements.txt       Python dependencies
  pytest.ini             PyTest configuration
  README.md              User-facing project guide
  AGENTS.md              Agent-facing project guide
```

## Core Question Style

Questions should be clean practice prompts, not tutorials.

Every question markdown file should include:

- Title with numeric id
- Problem
- Function signature
- Behavioral expectations
- Learning objectives
- Examples, when useful
- Complexity discussion
- Files section

Do not include:

- Hints
- Step-by-step solution guidance
- Required implementation recipes
- TODO lists
- Source-code snippets that reveal the intended answer
- Language like "use `np.max`", "use `np.clip`", "subtract the max", or "do this with vectorized NumPy"

It is okay to specify required behavior. For example, say "the result should remain finite when probabilities contain `0`." Do not say exactly how to achieve it.

The user wants the learner to figure out the implementation based on the question and the tests.

## Solution File Style

Solution files should be starter stubs only.

Use this shape:

```python
"""Starter code for Question 0004: Question Title."""


def function_name(...):
    raise NotImplementedError("Implement function_name in solutions/q0004.py")
```

Do not include:

- TODO markers
- Docstrings explaining how to solve the problem
- Imports unless the starter signature genuinely requires them
- Completed solutions
- Partial solutions

The default state of the project should be that tests fail because the solution functions are intentionally unimplemented.

## Test Style

Tests should verify behavior and understanding.

Each test file should:

- Import the matching function from `solutions/qXXXX.py`
- Use multiple inputs
- Include ordinary cases
- Include edge cases
- Include randomized cases where useful
- Use clear test names
- Stay easy to extend

Avoid tests that inspect the user's source code. Prefer behavioral tests over checking that a specific function or line of code was used.

It is acceptable for tests to contain expected values or internal helper logic. Keep helper logic readable and do not make tests depend on fragile formatting or implementation details.

The tests should fail initially because the starter solution raises `NotImplementedError`. If a correct implementation is temporarily added, all tests for that question should pass.

## Adding A New Question

When the user asks to add a question, create or update all of these:

1. `questions/XXXX.md`
2. `solutions/qXXXX.py`
3. `tests/test_XXXX.py`
4. `metadata/questions.json`
5. `README.md`, if the current question list or docs need updating

Use the next four-digit id unless the user specifies one.

Example metadata entry:

```json
{
  "id": "0004",
  "title": "Question Title",
  "tags": ["numpy"],
  "difficulty": "easy",
  "status": "unsolved"
}
```

Keep metadata easy to extend to hundreds of questions.

## Verification Workflow

After adding or editing questions:

1. Run CLI checks:

```bash
python cli.py list
python cli.py question XXXX
python cli.py run XXXX
```

2. Confirm the new question tests are discovered.

3. Confirm tests fail in starter state due to `NotImplementedError`.

4. Temporarily implement a correct solution to verify the tests are fair.

5. Run:

```bash
python cli.py run XXXX
python cli.py test-all
```

6. Restore the starter stub before finishing.

7. Confirm the solution file is back to `NotImplementedError`.

Do not leave completed answers in `solutions/` unless the user explicitly asks for solved solutions.

## CLI Expectations

The CLI lives in `cli.py` and should continue to support:

```bash
python cli.py list
python cli.py question 0001
python cli.py run 0001
python cli.py test-all
python cli.py random
python cli.py stats
```

Do not break these commands when extending the project.

## User Preference Summary

The user specifically wants:

- No hints in question markdown
- No suggestions about how the answer must be implemented
- No TODOs in solution files
- Questions that force the learner to infer the implementation
- Tests that validate correctness without prescribing source code
- A project that is easy to extend with future ML practice questions

When in doubt, make the problem statement clearer about expected behavior, not clearer about implementation strategy.
