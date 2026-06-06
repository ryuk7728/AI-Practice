# ML Practice Platform

A local LeetCode-style practice environment for machine learning concepts.

This project is not interview prep. It is a personal ML knowledge gym: a place to repeatedly practice small, useful implementations until they become muscle memory.

Each question focuses on one concept and should usually take about 5-20 minutes.

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
```

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## How To Solve A Question

1. Pick a question:

```bash
python cli.py list
```

2. Read the prompt:

```bash
python cli.py question 0001
```

3. Open the matching solution file:

```text
solutions/q0001.py
```

4. Implement the function.

5. Run the tests:

```bash
python cli.py run 0001
```

## CLI Commands

List all questions:

```bash
python cli.py list
```

Display one question:

```bash
python cli.py question 0001
```

Run one question:

```bash
python cli.py run 0001
```

Run every test:

```bash
python cli.py test-all
```

Display a random question:

```bash
python cli.py random
```

Show solved and unsolved counts:

```bash
python cli.py stats
```

## Current Questions

- `0001` ReLU Activation
- `0002` Softmax
- `0003` Cross Entropy Loss

## Adding Future Questions

To add question `0004`:

1. Create a markdown prompt at `questions/0004.md`.
2. Create starter code at `solutions/q0004.py`.
3. Create tests at `tests/test_0004.py`.
4. Add an entry to `metadata/questions.json`.

Use this metadata shape:

```json
{
  "id": "0004",
  "title": "New Question Title",
  "tags": ["numpy"],
  "difficulty": "easy",
  "status": "unsolved"
}
```

Keep each question independent, focused, and small enough to solve in one short practice session.

## Expected Initial Test Behavior

The starter solution files intentionally raise `NotImplementedError`.

That means tests should fail at first. After you implement the functions correctly, the tests should pass.
