# ML Practice Platform

A local LeetCode-style practice environment for machine learning concepts.

This project is not interview prep. It is a personal ML knowledge gym: a place to repeatedly practice small, useful implementations until they become muscle memory.

Each question focuses on one concept and should usually take about 5-20 minutes.

## Tech Stack

The project currently supports:

- Python 3.11+
- NumPy
- pandas
- scikit-learn
- PyTorch
- PyTest
- Python standard library

PyTorch is used for tensor, autograd, and model evaluation practice questions.

## Repository Structure

```text
ml-practice/
  questions/             Markdown problem statements
  solutions/             Starter function files
  templates/
    solutions/           Reset templates for solution files
  helpers/               Reusable code for selected questions
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

Reset every solution file back to its starter state:

```bash
python cli.py reset
```

`reset` overwrites all files in `solutions/` using the matching files in `templates/solutions/`.

## Current Questions

- `0001` ReLU Activation
- `0002` Softmax
- `0003` Cross Entropy Loss
- `0004` Mean Squared Error Loss
- `0005` Linear Regression Model
- `0006` Housing Data Preprocessing
- `0007` Train Linear Regression
- `0008` PyTorch Device Selection
- `0009` PyTorch Tensor Basics
- `0010` PyTorch Autograd Gradient
- `0011` PyTorch Model Accuracy
- `0012` MNIST Tensor DataLoader Prep
- `0013` PyTorch Two-Layer Classifier
- `0014` PyTorch Training Loop

## Adding Future Questions

To add question `0004`:

1. Create a markdown prompt at `questions/0004.md`.
2. Create starter code at `solutions/q0004.py`.
3. Create the matching reset template at `templates/solutions/q0004.py`.
4. Create tests at `tests/test_0004.py`.
5. Add an entry to `metadata/questions.json`.
6. Add helper files under `helpers/` when a question intentionally depends on provided code.

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

The file in `templates/solutions/` should be the bare starter version of the solution. The `python cli.py reset` command copies these templates into `solutions/`, overwriting any current solution work.

## Expected Initial Test Behavior

The starter solution files intentionally raise `NotImplementedError`.

That means tests should fail at first. After you implement the functions correctly, the tests should pass.
