"""Command line interface for the ML Practice Platform."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUESTIONS_DIR = ROOT / "questions"
SOLUTIONS_DIR = ROOT / "solutions"
TESTS_DIR = ROOT / "tests"
TEMPLATE_SOLUTIONS_DIR = ROOT / "templates" / "solutions"
METADATA_PATH = ROOT / "metadata" / "questions.json"


def load_questions() -> list[dict[str, object]]:
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_question(question_id: str) -> dict[str, object]:
    normalized_id = normalize_question_id(question_id)
    for question in load_questions():
        if question["id"] == normalized_id:
            return question
    raise SystemExit(f"Question {normalized_id} was not found.")


def normalize_question_id(question_id: str) -> str:
    if not question_id.isdigit():
        raise SystemExit("Question id must contain only digits, such as 0001.")
    return question_id.zfill(4)


def display_question(question_id: str) -> None:
    question = find_question(question_id)
    question_path = QUESTIONS_DIR / f"{question['id']}.md"
    if not question_path.exists():
        raise SystemExit(f"Question file is missing: {question_path}")
    print(question_path.read_text(encoding="utf-8"))


def list_questions() -> None:
    for question in load_questions():
        print(f"{question['id']} {question['title']}")


def run_question(question_id: str) -> int:
    question = find_question(question_id)
    test_path = TESTS_DIR / f"test_{question['id']}.py"
    if not test_path.exists():
        raise SystemExit(f"Test file is missing: {test_path}")
    return subprocess.call([sys.executable, "-m", "pytest", str(test_path), "-v"], cwd=ROOT)


def run_all_tests() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-v"], cwd=ROOT)


def show_random_question() -> None:
    question = random.choice(load_questions())
    display_question(str(question["id"]))


def show_stats() -> None:
    questions = load_questions()
    total = len(questions)
    solved = sum(1 for question in questions if question.get("status") == "solved")
    unsolved = total - solved

    print(f"Total questions: {total}")
    print(f"Solved questions: {solved}")
    print(f"Unsolved questions: {unsolved}")


def reset_solutions() -> None:
    missing_templates = []

    for question in load_questions():
        question_id = str(question["id"])
        template_path = TEMPLATE_SOLUTIONS_DIR / f"q{question_id}.py"
        solution_path = SOLUTIONS_DIR / f"q{question_id}.py"

        if not template_path.exists():
            missing_templates.append(str(template_path.relative_to(ROOT)))
            continue

        shutil.copyfile(template_path, solution_path)
        print(f"Reset {solution_path.relative_to(ROOT)}")

    if missing_templates:
        missing = "\n".join(f"- {path}" for path in missing_templates)
        raise SystemExit(f"Missing reset templates:\n{missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Personal LeetCode-style practice environment for ML concepts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all available questions.")

    question_parser = subparsers.add_parser("question", help="Display a question.")
    question_parser.add_argument("question_id", help="Question id, such as 0001.")

    run_parser = subparsers.add_parser("run", help="Run tests for one question.")
    run_parser.add_argument("question_id", help="Question id, such as 0001.")

    subparsers.add_parser("test-all", help="Run the entire test suite.")
    subparsers.add_parser("random", help="Display a random question.")
    subparsers.add_parser("stats", help="Show solved and unsolved counts.")
    subparsers.add_parser(
        "reset",
        help="Overwrite every solution file with its starter template.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        list_questions()
        return 0
    if args.command == "question":
        display_question(args.question_id)
        return 0
    if args.command == "run":
        return run_question(args.question_id)
    if args.command == "test-all":
        return run_all_tests()
    if args.command == "random":
        show_random_question()
        return 0
    if args.command == "stats":
        show_stats()
        return 0
    if args.command == "reset":
        reset_solutions()
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
