"""Offline evaluation of query classification.

Runs every labelled query in eval/queries.jsonl through the classifier and
reports accuracy, a confusion matrix, and per-class precision, recall and F1:

    Precision = TP / (TP + FP)     Recall = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)

Usage:  python -m support_agent.evaluation [--min-accuracy 0.85] [--out report.json]
CI runs this and fails the build if accuracy drops below the threshold, so a
prompt or rule change that makes the agent worse cannot be merged silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from support_agent.classifier import Category, Classifier, KeywordClassifier, LLMClassifier
from support_agent.config import PROJECT_ROOT, get_settings
from support_agent.llm import build_chat_model
from support_agent.pii import redact

LABELS = [c.value for c in Category]


@dataclass
class EvalReport:
    total: int
    correct: int
    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]
    errors: list[dict[str, str]]


def load_queries(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(classifier: Classifier, rows: list[dict[str, str]]) -> EvalReport:
    confusion = {a: {p: 0 for p in LABELS} for a in LABELS}
    errors = []
    for row in rows:
        predicted = classifier.classify(redact(row["query"]).text).value
        confusion[row["label"]][predicted] += 1
        if predicted != row["label"]:
            errors.append({"query": row["query"], "expected": row["label"], "got": predicted})

    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[a][label] for a in LABELS if a != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": tp + fn,
        }

    correct = sum(confusion[label][label] for label in LABELS)
    return EvalReport(
        len(rows), correct, round(correct / len(rows), 3), per_class, confusion, errors
    )


def format_report(r: EvalReport) -> str:
    lines = [
        f"Accuracy: {r.correct}/{r.total} = {r.accuracy:.1%}",
        "",
        f"{'class':<10} {'precision':>9} {'recall':>7} {'f1':>6} {'n':>4}",
    ]
    for label, m in r.per_class.items():
        lines.append(
            f"{label:<10} {m['precision']:>9.2f} {m['recall']:>7.2f} "
            f"{m['f1']:>6.2f} {m['support']:>4}"
        )
    lines += [
        "",
        "Confusion matrix (rows = actual, columns = predicted)",
        " " * 11 + " ".join(f"{p[:5]:>6}" for p in LABELS),
    ]
    for a in LABELS:
        lines.append(f"{a:<10} " + " ".join(f"{r.confusion[a][p]:>6}" for p in LABELS))
    if r.errors:
        lines += ["", "Misclassified:"]
        lines += [f"  - {e['query']!r}: expected {e['expected']}, got {e['got']}" for e in r.errors]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "eval" / "queries.jsonl")
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    llm = build_chat_model(get_settings())
    classifier = LLMClassifier(llm) if llm else KeywordClassifier()
    report = evaluate(classifier, load_queries(args.data))
    print(f"Classifier: {type(classifier).__name__}\n")
    print(format_report(report))
    if args.out:
        args.out.write_text(json.dumps(report.__dict__, indent=2))
    if report.accuracy < args.min_accuracy:
        print(f"\nFAIL: accuracy {report.accuracy:.1%} < {args.min_accuracy:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
