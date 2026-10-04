from support_agent.classifier import KeywordClassifier
from support_agent.config import PROJECT_ROOT
from support_agent.evaluation import evaluate, format_report, load_queries, main


def test_eval_metrics_are_consistent():
    rows = load_queries(PROJECT_ROOT / "eval" / "queries.jsonl")
    report = evaluate(KeywordClassifier(), rows)
    assert report.total == 40
    assert sum(sum(r.values()) for r in report.confusion.values()) == 40
    assert report.accuracy >= 0.90  # quality gate for the offline baseline
    assert "Confusion matrix" in format_report(report)


def test_eval_cli_fails_below_threshold(capsys):
    assert main(["--min-accuracy", "1.01"]) == 1
    assert main(["--min-accuracy", "0.5"]) == 0
