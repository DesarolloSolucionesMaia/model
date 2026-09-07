import numpy as np
import pandas as pd
import pytest

from data_validation import LABEL_NAMES
from train import build_pipeline, calculate_metrics, save_evaluation_reports


def test_build_pipeline_has_expected_configuration():
    pipeline = build_pipeline()

    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)
    assert pipeline.named_steps["tfidf"].max_features == 50_000
    assert pipeline.named_steps["classifier"].C == 2.0
    assert pipeline.named_steps["classifier"].random_state == 42


def test_calculate_metrics_returns_expected_perfect_scores():
    labels = sorted(LABEL_NAMES)
    y_true = pd.Series(labels)
    probabilities = np.eye(len(labels))

    metrics = calculate_metrics(y_true, labels, probabilities)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["top_3_accuracy"] == 1.0
    assert metrics["log_loss"] == pytest.approx(0.0)


def test_save_evaluation_reports_creates_all_report_files(tmp_path, monkeypatch):
    import train

    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(train, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(train, "CLASSIFICATION_REPORT_PATH", tmp_path / "classification_report.csv")
    monkeypatch.setattr(train, "CONFUSION_MATRIX_PATH", tmp_path / "confusion_matrix.png")
    labels = sorted(LABEL_NAMES)
    y_true = pd.Series(labels)
    metrics = calculate_metrics(y_true, labels, np.eye(len(labels)))

    save_evaluation_reports(y_true, labels, metrics)

    assert train.METRICS_PATH.exists()
    assert train.CLASSIFICATION_REPORT_PATH.exists()
    assert train.CONFUSION_MATRIX_PATH.exists()
