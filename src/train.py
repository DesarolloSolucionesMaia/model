import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_validation import DATASET_PATH, LABEL_NAMES, load_and_validate_data


RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = ARTIFACTS_DIR / "spanish_news_classifier.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"


def build_pipeline() -> Pipeline:
    """Construye el pipeline completo de vectorización y clasificación."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    max_iter=1_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    predictions,
    probabilities,
) -> dict:
    """Calcula las métricas principales del clasificador."""

    labels = sorted(LABEL_NAMES)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_precision": precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=labels,
        ),
        "top_3_accuracy": top_k_accuracy_score(
            y_true,
            probabilities,
            k=3,
            labels=labels,
        ),
    }


def save_evaluation_reports(
    y_true: pd.Series,
    predictions,
    metrics: dict,
) -> None:
    """Guarda métricas, reporte por clase y matriz de confusión."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        name: round(float(value), 6)
        for name, value in metrics.items()
    }

    METRICS_PATH.write_text(
        json.dumps(
            serializable_metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    labels = sorted(LABEL_NAMES)
    target_names = [LABEL_NAMES[label] for label in labels]

    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
    CLASSIFICATION_REPORT_PATH,
    encoding="utf-8",
    index_label="category",
    )

    figure, axis = plt.subplots(figsize=(12, 10))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=labels,
        display_labels=target_names,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        ax=axis,
    )

    axis.set_title("Matriz de confusión - Regresión logística")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close(figure)


def save_model(model: Pipeline, metrics: dict) -> None:
    """Empaqueta el pipeline, la homologación y sus metadatos."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": model,
        "label_names": LABEL_NAMES,
        "metadata": {
            "model_type": "TF-IDF + Logistic Regression",
            "dataset_path": str(DATASET_PATH),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "scikit_learn_version": sklearn.__version__,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                name: float(value)
                for name, value in metrics.items()
            },
        },
    }

    joblib.dump(model_bundle, MODEL_PATH)


def main() -> None:
    data = load_and_validate_data()

    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    model = build_pipeline()

    print(f"Registros de entrenamiento: {len(train_data)}")
    print(f"Registros de prueba: {len(test_data)}")
    print("Entrenando el modelo...")

    model.fit(train_data["text"], train_data["label"])

    predictions = model.predict(test_data["text"])
    probabilities = model.predict_proba(test_data["text"])

    metrics = calculate_metrics(
        test_data["label"],
        predictions,
        probabilities,
    )

    save_evaluation_reports(
        test_data["label"],
        predictions,
        metrics,
    )
    save_model(model, metrics)

    print("Entrenamiento finalizado.")
    print("Métricas:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Reporte guardado en: {REPORTS_DIR}")


if __name__ == "__main__":
    main()