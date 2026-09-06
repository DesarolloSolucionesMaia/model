import os
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_validation import LABEL_NAMES, load_and_validate_data
from train import RANDOM_STATE, TEST_SIZE, calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "spanish-news-classification"
DATASET_REVISION = "9f568e396857395bd803c63452b83316124d31c9"

CONFIGURATIONS = {
    "logistic_regression_baseline": {
        "ngram_range": (1, 1),
        "max_features": 30_000,
        "sublinear_tf": False,
        "c_value": 1.0,
    },
    "logistic_regression_improved": {
        "ngram_range": (1, 2),
        "max_features": 50_000,
        "sublinear_tf": True,
        "c_value": 2.0,
    },
    "logistic_regression_classmates": {
        "ngram_range": (1, 2),
        "max_features": 80_000,
        "sublinear_tf": True,
        "c_value": 4.0,
    },
    "logistic_regression_regularized": {
        "ngram_range": (1, 2),
        "max_features": 60_000,
        "sublinear_tf": True,
        "c_value": 0.5,
    },
    "logistic_regression_trigrams": {
        "ngram_range": (1, 3),
        "max_features": 100_000,
        "sublinear_tf": True,
        "c_value": 3.0,
    },
}


def get_git_commit() -> str:
    """Obtiene el commit Git asociado con la ejecución."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip()


def build_experiment_pipeline(configuration: dict) -> Pipeline:
    """Construye el modelo usando la configuración recibida."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=configuration["ngram_range"],
                    min_df=2,
                    max_df=0.95,
                    max_features=configuration["max_features"],
                    sublinear_tf=configuration["sublinear_tf"],
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=configuration["c_value"],
                    max_iter=1_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def log_evaluation_artifacts(y_true, predictions) -> None:
    """Registra en MLflow el reporte y la matriz de confusión."""

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

    mlflow.log_dict(
        report,
        "evaluation/classification_report.json",
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

    axis.set_title("Matriz de confusión")
    figure.tight_layout()

    mlflow.log_figure(
        figure,
        "evaluation/confusion_matrix.png",
    )

    plt.close(figure)


def run_experiment(
    run_name: str,
    configuration: dict,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict:
    """Entrena una configuración y registra sus resultados en MLflow."""

    model = build_experiment_pipeline(configuration)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tags(
            {
                "model_family": "logistic_regression",
                "problem_type": "multiclass_text_classification",
                "dataset": "mteb/SpanishNewsClassification",
                "dataset_revision": DATASET_REVISION,
                "git_commit": get_git_commit(),
            }
        )

        mlflow.log_params(
            {
                "tfidf_ngram_min": configuration["ngram_range"][0],
                "tfidf_ngram_max": configuration["ngram_range"][1],
                "tfidf_max_features": configuration["max_features"],
                "tfidf_min_df": 2,
                "tfidf_max_df": 0.95,
                "tfidf_sublinear_tf": configuration["sublinear_tf"],
                "logistic_regression_c": configuration["c_value"],
                "logistic_regression_max_iter": 1_000,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "train_records": len(train_data),
                "test_records": len(test_data),
            }
        )

        start_time = time.perf_counter()

        model.fit(
            train_data["text"],
            train_data["label"],
        )

        training_time = time.perf_counter() - start_time

        inference_start = time.perf_counter()

        predictions = model.predict(test_data["text"])
        probabilities = model.predict_proba(test_data["text"])

        inference_time = time.perf_counter() - inference_start

        metrics = calculate_metrics(
            test_data["label"],
            predictions,
            probabilities,
        )

        metrics["training_time_seconds"] = training_time
        metrics["inference_time_seconds"] = inference_time

        mlflow.log_metrics(
            {
                name: float(value)
                for name, value in metrics.items()
            }
        )

        log_evaluation_artifacts(
            test_data["label"],
            predictions,
        )

        mlflow.log_dict(
            {
                str(label): category
                for label, category in LABEL_NAMES.items()
            },
            "model/label_names.json",
        )

        input_example = train_data["text"].iloc[:3].to_numpy(dtype=str)
        model_output = model.predict(input_example)
        signature = infer_signature(input_example, model_output)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        return {
            "run_id": active_run.info.run_id,
            "run_name": run_name,
            **metrics,
        }


def main() -> None:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{LOCAL_MLFLOW_DB.resolve().as_posix()}",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_and_validate_data()

    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    results = []

    for run_name, configuration in CONFIGURATIONS.items():
        print(f"Ejecutando: {run_name}")

        result = run_experiment(
            run_name,
            configuration,
            train_data,
            test_data,
        )

        results.append(result)

        print(
            f"Macro F1: {result['macro_f1']:.4f} | "
            f"Accuracy: {result['accuracy']:.4f}"
        )

    comparison = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    print("\nComparación de ejecuciones:")
    print(
        comparison[
            [
                "run_name",
                "macro_f1",
                "accuracy",
                "log_loss",
                "training_time_seconds",
            ]
        ].to_string(index=False)
    )

    best_run = comparison.iloc[0]

    print(
        f"\nMejor ejecución: {best_run['run_name']} "
        f"con Macro F1={best_run['macro_f1']:.4f}"
    )
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()