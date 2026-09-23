"""Compara familias de modelos distintas con validación cruzada y las registra en MLflow.

Responde a tres observaciones del evaluador sobre la segunda entrega:

1. Las cinco configuraciones anteriores eran la misma familia (TF-IDF +
   regresión logística variando hiperparámetros). Aquí se comparan familias
   realmente distintas: regresión logística, Naive Bayes complementario y
   SVM lineal.
2. La selección se hacía sobre una única partición 80/20, donde la diferencia
   entre las dos mejores configuraciones equivalía a unos tres registros.
   Aquí se usa validación cruzada estratificada de cinco pliegues y se reporta
   la desviación estándar, que es lo que permite afirmar si una diferencia es
   real o es ruido de la partición.
3. Las matrices de confusión se registraban como artefacto pero no se
   analizaban. Aquí se extraen y guardan los pares de categorías que más se
   confunden entre sí.
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_validation import LABEL_NAMES, load_and_validate_data


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
EXPERIMENT_NAME = "spanish-news-model-families"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "model_comparison"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "model_comparison"

# El vectorizador se mantiene fijo en la mejor configuración conocida de la
# segunda entrega, para que la comparación aísle el efecto del clasificador.
TFIDF_SETTINGS = dict(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50_000,
    sublinear_tf=True,
)


def build_classifier(family: str):
    """Devuelve el clasificador correspondiente a cada familia comparada."""

    if family == "logistic_regression":
        return LogisticRegression(C=2.0, max_iter=1_000, random_state=RANDOM_STATE)

    if family == "complement_naive_bayes":
        return ComplementNB(alpha=0.3)

    if family == "linear_svm":
        # LinearSVC no expone predict_proba y el tablero necesita un valor de
        # confianza, así que se calibra. La calibración usa validación cruzada
        # interna sobre el conjunto de entrenamiento, no toca el de prueba.
        return CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=RANDOM_STATE),
            cv=3,
        )

    raise ValueError(f"Familia de modelo desconocida: {family}")


def build_pipeline(family: str) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
            ("classifier", build_classifier(family)),
        ]
    )


def top_confusions(matrix: np.ndarray, labels: list[int], top_n: int = 10) -> list[dict]:
    """Extrae los pares de categorías que más se confunden entre sí."""

    pairs = []

    for row, true_label in enumerate(labels):
        for column, predicted_label in enumerate(labels):
            if row == column or matrix[row][column] == 0:
                continue

            pairs.append(
                {
                    "categoria_real": LABEL_NAMES[true_label],
                    "categoria_predicha": LABEL_NAMES[predicted_label],
                    "casos": int(matrix[row][column]),
                }
            )

    pairs.sort(key=lambda pair: pair["casos"], reverse=True)

    return pairs[:top_n]


def save_confusion_artifacts(y_true, predictions, labels: list[int], family: str) -> list[Path]:
    """Guarda la matriz de confusión y, sobre todo, su lectura."""

    run_dir = REPORTS_DIR / family
    run_dir.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, predictions, labels=labels)
    target_names = [LABEL_NAMES[label] for label in labels]

    figure, axis = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=target_names,
    ).plot(cmap="Blues", xticks_rotation=45, values_format="d", ax=axis)
    axis.set_title(f"Matriz de confusión - {family}")
    figure.tight_layout()

    matrix_path = run_dir / "confusion_matrix.png"
    figure.savefig(matrix_path, dpi=160)
    plt.close(figure)

    confusions = top_confusions(matrix, labels)
    confusions_path = run_dir / "top_confusions.csv"
    pd.DataFrame(confusions).to_csv(confusions_path, index=False, encoding="utf-8")

    return [matrix_path, confusions_path]


def evaluate_family(
    family: str,
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    labels: list[int],
) -> dict:
    """Entrena, evalúa por validación cruzada y registra una familia en MLflow."""

    pipeline = build_pipeline(family)

    with mlflow.start_run(run_name=family):
        mlflow.log_params(
            {
                "model_family": family,
                "classifier": type(build_classifier(family)).__name__,
                "tfidf_ngram_range": str(TFIDF_SETTINGS["ngram_range"]),
                "tfidf_max_features": TFIDF_SETTINGS["max_features"],
                "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
            }
        )

        # Validación cruzada sobre el dataset completo: es lo que da la
        # desviación estándar y permite saber si una diferencia es real.
        splitter = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        cv_scores = cross_val_score(
            pipeline,
            data["text"],
            data["label"],
            cv=splitter,
            scoring="f1_macro",
        )

        # Además, la partición única 80/20 para poder comparar con la entrega 2.
        pipeline.fit(train_data["text"], train_data["label"])
        predictions = pipeline.predict(test_data["text"])
        probabilities = pipeline.predict_proba(test_data["text"])

        metrics = {
            "cv_macro_f1_mean": float(cv_scores.mean()),
            "cv_macro_f1_std": float(cv_scores.std()),
            "holdout_accuracy": accuracy_score(test_data["label"], predictions),
            "holdout_macro_f1": f1_score(
                test_data["label"],
                predictions,
                average="macro",
                zero_division=0,
            ),
            "holdout_log_loss": log_loss(test_data["label"], probabilities, labels=labels),
            "holdout_top_3_accuracy": top_k_accuracy_score(
                test_data["label"],
                probabilities,
                k=3,
                labels=labels,
            ),
        }

        mlflow.log_metrics(metrics)

        for fold, score in enumerate(cv_scores, start=1):
            mlflow.log_metric("cv_fold_macro_f1", float(score), step=fold)

        run_dir = REPORTS_DIR / family
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    **{name: round(value, 6) for name, value in metrics.items()},
                    "cv_fold_scores": [round(float(score), 6) for score in cv_scores],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        model_dir = ARTIFACTS_DIR / family
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        joblib.dump(pipeline, model_path)

        artifacts = [
            metrics_path,
            model_path,
            *save_confusion_artifacts(test_data["label"], predictions, labels, family),
        ]

        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))

        print(
            f"{family}: "
            f"CV macro F1 = {metrics['cv_macro_f1_mean']:.4f} "
            f"(+/- {metrics['cv_macro_f1_std']:.4f}), "
            f"holdout macro F1 = {metrics['holdout_macro_f1']:.4f}"
        )

        return {"family": family, **metrics}


def main() -> None:
    data = load_and_validate_data()
    labels = sorted(LABEL_NAMES)

    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = [
        evaluate_family(family, data, train_data, test_data, labels)
        for family in ("logistic_regression", "complement_naive_bayes", "linear_svm")
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results).sort_values("cv_macro_f1_mean", ascending=False)
    summary_path = REPORTS_DIR / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"\nResumen guardado en: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
