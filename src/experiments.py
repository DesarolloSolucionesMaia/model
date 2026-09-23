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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_validation import LABEL_NAMES, load_and_validate_data
from train import RANDOM_STATE, TEST_SIZE, calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "spanish-news-classification"
DATASET_REVISION = "9f568e396857395bd803c63452b83316124d31c9"

# Número de particiones de la validación cruzada. Una única división 80/20 no
# permite ordenar configuraciones cuyas diferencias son del orden del ruido de
# muestreo (~0,007 en F1 macro, unos 3 documentos sobre 410).
CV_FOLDS = 5

CONFIGURATIONS = {
    "logistic_regression_baseline": {
        "family": "logistic_regression",
        "ngram_range": (1, 1),
        "max_features": 30_000,
        "sublinear_tf": False,
        "c_value": 1.0,
    },
    "logistic_regression_improved": {
        "family": "logistic_regression",
        "ngram_range": (1, 2),
        "max_features": 50_000,
        "sublinear_tf": True,
        "c_value": 2.0,
    },
    # Configuración equivalente a la desplegada en la API (ver src/train.py).
    "logistic_regression_deployed": {
        "family": "logistic_regression",
        "ngram_range": (1, 2),
        "max_features": 80_000,
        "sublinear_tf": True,
        "c_value": 3.0,
    },
    # Familias alternativas: la comparación entre configuraciones de un mismo
    # algoritmo es ajuste de hiperparámetros, no comparación de modelos, y no
    # responde si el techo cercano a 0,93 lo impone el enfoque o el clasificador.
    "complement_naive_bayes": {
        "family": "complement_naive_bayes",
        "ngram_range": (1, 2),
        "max_features": 80_000,
        "sublinear_tf": True,
        "alpha": 0.3,
    },
    "linear_svm_calibrated": {
        "family": "linear_svm",
        "ngram_range": (1, 2),
        "max_features": 80_000,
        "sublinear_tf": True,
        "c_value": 1.0,
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


def build_classifier(configuration: dict):
    """Construye el clasificador correspondiente a la familia configurada.

    Todas las familias exponen `predict_proba`, requisito para calcular la
    pérdida logarítmica y la exactitud top-3, y para que la API pueda mostrar
    la confianza en el tablero. `LinearSVC` no la expone de forma nativa, así
    que se envuelve en una calibración que deriva probabilidades del margen.
    """

    family = configuration["family"]

    if family == "logistic_regression":
        return LogisticRegression(
            C=configuration["c_value"],
            max_iter=1_000,
            random_state=RANDOM_STATE,
        )

    if family == "complement_naive_bayes":
        return ComplementNB(alpha=configuration["alpha"])

    if family == "linear_svm":
        return CalibratedClassifierCV(
            estimator=LinearSVC(
                C=configuration["c_value"],
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            cv=CV_FOLDS,
        )

    raise ValueError(f"Familia de modelo desconocida: {family}")


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
            ("classifier", build_classifier(configuration)),
        ]
    )


def build_logged_params(configuration: dict) -> dict:
    """Arma los parámetros a registrar, propios de cada familia."""

    params = {
        "model_family": configuration["family"],
        "tfidf_ngram_min": configuration["ngram_range"][0],
        "tfidf_ngram_max": configuration["ngram_range"][1],
        "tfidf_max_features": configuration["max_features"],
        "tfidf_min_df": 2,
        "tfidf_max_df": 0.95,
        "tfidf_sublinear_tf": configuration["sublinear_tf"],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
    }

    if configuration["family"] == "logistic_regression":
        params["logistic_regression_c"] = configuration["c_value"]
        params["logistic_regression_max_iter"] = 1_000
    elif configuration["family"] == "complement_naive_bayes":
        params["complement_nb_alpha"] = configuration["alpha"]
    elif configuration["family"] == "linear_svm":
        params["linear_svc_c"] = configuration["c_value"]
        params["calibration_method"] = "sigmoid"

    return params


def cross_validate_model(configuration: dict, data: pd.DataFrame) -> dict:
    """Evalúa la configuración con validación cruzada estratificada.

    Devuelve media y desviación estándar del F1 macro. La desviación es lo que
    permite juzgar si una diferencia entre modelos es real o cabe dentro de la
    variabilidad del muestreo.
    """

    scores = cross_val_score(
        build_experiment_pipeline(configuration),
        data["text"],
        data["label"],
        cv=StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        scoring="f1_macro",
    )

    return {
        "cv_macro_f1_mean": float(np.mean(scores)),
        "cv_macro_f1_std": float(np.std(scores)),
        "cv_fold_scores": [float(score) for score in scores],
    }


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
    full_data: pd.DataFrame,
) -> dict:
    """Entrena una configuración y registra sus resultados en MLflow."""

    model = build_experiment_pipeline(configuration)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tags(
            {
                "model_family": configuration["family"],
                "problem_type": "multiclass_text_classification",
                "dataset": "mteb/SpanishNewsClassification",
                "dataset_revision": DATASET_REVISION,
                "git_commit": get_git_commit(),
            }
        )

        params = build_logged_params(configuration)
        params["train_records"] = len(train_data)
        params["test_records"] = len(test_data)

        mlflow.log_params(params)

        cv_results = cross_validate_model(configuration, full_data)

        mlflow.log_metrics(
            {
                "cv_macro_f1_mean": cv_results["cv_macro_f1_mean"],
                "cv_macro_f1_std": cv_results["cv_macro_f1_std"],
            }
        )

        mlflow.log_dict(
            cv_results,
            "evaluation/cross_validation.json",
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
            "model_family": configuration["family"],
            "cv_macro_f1_mean": cv_results["cv_macro_f1_mean"],
            "cv_macro_f1_std": cv_results["cv_macro_f1_std"],
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
            data,
        )

        results.append(result)

        print(
            f"CV Macro F1: {result['cv_macro_f1_mean']:.4f} "
            f"+/- {result['cv_macro_f1_std']:.4f} | "
            f"Macro F1: {result['macro_f1']:.4f} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Log loss: {result['log_loss']:.4f}"
        )

    comparison = pd.DataFrame(results).sort_values(
        by="cv_macro_f1_mean",
        ascending=False,
    )

    print("\nComparación de ejecuciones:")
    print(
        comparison[
            [
                "run_name",
                "model_family",
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
                "macro_f1",
                "accuracy",
                "log_loss",
            ]
        ].to_string(index=False)
    )

    best_run = comparison.iloc[0]

    print(
        f"\nMejor ejecución por F1 macro en validación cruzada: "
        f"{best_run['run_name']} con "
        f"{best_run['cv_macro_f1_mean']:.4f} "
        f"+/- {best_run['cv_macro_f1_std']:.4f}"
    )

    # Una diferencia menor que la desviación estándar de la validación cruzada
    # no permite declarar un ganador: es ruido de muestreo, no mejora.
    margin = best_run["cv_macro_f1_mean"] - comparison.iloc[1]["cv_macro_f1_mean"]

    if margin < best_run["cv_macro_f1_std"]:
        print(
            f"Aviso: la ventaja sobre la segunda ({margin:.4f}) es menor que "
            f"la desviación estándar ({best_run['cv_macro_f1_std']:.4f}); "
            f"las dos son estadísticamente indistinguibles."
        )

    best_by_log_loss = comparison.sort_values(by="log_loss").iloc[0]

    print(
        f"Mejor calibración (menor pérdida logarítmica): "
        f"{best_by_log_loss['run_name']} con "
        f"{best_by_log_loss['log_loss']:.4f}"
    )
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()