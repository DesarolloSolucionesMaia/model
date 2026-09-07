"""Genera evidencia reproducible de exploración del dataset de noticias."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from data_validation import LABEL_NAMES, load_and_validate_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
PROFILE_PATH = REPORTS_DIR / "data_profile_by_class.csv"
EXAMPLES_PATH = REPORTS_DIR / "data_examples.csv"
CLASS_DISTRIBUTION_PATH = REPORTS_DIR / "class_distribution.png"
TEXT_LENGTHS_PATH = REPORTS_DIR / "text_length_by_class.png"


def build_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Resume tamaño y longitud de texto para cada una de las 12 categorías."""

    profiled_data = data.assign(text_length_chars=data["text"].str.len())
    profile = (
        profiled_data.groupby("label")["text_length_chars"]
        .agg(
            records="size",
            mean_chars="mean",
            median_chars="median",
            min_chars="min",
            max_chars="max",
        )
        .sort_index()
    )
    profile.insert(0, "category", profile.index.map(LABEL_NAMES))
    profile.insert(2, "percentage", profile["records"] / len(data) * 100)
    return profile.round({"percentage": 2, "mean_chars": 2, "median_chars": 2})


def save_examples(data: pd.DataFrame) -> None:
    """Guarda una noticia representativa por categoría, sin modificar el origen."""

    examples = (
        data.groupby("label", group_keys=False)
        .sample(n=1, random_state=42)
        .sort_values("label")
        .copy()
    )
    examples.insert(1, "category", examples["label"].map(LABEL_NAMES))
    examples["text"] = examples["text"].str.slice(0, 500)
    examples.rename(columns={"text": "text_excerpt"}).to_csv(
        EXAMPLES_PATH,
        index=False,
        encoding="utf-8",
    )


def save_charts(data: pd.DataFrame, profile: pd.DataFrame) -> None:
    """Produce gráficos para la evidencia de distribución y dispersión por clase."""

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(profile["category"], profile["records"], color="#2563eb")
    axis.set(title="Distribución de noticias por categoría", ylabel="Registros")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(CLASS_DISTRIBUTION_PATH, dpi=160)
    plt.close(figure)

    chart_data = data.assign(
        category=data["label"].map(LABEL_NAMES),
        text_length_chars=data["text"].str.len(),
    )
    categories = [LABEL_NAMES[label] for label in sorted(LABEL_NAMES)]
    lengths = [
        chart_data.loc[chart_data["category"] == category, "text_length_chars"]
        for category in categories
    ]
    figure, axis = plt.subplots(figsize=(13, 7))
    axis.boxplot(lengths, tick_labels=categories, showfliers=False)
    axis.set(
        title="Longitud de texto por categoría",
        ylabel="Caracteres",
    )
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(TEXT_LENGTHS_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    data = load_and_validate_data()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    profile = build_profile(data)
    profile.to_csv(PROFILE_PATH, encoding="utf-8", index_label="label")
    save_examples(data)
    save_charts(data, profile)

    print(f"Registros analizados: {len(data)}")
    print(f"Perfil por clase: {PROFILE_PATH}")
    print(f"Ejemplos por clase: {EXAMPLES_PATH}")
    print(f"Gráficos: {CLASS_DISTRIBUTION_PATH}, {TEXT_LENGTHS_PATH}")


if __name__ == "__main__":
    main()
