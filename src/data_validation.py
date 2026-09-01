from pathlib import Path

import pandas as pd


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "pdf-data"
    / "data"
    / "raw"
    / "spanish-news-classification"
    / "data"
    / "train-00000-of-00001.parquet"
)

EXPECTED_COLUMNS = {"text", "label"}
LABEL_NAMES = {
    0: "Alimentación",
    1: "Astronomía",
    2: "Economía",
    3: "Moda",
    4: "Medicina",
    5: "Defensa",
    6: "Motor",
    7: "Entretenimiento",
    8: "Política",
    9: "Religión",
    10: "Deportes",
    11: "Tecnología",
}

EXPECTED_LABELS = set(LABEL_NAMES)


def load_and_validate_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Carga el dataset y valida las condiciones mínimas para entrenar."""

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {path}")

    dataframe = pd.read_parquet(path)

    missing_columns = EXPECTED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("El dataset está vacío.")

    if dataframe[["text", "label"]].isna().any().any():
        raise ValueError("El dataset contiene valores nulos.")

    clean_text = dataframe["text"].astype(str).str.strip()

    if clean_text.eq("").any():
        raise ValueError("El dataset contiene textos vacíos.")

    duplicate_count = dataframe.duplicated(subset=["text"]).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"El dataset contiene {duplicate_count} textos duplicados."
        )

    actual_labels = set(dataframe["label"].unique())
    if actual_labels != EXPECTED_LABELS:
        raise ValueError(
            f"Etiquetas inesperadas. Encontradas: {sorted(actual_labels)}"
        )

    dataframe = dataframe.copy()
    dataframe["text"] = clean_text

    return dataframe


def print_data_summary(dataframe: pd.DataFrame) -> None:
    """Imprime un resumen verificable del dataset."""

    text_lengths = dataframe["text"].str.len()
    distribution = (
    dataframe["label"]
    .value_counts()
    .sort_index()
    .rename("registros")
    .to_frame()
    )

    distribution["categoría"] = distribution.index.map(LABEL_NAMES)
    distribution = distribution[["categoría", "registros"]]

    print(f"Registros: {len(dataframe)}")
    print(f"Columnas: {dataframe.columns.tolist()}")
    print(f"Clases: {dataframe['label'].nunique()}")
    print(f"Duplicados: {dataframe.duplicated(subset=['text']).sum()}")
    print(f"Textos vacíos: {dataframe['text'].eq('').sum()}")
    print("Distribución por clase:")
    print(distribution)
    print("Longitud de textos:")
    print(
        {
            "mínima": int(text_lengths.min()),
            "mediana": int(text_lengths.median()),
            "promedio": round(float(text_lengths.mean()), 2),
            "máxima": int(text_lengths.max()),
        }
    )


if __name__ == "__main__":
    data = load_and_validate_data()
    print_data_summary(data)