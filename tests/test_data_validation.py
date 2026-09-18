import pandas as pd
import pytest

from data_validation import EXPECTED_LABELS, load_and_validate_data


def valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"Noticia de prueba {label}" for label in sorted(EXPECTED_LABELS)],
            "label": sorted(EXPECTED_LABELS),
        }
    )


def test_load_and_validate_data_normalizes_text(tmp_path):
    data = valid_data()
    data.loc[0, "text"] = "  Noticia de prueba 0  "
    dataset_path = tmp_path / "dataset.parquet"
    data.to_parquet(dataset_path)

    result = load_and_validate_data(dataset_path)

    assert result.loc[0, "text"] == "Noticia de prueba 0"
    assert set(result["label"]) == EXPECTED_LABELS


def test_load_and_validate_data_rejects_missing_labels(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    valid_data().iloc[:-1].to_parquet(dataset_path)

    with pytest.raises(ValueError, match="Etiquetas inesperadas"):
        load_and_validate_data(dataset_path)


def test_load_and_validate_data_rejects_duplicate_texts(tmp_path):
    data = valid_data()
    data.loc[1, "text"] = data.loc[0, "text"]
    dataset_path = tmp_path / "dataset.parquet"
    data.to_parquet(dataset_path)

    with pytest.raises(ValueError, match="textos duplicados"):
        load_and_validate_data(dataset_path)


def test_load_and_validate_data_rejects_empty_texts(tmp_path):
    data = valid_data()
    data.loc[0, "text"] = "   "
    dataset_path = tmp_path / "dataset.parquet"
    data.to_parquet(dataset_path)

    with pytest.raises(ValueError, match="textos vacíos"):
        load_and_validate_data(dataset_path)
