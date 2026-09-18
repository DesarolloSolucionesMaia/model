import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from data_validation import LABEL_NAMES
from predict import load_model_bundle, predict_text


def saved_bundle(tmp_path):
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", DummyClassifier(strategy="constant", constant=1)),
        ]
    )
    model.fit(["economía", "deportes"], [0, 1])
    model_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "label_names": LABEL_NAMES,
            "metadata": {"model_type": "test"},
        },
        model_path,
    )
    return model_path


def test_predict_text_returns_ordered_top_predictions(tmp_path):
    result = predict_text("El equipo ganó el campeonato", saved_bundle(tmp_path))

    assert result["predicted_label"] == 1
    assert result["predicted_category"] == LABEL_NAMES[1]
    assert result["top_3"][0]["probability"] == 1.0
    assert len(result["top_3"]) == 2


def test_predict_text_rejects_blank_text():
    with pytest.raises(ValueError, match="está vacío"):
        predict_text("  ")


def test_load_model_bundle_rejects_missing_keys(tmp_path):
    model_path = tmp_path / "invalid.joblib"
    joblib.dump({"model": "not-a-model"}, model_path)

    with pytest.raises(ValueError, match="claves requeridas"):
        load_model_bundle(model_path)
