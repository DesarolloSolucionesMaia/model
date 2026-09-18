import src.prediction_history as history_store
from fastapi import HTTPException
from src.api import (
    categories,
    health,
    prediction_detail,
    prediction_history,
    public_prediction,
    remove_prediction,
)


def setup_function() -> None:
    history_store.DATABASE_PATH = history_store.PROJECT_ROOT / "data" / "test.db"
    history_store.DATABASE_PATH.unlink(missing_ok=True)


def test_health() -> None:
    assert health()["status"] == "ok"


def test_categories_lists_twelve_categories() -> None:
    assert len(categories()["categories"]) == 12


def test_predict_returns_ranked_probabilities() -> None:
    prediction = public_prediction(
        "El equipo ganó el partido con dos goles en la final."
    )

    assert len(prediction.top_3) == 3
    assert prediction.confidence == prediction.top_3[0].probability
    assert prediction.prediction_id
    assert prediction.source_type == "text"


def test_history_returns_saved_prediction() -> None:
    prediction = public_prediction(
        "El banco central anunció nuevas medidas para la economía."
    )

    history = prediction_history(limit=20, offset=0)
    detail = prediction_detail(prediction.prediction_id)

    assert history.total == 1
    assert history.items[0].prediction_id == prediction.prediction_id
    assert detail.predicted_category == prediction.predicted_category


def test_delete_removes_only_selected_prediction() -> None:
    first = public_prediction("El equipo ganó el partido de fútbol.")
    second = public_prediction("El banco anunció nuevas tasas de interés.")

    response = remove_prediction(first.prediction_id)
    history = prediction_history(limit=20, offset=0)

    assert response.status_code == 204
    assert history.total == 1
    assert history.items[0].prediction_id == second.prediction_id

    try:
        prediction_detail(first.prediction_id)
    except HTTPException as error:
        assert error.status_code == 404
    else:
        raise AssertionError("La predicción eliminada todavía existe.")
