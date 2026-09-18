import argparse
from functools import lru_cache
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "spanish_news_classifier.joblib"
)


@lru_cache(maxsize=4)
def load_model_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Carga y valida el paquete del modelo."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {model_path}"
        )

    bundle = joblib.load(model_path)

    required_keys = {"model", "label_names", "metadata"}
    missing_keys = required_keys - set(bundle)

    if missing_keys:
        raise ValueError(
            f"El modelo no contiene las claves requeridas: "
            f"{sorted(missing_keys)}"
        )

    return bundle


def predict_text(text: str, model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Clasifica un texto y retorna las tres categorías más probables."""

    clean_text = text.strip()

    if not clean_text:
        raise ValueError("El texto para clasificar está vacío.")

    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    label_names = bundle["label_names"]

    predicted_label = int(model.predict([clean_text])[0])
    probabilities = model.predict_proba([clean_text])[0]

    ranked_predictions = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    top_predictions = [
        {
            "label": int(label),
            "category": label_names[int(label)],
            "probability": round(float(probability), 6),
        }
        for label, probability in ranked_predictions[:3]
    ]

    return {
        "predicted_label": predicted_label,
        "predicted_category": label_names[predicted_label],
        "confidence": top_predictions[0]["probability"],
        "top_3": top_predictions,
        "model_metadata": bundle["metadata"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasifica una noticia en español."
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Texto de la noticia que se desea clasificar.",
    )

    arguments = parser.parse_args()
    result = predict_text(arguments.text)

    print(f"Categoría: {result['predicted_category']}")
    print(f"Confianza: {result['confidence']:.2%}")
    print("Top 3:")

    for prediction in result["top_3"]:
        print(
            f"  {prediction['category']}: "
            f"{prediction['probability']:.2%}"
        )


if __name__ == "__main__":
    main()
