import io
import os
from hashlib import sha256
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

from src.predict import DEFAULT_MODEL_PATH, load_model_bundle, predict_text
from src.prediction_history import (
    delete_prediction,
    get_prediction,
    initialize_database,
    list_predictions,
    save_prediction,
)


API_VERSION = "1.0.0"
MAX_PDF_BYTES = 50 * 1024 * 1024
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=50)


class RankedPrediction(BaseModel):
    label: int
    category: str
    probability: float


class PredictionResponse(BaseModel):
    prediction_id: str
    created_at: str
    source_type: str
    file_name: str | None
    predicted_label: int
    predicted_category: str
    confidence: float
    top_3: list[RankedPrediction]


class PredictionDetail(PredictionResponse):
    model_trained_at: str | None = None


class PredictionHistory(BaseModel):
    items: list[PredictionDetail]
    total: int
    limit: int
    offset: int


def public_prediction(
    text: str,
    *,
    source_type: str = "text",
    file_name: str | None = None,
    document_hash: str | None = None,
) -> PredictionResponse:
    try:
        result = predict_text(text, MODEL_PATH)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    public_result = {
        key: result[key]
        for key in ("predicted_label", "predicted_category", "confidence", "top_3")
    }
    stored_result = save_prediction(
        public_result,
        source_type=source_type,
        file_name=file_name,
        document_hash=document_hash,
        model_trained_at=result["model_metadata"].get("trained_at_utc"),
    )
    return PredictionResponse(**stored_result)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model_bundle(MODEL_PATH)
    initialize_database()
    yield


app = FastAPI(
    title="API del clasificador de noticias en español",
    description="Sirve inferencias del modelo TF-IDF + regresión logística.",
    version=API_VERSION,
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.get("/health", tags=["operación"])
def health() -> dict:
    return {"status": "ok", "version": API_VERSION}


@app.get("/v1/model", tags=["modelo"])
def model_info() -> dict:
    bundle = load_model_bundle(MODEL_PATH)
    return {"metadata": bundle["metadata"], "model_file": MODEL_PATH.name}


@app.get("/v1/categories", tags=["modelo"])
def categories() -> dict:
    label_names = load_model_bundle(MODEL_PATH)["label_names"]
    return {
        "categories": [
            {"label": int(label), "category": category}
            for label, category in sorted(label_names.items())
        ]
    }


@app.get(
    "/v1/predictions",
    response_model=PredictionHistory,
    tags=["historial"],
)
def prediction_history(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PredictionHistory:
    return PredictionHistory(**list_predictions(limit, offset))


@app.get(
    "/v1/predictions/{prediction_id}",
    response_model=PredictionDetail,
    tags=["historial"],
)
def prediction_detail(prediction_id: str) -> PredictionDetail:
    prediction = get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontró la predicción.",
        )
    return PredictionDetail(**prediction)


@app.delete(
    "/v1/predictions/{prediction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["historial"],
)
def remove_prediction(prediction_id: str) -> Response:
    if not delete_prediction(prediction_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontró la predicción.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/v1/predictions", response_model=PredictionResponse, tags=["predicción"])
def predict(request: PredictionRequest) -> PredictionResponse:
    return public_prediction(request.text)


@app.post(
    "/v1/predictions/batch",
    response_model=list[PredictionResponse],
    tags=["predicción"],
)
def predict_batch(request: BatchPredictionRequest) -> list[PredictionResponse]:
    return [public_prediction(text, source_type="batch") for text in request.texts]


@app.post("/v1/predictions/pdf", response_model=PredictionResponse, tags=["predicción"])
async def predict_pdf(
    file: Annotated[UploadFile, File(description="Documento PDF de máximo 50 MB")],
) -> PredictionResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="El archivo debe ser un PDF.",
        )

    content = await file.read(MAX_PDF_BYTES + 1)
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="El PDF supera el límite de 50 MB.",
        )

    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fue posible leer el PDF.",
        ) from error

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El PDF no contiene texto extraíble.",
        )

    safe_file_name = Path(file.filename).name if file.filename else None
    return public_prediction(
        text,
        source_type="pdf",
        file_name=safe_file_name,
        document_hash=sha256(content).hexdigest(),
    )
