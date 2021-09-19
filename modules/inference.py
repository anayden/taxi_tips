from fastapi import FastAPI
from starlette_exporter import PrometheusMiddleware, handle_metrics

from .model import model
from .inference_request import InferenceRequest

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)


@app.post("/predict")
def predict(inference_request: InferenceRequest):
    return {
        "tip": model.predict(inference_request.to_nparray()).tolist()[0]
    }


@app.get("/health")
def health_check():
    return "OK"
