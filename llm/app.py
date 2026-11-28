from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from llm.schemas import AnalysisResponse, URLAnalysisRequest
from llm.url.service import analyze_url


app = FastAPI(title="P-Project03 AI Server", version="0.1.0")


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/internal/url", response_model=AnalysisResponse)
def analyze_url_endpoint(payload: URLAnalysisRequest) -> dict:
    return analyze_url(payload.id, payload.url)
