"""HTTP interface for an offline, release-approved LMP scorer."""

import asyncio
from contextlib import asynccontextmanager
import hmac
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from lmp import Scorer

LOGGER = logging.getLogger(__name__)
MAX_BODY_BYTES = 256 * 1024


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    id: str | None = Field(default=None, max_length=100)
    language: Literal["en", "hi"]
    original: str = Field(min_length=1, max_length=4000)
    simplified: str = Field(min_length=1, max_length=4000)
    context: str = Field(default="", max_length=8000)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    pairs: list[Pair] = Field(min_length=1, max_length=16)


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        token = os.environ.get("LMP_API_TOKEN", "")
        if len(token) < 32 or not token.isascii():
            raise RuntimeError("LMP_API_TOKEN must contain at least 32 ASCII characters")
        path = Path(os.environ.get("LMP_MODEL_PATH", "out/release/model.json"))
        allow_synthetic = os.environ.get("LMP_ALLOW_SYNTHETIC") == "1"
        app.state.scorer = await run_in_threadpool(
            lambda: Scorer(path, local_only=True, allow_demo=allow_synthetic,
                           require_release=True))
        app.state.model_origin = "synthetic" if app.state.scorer.artifact.get("demo") is True else "research"
        app.state.token = token
        app.state.capacity = asyncio.Semaphore(1)
        yield

    app = FastAPI(title="Legal Meaning Preservation", version="1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return {"status": "ready"}

    @app.post("/v1/score", responses={
        401: {"description": "Bearer token required"},
        413: {"description": "Request too large"},
        415: {"description": "JSON required"},
        422: {"description": "Invalid pairs"},
        503: {"description": "Scoring capacity exhausted"},
    })
    async def score(request: Request):
        provided = request.headers.get("authorization", "")
        expected = f"Bearer {app.state.token}"
        if not hmac.compare_digest(provided.encode(), expected.encode()):
            raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            raise HTTPException(415, "Use application/json")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                raise HTTPException(413, "Request too large")
            body.extend(chunk)
        try:
            batch = Batch.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(422, "Provide 1 to 16 valid English or Hindi pairs") from exc

        if app.state.capacity.locked():
            raise HTTPException(503, "Scorer busy", headers={"Retry-After": "1"})
        async with app.state.capacity:
            try:
                result = await run_in_threadpool(
                    app.state.scorer.predict, [pair.model_dump() for pair in batch.pairs])
            except ValueError as exc:
                raise HTTPException(422, "A passage exceeds the model token budget") from exc
            except Exception:
                LOGGER.error("Scoring failed")
                raise HTTPException(500, "Scoring failed") from None
        public_fields = ("id", "language", "score", "decision", "mode", "evidence")
        return {"model_origin": app.state.model_origin,
                "predictions": [{field: item[field] for field in public_fields}
                                 for item in result]}

    return app
