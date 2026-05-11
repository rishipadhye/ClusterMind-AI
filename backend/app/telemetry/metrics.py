from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


REQUEST_COUNT = Counter("clustermind_requests_total", "Total HTTP requests", ["method", "path", "status_code"])
REQUEST_LATENCY = Histogram("clustermind_request_latency_seconds", "HTTP request latency", ["method", "path"])


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path
        with REQUEST_LATENCY.labels(method=method, path=path).time():
            response = await call_next(request)
        REQUEST_COUNT.labels(method=method, path=path, status_code=response.status_code).inc()
        return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

