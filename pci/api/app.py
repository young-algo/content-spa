import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pci.api.config import API_DEBUG, FRONTEND_DIST
from pci.api.routes import (
    documents_router,
    search_router,
    ingestion_router,
    system_router,
)


_HAS_FRONTEND = os.path.isdir(FRONTEND_DIST) and os.path.isfile(
    os.path.join(FRONTEND_DIST, "index.html")
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Content Index",
        version="0.1.0",
        docs_url="/api/docs" if API_DEBUG else None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(ingestion_router)
    app.include_router(system_router)

    if _HAS_FRONTEND:
        app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

        @app.get("/", response_class=HTMLResponse)
        async def serve_index():
            return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

        @app.get("/{catch_all:path}", response_class=HTMLResponse)
        async def serve_spa(catch_all: str):
            index_path = os.path.join(FRONTEND_DIST, "index.html")
            if os.path.isfile(index_path):
                return FileResponse(index_path)
            return HTMLResponse("<h1>Frontend not built</h1>", status_code=404)

    return app


app = create_app()
