from pci.api.routes.documents import router as documents_router
from pci.api.routes.search import router as search_router
from pci.api.routes.ingestion import router as ingestion_router
from pci.api.routes.system import router as system_router

__all__ = [
    "documents_router",
    "search_router",
    "ingestion_router",
    "system_router",
]
