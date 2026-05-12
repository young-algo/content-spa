import os

API_HOST = os.environ.get("PCI_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("PCI_API_PORT", "8000"))
API_DEBUG = os.environ.get("PCI_API_DEBUG", "0") == "1"
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
