"""Entry point for the FastAPI server."""
import uvicorn
from src.api.routes import app

if __name__ == "__main__":
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
