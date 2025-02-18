from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from core.cag_engine import CAGEngine
from config.config_manager import ConfigManager

# Initialize FastAPI app
app = FastAPI(title="CAG API", version="1.0", description="Cached Augmented Generation API")

# Load Config & Initialize CAG Engine
config_path = "config/config.yaml"
cag_engine = CAGEngine(config_path=config_path)

# Pydantic model for request validation
class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to CAG API!"}

@app.post("/query")
def query_cag(request: QueryRequest):
    """Process a user query using the CAG Engine."""
    start_time = time.time()
    try:
        response = cag_engine.query(request.query)
        execution_time = time.time() - start_time
        return {"query": request.query, "response": response, "execution_time": execution_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Check if the service is running."""
    return {"status": "ok", "message": "CAG API is running smoothly!"}

@app.get("/monitoring")
def get_monitoring_stats():
    """Retrieve monitoring stats from the CAG Engine."""
    if cag_engine.monitoring:
        return cag_engine.monitoring.get_stats()
    return {"status": "monitoring disabled"}
