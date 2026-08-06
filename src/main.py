import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from src.graph import app as graph_app
from src.ingest import ingest_document

load_dotenv()
app = FastAPI(
    title="DocBuddy API",
    description="Smart Document Assistant & RAG Engine",
)

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "active", "info": "Agentic AI RAG System"}

@app.post("/ask")
async def ask_question(query: Query):
    # This calls your LangGraph
    inputs = {"question": query.question}
    try:
        result = await graph_app.ainvoke(inputs)
        return {
            "answer": result["answer"],
            "confidence": result["score"],
            "context": result["context"]
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "AI model request failed", "details": str(exc)}
        )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".txt", ".docx"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported file type. Upload .pdf, .txt, or .docx."}
        )

    tmp_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(contents)

        ingest_document(tmp_path)
        return {"status": "success", "filename": file.filename}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
