import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from src.graph import app as graph_app

load_dotenv()

app = FastAPI(title="Agentic AI RAG API")

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "active", "info": "Agentic AI RAG System"}

@app.post("/ask")
async def ask_question(query: Query):
    # This calls your LangGraph
    inputs = {"question": query.question}
    result = await graph_app.ainvoke(inputs)
    
    return {
        "answer": result["answer"],
        "confidence": result["score"],
        "context": result["context"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)