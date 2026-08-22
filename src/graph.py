import os
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from src.ingest import SlicedGeminiEmbeddings


# ---------------------------------------------------------------------------
# 1. State & Structured Output Schemas
# ---------------------------------------------------------------------------
class StructuredAnswer(BaseModel):
    answer: str = Field(description="Direct answer based strictly on the context.")
    source_quote: str = Field(description="The exact or nearest text segment from the document used to support the answer.")


class GraphState(TypedDict):
    question: str
    original_question: str
    context: str
    raw_chunks: List[str]
    answer: str
    source_quote: str
    score: float
    retry_count: int
    is_relevant: bool
    is_validated: bool


# ---------------------------------------------------------------------------
# Helper: Cosine Similarity for Quote Validation
# ---------------------------------------------------------------------------
def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# 2. Node Implementations
# ---------------------------------------------------------------------------

# Stage 1: Retriever (Top-5 chunks, 512 dimensions)
def retrieve_node(state: GraphState):
    print(f"---[STAGE 1] RETRIEVE (Attempt {state.get('retry_count', 0) + 1}/3)---")
    
    embeddings = SlicedGeminiEmbeddings(model="models/text-embedding-004")
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )

    query = state["question"]
    docs = vectorstore.similarity_search_with_score(query, k=5)
    
    if not docs:
        return {
            "context": "",
            "raw_chunks": [],
            "score": 0.0,
            "original_question": state.get("original_question") or query
        }

    top_score = float(docs[0][1])
    raw_chunks = [doc.page_content for doc, _ in docs]
    context_text = "\n\n".join(
        [f"Source: {doc.metadata.get('source', 'document')}\n{doc.page_content}" for doc, _ in docs]
    )

    return {
        "context": context_text,
        "raw_chunks": raw_chunks,
        "score": top_score,
        "original_question": state.get("original_question") or query
    }


# Stage 2: Relevance Checker & Query Rewriter
def relevance_checker_node(state: GraphState):
    print("---[STAGE 2] RELEVANCE CHECKER---")
    score = state.get("score", 0.0)
    current_retries = state.get("retry_count", 0)

    # Threshold check: 0.7
    if score >= 0.70:
        return {"is_relevant": True}

    # If score < 0.7 and retries remaining, rewrite query
    if current_retries < 3:
        print(f"---[REWRITE] Score {score:.2f} < 0.70. Rewriting query (Retry {current_retries + 1})...---")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        rewrite_prompt = (
            "You are an expert query optimizer for a document search system. "
            "Rewrite the following search query into a clearer, more specific semantic search formulation:\n\n"
            f"Query: {state['question']}\n\n"
            "Return ONLY the rewritten query text."
        )
        rewritten = llm.invoke(rewrite_prompt).content.strip()
        return {
            "is_relevant": False,
            "question": rewritten,
            "retry_count": current_retries + 1
        }

    # Exceeded max retries
    return {"is_relevant": False, "retry_count": current_retries + 1}


# Stage 3: Generator (Structured Output with Source Quote)
def generate_node(state: GraphState):
    print("---[STAGE 3] GENERATOR---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(StructuredAnswer)

    prompt = (
        "You are an enterprise document assistant. Answer the question using ONLY the provided context. "
        "You must extract and provide a direct `source_quote` from the context that proves your answer. "
        "Do not invent or extrapolate.\n\n"
        f"Context:\n{state['context']}\n\n"
        f"Question:\n{state.get('original_question', state['question'])}"
    )

    try:
        response: StructuredAnswer = structured_llm.invoke(prompt)
        return {
            "answer": response.answer,
            "source_quote": response.source_quote
        }
    except Exception:
        # Fallback if parsing fails
        return {
            "answer": "I cannot find this in the document.",
            "source_quote": ""
        }


# Stage 4: Validator (Cosine Similarity Quote Check >= 0.70)
def validator_node(state: GraphState):
    print("---[STAGE 4] VALIDATOR (SEMANTIC QUOTE CHECK)---")
    
    source_quote = state.get("source_quote", "").strip()
    raw_chunks = state.get("raw_chunks", [])

    if not source_quote or not raw_chunks:
        return {
            "answer": "I cannot find this in the document.",
            "is_validated": False
        }

    embeddings = SlicedGeminiEmbeddings(model="models/text-embedding-004")
    quote_vec = embeddings.embed_query(source_quote)
    chunk_vecs = embeddings.embed_documents(raw_chunks)

    # Compute maximum similarity across all retrieved chunks
    max_sim = max(
        [compute_cosine_similarity(quote_vec, c_vec) for c_vec in chunk_vecs],
        default=0.0
    )
    print(f"---[VALIDATOR] Max Cosine Similarity of Quote to Context: {max_sim:.3f}---")

    if max_sim >= 0.70:
        return {"is_validated": True}
    
    # Block ungrounded answer
    print("---[VALIDATOR BLOCKED] Quote failed threshold (< 0.70)---")
    return {
        "answer": "I cannot find this in the document.",
        "is_validated": False
    }


# Fallback Node for out-of-scope / ungrounded queries
def fallback_node(state: GraphState):
    print("---[FALLBACK NODE] OUT OF SCOPE---")
    return {
        "answer": "I cannot find this in the document.",
        "is_relevant": False,
        "is_validated": False
    }


# ---------------------------------------------------------------------------
# 3. Deterministic Python Routing Edges
# ---------------------------------------------------------------------------
def route_after_relevance(state: GraphState) -> Literal["generate", "retrieve", "fallback"]:
    if state.get("is_relevant"):
        return "generate"
    if state.get("retry_count", 0) < 3:
        return "retrieve"
    return "fallback"


# ---------------------------------------------------------------------------
# 4. Build LangGraph StateMachine
# ---------------------------------------------------------------------------
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("relevance_checker", relevance_checker_node)
workflow.add_node("generate", generate_node)
workflow.add_node("validator", validator_node)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "relevance_checker")

workflow.add_conditional_edges(
    "relevance_checker",
    route_after_relevance,
    {
        "generate": "generate",
        "retrieve": "retrieve",
        "fallback": "fallback"
    }
)

workflow.add_edge("generate", "validator")
workflow.add_edge("validator", END)
workflow.add_edge("fallback", END)

app = workflow.compile()
