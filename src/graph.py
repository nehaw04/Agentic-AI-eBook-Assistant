import os
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from src.ingest import SlicedGeminiEmbeddings 

# 1. Define the State
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    score: float

# 2. Define the Nodes
def retrieve_node(state: GraphState):
    print("---RETRIEVING FROM PINECONE---")
    # Setup 512-dim embeddings
    embeddings = SlicedGeminiEmbeddings(model="models/text-embedding-004")
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("agentic-ai-index"), 
        embedding=embeddings
    )
    
    docs = vectorstore.similarity_search_with_score(state["question"], k=3)
    context_text = "\n\n".join([doc.page_content for doc, score in docs])
    return {"context": context_text, "score": docs[0][1] if docs else 0.0}

from langchain_google_genai import ChatGoogleGenerativeAI

def generate_node(state: GraphState):
    print("---GENERATING ANSWER---")
    
    # We remove 'client_options' entirely to stop the Pydantic error.
    # We use 'gemini-1.5-flash' which is the most stable 2026 entry point.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
        # If you still get a 404 after this, we will try a different model name
    )
    
    prompt = (
        f"Answer based ONLY on the context:\n"
        f"Context: {state['context']}\n"
        f"Question: {state['question']}"
    )
    
    response = llm.invoke(prompt)
    return {"answer": response.content}
# 3. Build Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()