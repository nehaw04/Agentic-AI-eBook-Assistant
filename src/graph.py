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
    embeddings = SlicedGeminiEmbeddings(model="gemini-embedding-2")
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    
    docs = vectorstore.similarity_search_with_score(state["question"], k=3)
    if not docs:
        return {
            "context": "No relevant document content was found. Please upload a document and try again.",
            "score": 0.0,
        }

    context_text = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc, score in docs
        ]
    )
    return {"context": context_text, "score": docs[0][1]}

from langchain_google_genai import ChatGoogleGenerativeAI

def generate_node(state: GraphState):
    print("---GENERATING ANSWER---")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    
    prompt = (
        "You are an expert assistant. Answer using ONLY the provided context below. "
        "If the answer is not contained in the context, respond with 'I don't know based on the document.' "
        "Do not invent or hallucinate facts.\n\n"
        f"Context:\n{state['context']}\n\n"
        f"Question:\n{state['question']}"
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