import os
from dotenv import load_dotenv
from src.graph import app

# 1. Load keys
load_dotenv()

def run_test(query: str):
    print(f"\n{'='*50}")
    print(f"USER QUESTION: {query}")
    print(f"{'='*50}")
    
    # 2. Run the LangGraph
    # We pass the question as the starting 'State'
    inputs = {"question": query}
    
    # app.invoke runs the whole graph from Start to End
    result = app.invoke(inputs)
    
    # 3. Display the components required by the task
    print("\n[RETRIEVED CONTEXT CHUNKS]:")
    print(result["context"][:500] + "...") # Show first 500 chars
    
    print(f"\n[CONFIDENCE/MATCH SCORE]: {result['score']}")
    
    print("\n[FINAL ANSWER]:")
    print(result["answer"])
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Test 1: A factual question from the PDF
    run_test("What is the definition of Agentic AI?")
    
    # Test 2: The 'Grounding Trap' (Something NOT in the PDF)
    run_test("Who won the FIFA World Cup in 2022?")