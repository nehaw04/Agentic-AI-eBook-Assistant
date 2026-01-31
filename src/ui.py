import gradio as gr
import requests

# The URL of your running FastAPI server
API_URL = "http://127.0.0.1:8000/ask"

def chat_with_ebook(message, history):
    # 1. Format the request for your FastAPI endpoint
    payload = {"question": message}
    
    try:
        # 2. Send the request to your backend
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Raise error for 4xx/5xx codes
        data = response.json()
        
        # 3. Extract the answer and context
        answer = data.get("answer", "No answer received.")
        context = data.get("context_used", "")
        
        # We return the answer; Gradio handles the chat history automatically
        return answer
        
    except Exception as e:
        return f"Error: Could not connect to the AI server. ({e})"

# Create the Gradio Interface (Remove 'theme' from here)
demo = gr.ChatInterface(
    fn=chat_with_ebook,
    title="Agentic AI eBook Assistant",
    description="Ask anything about the Agentic AI for Executives eBook. This bot uses Strict Grounding to ensure accuracy.",
    examples=["What is the definition of Agentic AI?", "What value does it bring to businesses?"],
    # theme="soft"  <-- DELETE THIS LINE FROM HERE
)

if __name__ == "__main__":
    # Move 'theme' to the launch() method instead
    demo.launch(theme="soft", share=True)