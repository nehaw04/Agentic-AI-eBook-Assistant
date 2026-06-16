import os
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001/ask")
UPLOAD_URL = os.getenv("UPLOAD_URL", "http://127.0.0.1:8001/upload")


def upload_document(file):
    if not file:
        return "No file selected."

    path = file if isinstance(file, str) else file.name
    try:
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f)}
            response = requests.post(UPLOAD_URL, files=files)
            response.raise_for_status()
            data = response.json()
            return f"Uploaded: {data.get('filename')}"
    except Exception as e:
        return f"Upload failed: {e}"


def chat_with_ebook(message, history):
    payload = {"question": message}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "No answer received.")
    except Exception as e:
        return f"Error: Could not connect to the AI server. ({e})"


demo = gr.Blocks()
with demo:
    
    gr.Markdown("Upload a PDF, TXT, or DOCX file first, then ask questions about its content.")

    with gr.Row():
        file_input = gr.File(label="Upload document", file_types=[".pdf", ".txt", ".docx"])
        upload_button = gr.Button("Upload Document")
    upload_status = gr.Textbox(label="Upload Status", interactive=False)

    upload_button.click(upload_document, inputs=file_input, outputs=upload_status)

    gr.Markdown("---")

    gr.ChatInterface(
        fn=chat_with_ebook,
        title="DocBuddy: Smart Document Assistant & RAG Engine",
        description="This bot uses Strict Grounding to ensure accuracy.",
        examples=["Summarise the Doc", "What is the central theme?"]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
