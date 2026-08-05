import os
import asyncio
import gradio as gr
from .graph import app as graph_app
from .ingest import ingest_document


def upload_document(file):
    if not file:
        return "No file selected."

    path = file if isinstance(file, str) else file.name
    try:
        # Directly call your backend ingestion pipeline
        ingest_document(path)
        filename = os.path.basename(path)
        return f"Uploaded and indexed: {filename}"
    except Exception as e:
        return f"Upload failed: {e}"


def chat_with_ebook(message, history):
    payload = {"question": message}
    try:
        # Directly invoke your LangGraph workflow asynchronously
        result = asyncio.run(graph_app.ainvoke(payload))
        return result.get("answer", "No answer received.")
    except Exception as e:
        return f"Error: Could not process request. ({e})"


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