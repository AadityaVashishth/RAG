from ragmodel import RAGPipeline
import gradio as gr

rag = RAGPipeline()

def handle_upload(file):    
    if file is None:
        return "Please upload a file first."
    result = rag.process_file(file.name)
    return f"✅ {result}\n\nSummary:\n\n{rag.summary}"

def handle_question(question):
    return rag.answer_question(question)

with gr.Blocks(title="RAG Document Q&A") as demo:
    gr.Markdown("# 🧠 Mistral RAG Assistant")
    
    with gr.Row():
        file_input = gr.File(label="📁 Upload File", file_types=[".pdf", ".csv", ".xls", ".xlsx", ".json", ".txt"])
        upload_output = gr.Textbox(label="Status", lines=5)

    upload_button = gr.Button("Upload & Process")
    upload_button.click(fn=handle_upload, inputs=[file_input], outputs=[upload_output])

    question_input = gr.Textbox(label="Ask a question", placeholder="What is this document about?")
    ask_button = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=5)

    ask_button.click(fn=handle_question, inputs=[question_input], outputs=[answer_output])

demo.launch()