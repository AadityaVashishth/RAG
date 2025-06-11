RAG Web Application
Overview
This project implements a Retrieval-Augmented Generation (RAG) system with a web interface, combining document retrieval with large language models to provide accurate, context-aware responses.

Features
Single-script RAG pipeline (all retrieval and generation logic in one file)

Simple web interface (Gradio/Streamlit)

Supports multiple document formats (PDF, TXT)

Customizable chunking and retrieval parameters

Easy to deploy and extend

# Query the system
response = rag.query("What is retrieval-augmented generation?")
print(response)
Project Structure
text
.
├── data/                  # Directory for documents
├── rag.py                 # Core RAG implementation
├── app.py                 # Web interface
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (API keys)
Customization Options
In rag.py:
Change embedding model (line XX)

Adjust chunking parameters (chunk_size, chunk_overlap)

Modify retrieval settings (number of chunks to retrieve)

Change LLM or prompt template

In app.py:
Modify the UI theme/layout

Add examples or additional input fields

Change the web framework (Gradio/Streamlit/FastAPI)

Troubleshooting
Issue: Documents not loading
✅ Ensure files are in the data/ directory
✅ Check file formats are supported (PDF, TXT)

Issue: API errors
✅ Verify your .env file contains correct API keys
✅ Check your internet connection
