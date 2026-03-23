# RAG Web Application

# Overview
This project implements a Retrieval-Augmented Generation (RAG) system with a web interface, combining document retrieval with large language models to provide accurate, context-aware responses.

# Features
1. Single-script RAG pipeline (all retrieval and generation logic in one file)
2. Simple web interface (Gradio/Streamlit)
3. Supports multiple document formats (PDF, TXT)
4. Customizable chunking and retrieval parameters
5. Easy to deploy and extend

# Query the system
1. response = rag.query("What is retrieval-augmented generation?")
2. print(response)
3. Project Structure
4. text
   
# File Structure
1. data/  - Directory for documents
2. rag.py - Core RAG implementation
3. app.py - Web interface
4. requirements.txt - Python dependencies
5. .env - Environment variables (API keys)

# Customization Options
In rag.py:
1. Change embedding model
2. Adjust chunking parameters (chunk_size, chunk_overlap)
3. Modify retrieval settings (number of chunks to retrieve)
4. Change LLM or prompt template

In app.py:
1. Modify the UI theme/layout
2. Add examples or additional input fields
3. Change the web framework (Gradio/Streamlit/FastAPI)

# Troubleshooting

1. Issue: Documents not loading - Ensure files are in the "data/" directory & check if the file formats are supported(PDF, TXT).

2. Issue: API errors - Verify your .env file contains correct API keys & Check your internet connection
