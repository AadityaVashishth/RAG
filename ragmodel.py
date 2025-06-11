import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader, JSONLoader
)
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)

class RAGPipeline:
    def __init__(self):
        self.llm = self._load_model()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        # Use a dedicated embeddings model (Mistral is not for embeddings)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = None
        self.documents = []
        self.summary = ""

    def _load_model(self):
        logging.info("Loading Mistral-7B-Instruct...")
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        
        # Tokenizer without revision (use latest)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # 4-bit quantization to reduce GPU memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )
        
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512  # Limit response length
        )

    def load_file(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext in [".xls", ".xlsx"]:
            loader = UnstructuredExcelLoader(file_path)
        elif ext == ".json":
            loader = JSONLoader(file_path, jq_schema=".", text_content=False)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader.load()

    def process_file(self, file_path):
        logging.info(f"Processing file: {file_path}")
        self.documents = self.load_file(file_path)
        chunks = self.text_splitter.split_documents(self.documents)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.summary = self.generate_summary()
        return "File processed successfully"

    def generate_summary(self) -> str:
        if not self.documents:
            return "No documents to summarize."
        full_text = " ".join([doc.page_content for doc in self.documents])
        prompt = f"""<s>[INST] Summarize the following in 3-5 sentences: {full_text} [/INST]"""
        output = self.llm(prompt)[0]["generated_text"]
        return output.split("[/INST]")[-1].strip()

    def answer_question(self, query: str) -> str:
        if not self.vectorstore:
            return "Please upload a file first."
        docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""<s>[INST] Answer based on the context below. If unsure, say "I don't know."\n\nContext: {context}\n\nQuestion: {query} [/INST]"""
        output = self.llm(prompt)[0]["generated_text"]
        return output.split("[/INST]")[-1].strip()