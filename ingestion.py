from langchain.text_splitter import CharacterTextSplitter,  RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from utils import clean_text
import glob
from angle_emb import AnglE

from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import os
import PyPDF2
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
embedding = embedding_functions.SentenceTransformerEmbeddingFunction("WhereIsAI/UAE-Large-V1")

def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ''
    for i in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[i]
        text += page.extract_text()
        return text

def summarize_chunk(chunk):
    summ_text= summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
    return summ_text


def get_chunks():
    all_files = glob.glob("data/rag_data/*.pdf")
    ##all_files = ["D:/projects/FinanceFusionRAG/data/rag_data/Investopedia_B_What_Is_Backoffice.pdf"]
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%d-%m-%y:%H:%M")
    all_chunk = []
    metadata = []
    for file_path in all_files:
        loader = PyMuPDFLoader(file_path)
        doc = loader.load()
        text_splitter =  RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        chunks = text_splitter.split_documents(doc)
        document_name = os.path.basename(file_path)
        for chunk in chunks:
            all_chunk.append(chunk.page_content)
            chunk_summary = summarize_chunk(chunk.page_content)

            metadata.append({"document_name":document_name,"created_date":formatted_date, "chunk_summary":chunk_summary})
        print(f"[Info]: File has been processed.{file_path}")
    return all_chunk, metadata


if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path="chroma_db/")
    collection = chroma_client.create_collection(name="finance_fusion",embedding_function=embedding)
    print("Collection has been created.")
    docs,meta=get_chunks()
    print("Chunking has been done.")
    collection.add(documents=docs, metadatas=meta,ids=[str(i) for i in range(1,len(docs)+1)])
    print("Documents has been added in VectorDB.")
