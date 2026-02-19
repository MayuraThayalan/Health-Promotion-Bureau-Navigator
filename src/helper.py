from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls = PyPDFLoader
    )

    documents = loader.load()
    return documents


#filtering metadata from the docs
def filter_metadata(docs):
    minimal_docs = []

    for doc in docs:
        source = doc.metadata.get("source")

        new_doc = Document(
            page_content=doc.page_content,
            metadata={"source": source}
        )

        minimal_docs.append(new_doc)

    return minimal_docs


#split the doc into small chunks

def text_splitter(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks


def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embeddings

embedding = download_embeddings()