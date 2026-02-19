from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
from src.helper import load_pdf_files, filter_metadata, text_splitter, download_embeddings

load_dotenv()       

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data = load_pdf_files(data="data/")
filter_data = filter_metadata(extracted_data)
text_chunks = text_splitter(filter_data)

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-public-health-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension= 384, #dimension of the embeddings
        metric= "cosine", #cosine similarity
        spec = ServerlessSpec(cloud="aws",region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name = index_name,
    embedding= embeddings,
)

