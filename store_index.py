from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from src.helper import load_pdf_files, filter_metadata, text_splitter, download_embeddings

load_dotenv()       

extracted_data = load_pdf_files(data="data/")
filter_data = filter_metadata(extracted_data)
text_chunks = text_splitter(filter_data)
embeddings = download_embeddings()

#Initializing the Database
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "medical-public-health-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension= 384, #dimension of the embeddings
        metric= "dotproduct", 
        spec = ServerlessSpec(cloud="aws",region="us-east-1")
    )

index = pc.Index(index_name)

#Training the keyword search
bm25_encoder = BM25Encoder().default()
bm25_encoder.fit([d.page_content for d in text_chunks])
bm25_encoder.dump("bm25_values.json")

#Creating the hybrid retriver
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, 
    sparse_encoder=bm25_encoder, 
    index=index
)
retriever.add_documents(text_chunks)

