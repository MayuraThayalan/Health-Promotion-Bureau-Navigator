import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from src.helper import download_embeddings
from src.prompt import prompt

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"])

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "medical-public-health-chatbot"

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not found in .env file!")

embeddings = download_embeddings()
bm25_encoder = BM25Encoder().load("bm25_values.json")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index,
    top_k=4,
    text_key="text" 
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain_input(user_input: str) -> dict:
    docs = retriever.invoke(user_input)
    context = format_docs(docs)
    return {"context": context, "input": user_input}

rag_chain = RunnableLambda(build_chain_input) | prompt | llm | StrOutputParser()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        result = rag_chain.invoke(user_input)
        return jsonify({"answer": result})
    except Exception as e:
        print(f"Error during RAG: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": "I encountered an error processing your request."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)