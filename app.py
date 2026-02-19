import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from src.helper import download_embeddings
from src.prompt import prompt 

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

base_path = Path(__file__).resolve().parent
env_path = base_path / '.env'
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"])

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "medical-public-health-chatbot"

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not found in .env file!")

embeddings = download_embeddings()
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

# Build the RAG Chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

#Route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        result = rag_chain.invoke({"input": user_input})
        return jsonify({"answer": result.get("answer")})
    except Exception as e:
        print(f"Error during RAG: {e}")
        return jsonify({"answer": "I encountered an error processing your request."}), 500

    
if __name__ == "__main__":
    app.run(host = "0.0.0.0",port =5000,debug = True)