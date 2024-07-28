from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import PGVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceLLM
from sqlalchemy import create_engine
import os
import tempfile

app = FastAPI()

# Database configuration
DB_URL = "postgresql://username:password@localhost:5432/your_database"
engine = create_engine(DB_URL)

# LlamaIndex configuration
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = HuggingFaceLLM(model_name="google/flan-t5-base", device_map="auto")
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

vector_store = PGVectorStore.from_params(
    database=DB_URL.split("/")[-1],
    host=DB_URL.split("@")[-1].split(":")[0],
    password=DB_URL.split(":")[2].split("@")[0],
    port=int(DB_URL.split(":")[-1].split("/")[0]),
    user=DB_URL.split("://")[1].split(":")[0],
    table_name="document_embeddings",
    embed_dim=384  # Dimension of the chosen embedding model
)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        documents = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=vector_store, service_context=service_context
        )
        
        os.unlink(temp_file_path)
        return {"message": "Document uploaded and indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class Query(BaseModel):
    text: str

@app.post("/query")
async def query_document(query: Query):
    try:
        index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(query.text)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
