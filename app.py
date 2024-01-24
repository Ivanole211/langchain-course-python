import os
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from pinecone import Pinecone

load_dotenv()

app = FastAPI()

# Setup environment variables
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")

# Initialize pinecone client with the specified environment
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Access existing index
index = pc.Index(name=index_name)

# Middleware to secure HTTP endpoint
security = HTTPBearer()

def validate_token(
    http_auth_credentials: HTTPAuthorizationCredentials = Security(security),
):
    if http_auth_credentials.scheme.lower() == "bearer":
        token = http_auth_credentials.credentials
        if token != os.getenv("RENDER_API_TOKEN"):
            raise HTTPException(status_code=403, detail="Invalid token")
    else:
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")

class QueryModel(BaseModel):
    query: str

@app.post("/")
async def get_context(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    # convert query to embeddings
    res = openai_client.embeddings.create(
        input=[query_data.query], model="text-embedding-ada-002"
    )
    embedding = res.data[0].embedding

    # Search for matching Vectors using keyword arguments
    results = index.query(vector=embedding, top_k=6, include_metadata=True).to_dict()

    # Filter out metadata from search result
    context = []
    for match in results.get("matches", []):
        if "metadata" in match and "key" in match["metadata"] and "text" in match["metadata"]:
            namespace = match["metadata"]["key"].split(":")[0]
            if namespace in ["QA", "Message"]:
                context.append(match["metadata"]["text"])

    # Return context
    return context



# @app.get("/")
# async def get_context(query: str = None, credentials: HTTPAuthorizationCredentials = Depends(validate_token)):

#     # convert query to embeddings
#     res = openai_client.embeddings.create(
#         input=[query],
#         model="text-embedding-ada-002"
#     )
#     embedding = res.data[0].embedding
#     # Search for matching Vectors
#     results = index.query(embedding, top_k=6, include_metadata=True).to_dict()
#     # Filter out metadata fron search result
#     context = [match['metadata']['text'] for match in results['matches']]
#     # Retrun context
#     return context
