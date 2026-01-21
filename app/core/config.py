import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
    GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
    PINECONE_INDEX_NAME = "embedding-benchmark"

settings = Settings()
