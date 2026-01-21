from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
import time

class PineconeService:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.spec = ServerlessSpec(cloud="aws", region="us-east-1") # Defaulting to us-east-1, can be changed

    def create_index_if_not_exists(self, index_name: str, dimension: int):
        """
        Create a Pinecone index if it doesn't exist.
        """
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"Creating index: {index_name} with dimension {dimension}")
            try:
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=self.spec
                )
                # Wait for index to be ready
                while not self.pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
                print(f"Index {index_name} created successfully.")
            except Exception as e:
                print(f"Error creating index {index_name}: {e}")
        else:
            print(f"Index {index_name} already exists.")

    def upsert_vectors(self, index_name: str, vectors: list):
        """
        Upsert vectors to the specified index.
        vectors format: [(id, values, metadata), ...]
        """
        try:
            index = self.pc.Index(index_name)
            index.upsert(vectors=vectors)
            print(f"Upserted {len(vectors)} vectors to {index_name}")
        except Exception as e:
            print(f"Error upserting to {index_name}: {e}")

    def query_vectors(self, index_name: str, vector: list, top_k: int = 5):
        """
        Query the specified index.
        """
        try:
            index = self.pc.Index(index_name)
            return index.query(vector=vector, top_k=top_k, include_metadata=True)
        except Exception as e:
            print(f"Error querying {index_name}: {e}")
            return None

    def delete_index(self, index_name: str):
         if index_name in [index.name for index in self.pc.list_indexes()]:
             self.pc.delete_index(index_name)
