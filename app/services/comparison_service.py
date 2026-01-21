import numpy as np
import pandas as pd
from app.services.openai_service import OpenAIEmbeddingService
from app.services.gemini_service import GeminiEmbeddingService
from app.services.pinecone_service import PineconeService
from app.core.config import settings
import uuid

class ComparisonService:
    def __init__(self):
        self.openai_service = OpenAIEmbeddingService()
        self.gemini_service = GeminiEmbeddingService()
        self.pinecone_service = PineconeService()
        
        # Dimensions
        self.openai_dim = 3072 # text-embedding-3-large
        self.gemini_dim = 768  # text-embedding-004

    def cosine_similarity_matrix(self, embeddings):
        """
        Calculate cosine similarity matrix for a list of embeddings.
        """
        if not embeddings:
            return None
        
        embeds_arr = np.array(embeddings)
        norm = np.linalg.norm(embeds_arr, axis=1, keepdims=True)
        # Avoid division by zero
        norm[norm == 0] = 1
        normalized_embeds = embeds_arr / norm
        return np.dot(normalized_embeds, normalized_embeds.T)

    def process_openai_embeddings(self, sentences: list[str]):
        """
        Process OpenAI embeddings: generate, store, and return details.
        """
        print("Processing OpenAI Embeddings...")
        openai_embeds = self.openai_service.get_embeddings(sentences)
        if not openai_embeds:
            return None
            
        result = {
            'embeddings': openai_embeds,
            'similarity_matrix': self.cosine_similarity_matrix(openai_embeds).tolist(),
            'dimension': len(openai_embeds[0])
        }
        
        # Store in Pinecone
        index_name = f"{settings.PINECONE_INDEX_NAME}-openai"
        self.pinecone_service.create_index_if_not_exists(index_name, self.openai_dim)
        vectors = []
        for i, embed in enumerate(openai_embeds):
            # Use a deterministic ID based on index or hash, or random UUID
            # For this demo, simple ID is fine, but for separate calls, maybe UUID
            vector_id = f"openai-{uuid.uuid4()}"
            vectors.append((vector_id, embed, {"text": sentences[i]}))
        self.pinecone_service.upsert_vectors(index_name, vectors)
        
        return result

    def process_gemini_embeddings(self, sentences: list[str]):
        """
        Process Gemini embeddings: generate, store, and return details.
        """
        print("Processing Gemini Embeddings...")
        gemini_embeds = self.gemini_service.get_embeddings(sentences)
        if not gemini_embeds:
            return None

        result = {
            'embeddings': gemini_embeds,
            'similarity_matrix': self.cosine_similarity_matrix(gemini_embeds).tolist(),
            'dimension': len(gemini_embeds[0])
        }
        
        # Store in Pinecone
        index_name = f"{settings.PINECONE_INDEX_NAME}-gemini"
        self.pinecone_service.create_index_if_not_exists(index_name, self.gemini_dim)
        vectors = []
        for i, embed in enumerate(gemini_embeds):
             vector_id = f"gemini-{uuid.uuid4()}"
             vectors.append((vector_id, embed, {"text": sentences[i]}))
        self.pinecone_service.upsert_vectors(index_name, vectors)
        
        return result

    def search_vectors(self, query: str, embed_type: str, top_k: int = 5):
        """
        Vectorize query and search in Pinecone.
        """
        embed_type = embed_type.lower()
        if embed_type == 'openai':
            print("Generating OpenAI embedding for search query...")
            embeds = self.openai_service.get_embeddings([query])
            if not embeds:
                return None
            query_vector = embeds[0]
            index_name = f"{settings.PINECONE_INDEX_NAME}-openai"
            
        elif embed_type == 'gemini':
            print("Generating Gemini embedding for search query...")
            embeds = self.gemini_service.get_embeddings([query])
            if not embeds:
                return None
            query_vector = embeds[0]
            index_name = f"{settings.PINECONE_INDEX_NAME}-gemini"
            
        else:
            raise ValueError("Invalid embed_type. Must be 'openai' or 'gemini'.")

        # Perform search
        print(f"Searching in index: {index_name}")
        # Ensure index exists (it might not if no data was ever upserted)
        # But we assume data exists for search. If not, Pinecone might error or return empty.
        
        try:
            results = self.pinecone_service.query_vectors(index_name, query_vector, top_k=top_k)
            return results
        except Exception as e:
            print(f"Error searching Pinecone: {e}")
            raise e

    def get_similarity_dataframe(self, sentences, similarity_matrix):
        """
        Helper to create a readable DataFrame for similarity (mostly for local debugging/logs)
        """
        df = pd.DataFrame(similarity_matrix, columns=sentences, index=sentences)
        return df
