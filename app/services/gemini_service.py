from google import genai
from google.genai import types
from app.core.config import settings

class GeminiEmbeddingService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.GEMINI_EMBEDDING_MODEL

    def get_embeddings(self, texts: list[str]):
        """
        Generate embeddings for a list of texts using Gemini (google-genai SDK).
        """
        try:
            # The new SDK supports embedding content directly
            # Reference: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb
            
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    title="Embedding of list of strings" # Title is optional but good for retrieval task
                )
            )
            
            # The result object contains 'embeddings' which is a list of Embedding objects.
            # We need to extract the 'values' from each Embedding object.
            if result.embeddings:
                return [e.values for e in result.embeddings]
            return []
                
        except Exception as e:
            print(f"Error generating Gemini embeddings: {e}")
            return []
