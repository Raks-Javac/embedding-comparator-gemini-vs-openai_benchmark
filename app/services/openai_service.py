import openai
from app.core.config import settings

class OpenAIEmbeddingService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_EMBEDDING_MODEL

    def get_embeddings(self, texts: list[str]):
        """
        Generate embeddings for a list of texts using OpenAI.
        """
        try:
            res = self.client.embeddings.create(input=texts, model=self.model)
            return [r.embedding for r in res.data]
        except Exception as e:
            print(f"Error generating OpenAI embeddings: {e}")
            return []
