from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.comparison_service import ComparisonService
from enum import Enum

router = APIRouter()
comparison_service = ComparisonService()

class EmbedType(str, Enum):
    openai = "openai"
    gemini = "gemini"

class SearchRequest(BaseModel):
    text: str
    embed_type: EmbedType

class EmbeddingRequest(BaseModel):
    sentences: list[str]

@router.post("/compare")
async def search_vectors(request: SearchRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Search text cannot be empty")
    
    try:
        results = comparison_service.search_vectors(request.text, request.embed_type)
        if results is None:
             raise HTTPException(status_code=500, detail=f"Failed to generate embedding or search for {request.embed_type}")

        return {
            "query": request.text,
            "embed_type": request.embed_type,
            "results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings/openai")
async def openai_embeddings(request: EmbeddingRequest):
    if not request.sentences:
        raise HTTPException(status_code=400, detail="List of sentences cannot be empty")
    
    try:
        results = comparison_service.process_openai_embeddings(request.sentences)
        if not results:
             raise HTTPException(status_code=500, detail="Failed to generate OpenAI embeddings")

        response = {
            "model": "openai",
            "sentences": request.sentences,
            "dimension": results.get('dimension'),
            "similarity_matrix": results.get('similarity_matrix')
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings/gemini")
async def gemini_embeddings(request: EmbeddingRequest):
    if not request.sentences:
        raise HTTPException(status_code=400, detail="List of sentences cannot be empty")
    
    try:
        results = comparison_service.process_gemini_embeddings(request.sentences)
        if not results:
             raise HTTPException(status_code=500, detail="Failed to generate Gemini embeddings")
        
        response = {
            "model": "gemini",
            "sentences": request.sentences,
            "dimension": results.get('dimension'),
            "similarity_matrix": results.get('similarity_matrix')
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
