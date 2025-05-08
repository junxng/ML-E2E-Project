from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

from app.rag.qa import get_rag_chain, query_documents
from app.rag.documents import get_knowledge_base, remove_from_knowledge_base

# Get logger but don't configure it (main module handles configuration)
logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []

class KnowledgeBaseEntry(BaseModel):
    id: int
    file_name: str
    file_path: str
    file_size_bytes: int
    date_added: str

@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question to the RAG system
    """
    logger = logging.getLogger(__name__)
    
    if not request.query or len(request.query.strip()) < 3:
        return QueryResponse(
            answer="Please provide a more specific question.",
            sources=[]
        )
    
    try:
        # Get answer from RAG system with timeout
        result = query_documents(request.query)
        
        # Validate that we got a proper response
        if not result or not isinstance(result, dict) or "answer" not in result:
            logger.error(f"Invalid response from query_documents: {result}")
            raise ValueError("Invalid response structure from RAG system")
            
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", [])
        )
    except Exception as e:
        logger.exception(f"Error processing query '{request.query}': {str(e)}")
        
        # Return a user-friendly error response instead of raising an exception
        return QueryResponse(
            answer="I'm sorry, I encountered an error while processing your question. Please try again with a different question or contact support if the issue persists.",
            sources=[]
        )

@router.get("/knowledge-base", response_model=List[KnowledgeBaseEntry])
async def get_knowledge_base_entries():
    """
    Get all entries in the knowledge base
    """
    try:
        kb = get_knowledge_base()
        return kb
    except Exception as e:
        logger.exception(f"Error retrieving knowledge base: {str(e)}")
        # Return empty list instead of raising an exception
        return []

@router.delete("/knowledge-base/{file_id}", response_model=Dict[str, bool])
async def delete_knowledge_base_entry(file_id: int):
    """
    Remove a file from the knowledge base
    """
    try:
        success = remove_from_knowledge_base(file_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting knowledge base entry: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
