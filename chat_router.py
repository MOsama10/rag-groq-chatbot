import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
# Try to import from simple_rag first, then fall back to rag if needed
try:
    from simple_rag import is_data_question, answer_from_data, get_column_similarities, column_names
    logger = logging.getLogger(__name__)
    logger.info("Using simple_rag module (no sentence-transformers)")
except ImportError:
    from rag import is_data_question, answer_from_data, get_column_similarities, column_names
    logger = logging.getLogger(__name__)
    logger.info("Using full rag module with sentence-transformers")
    
from groq_api import answer_from_groq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message to the chatbot")
    include_debug_info: bool = Field(False, description="Whether to include debug information in the response")

class ChatResponse(BaseModel):
    source: str = Field(..., description="Source of the answer (data, groq, or hybrid)")
    answer: str = Field(..., description="The chatbot's answer")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information (if requested)")

@router.post("/message", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """
    Process a user chat message and return an appropriate response.
    The system will determine if the question is about the loaded data or requires general knowledge.
    """
    query = request.message.strip()
    debug_info = {}
    
    try:
        logger.info(f"Received query: {query}")
        
        # Check if this is a data question
        data_relevant = is_data_question(query)
        logger.info(f"Is data question? {data_relevant}")
        
        if request.include_debug_info:
            # Get column similarities for debugging
            similarities = get_column_similarities(query)
            debug_info["column_similarities"] = [
                {"column": col, "score": float(score)}
                for col, score in similarities[:5]  # Include top 5 for brevity
            ]
            debug_info["is_data_question"] = data_relevant
        
        if data_relevant:
            # Try to answer from data
            data_answer = answer_from_data(query)
            
            if data_answer:
                logger.info("Answering from data")
                if request.include_debug_info:
                    debug_info["answer_source"] = "data"
                
                return ChatResponse(
                    source="data",
                    answer=data_answer,
                    debug_info=debug_info if request.include_debug_info else None
                )
            else:
                # We identified it as a data question but couldn't answer
                # Pass to Groq with data context
                logger.info("Data question but no direct answer, passing to Groq with context")
                if request.include_debug_info:
                    debug_info["answer_source"] = "hybrid"
                
                answer = answer_from_groq(query, is_data_question=True, data_answer=None)
                return ChatResponse(
                    source="hybrid",
                    answer=answer,
                    debug_info=debug_info if request.include_debug_info else None
                )
        
        # Fallback to LLM for general knowledge
        logger.info("Using Groq for general knowledge")
        if request.include_debug_info:
            debug_info["answer_source"] = "groq"
            
        answer = answer_from_groq(query, is_data_question=False)
        return ChatResponse(
            source="groq",
            answer=answer,
            debug_info=debug_info if request.include_debug_info else None
        )
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your request: {str(e)}")