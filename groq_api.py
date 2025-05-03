import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Try simple_rag first, then fall back to rag
try:
    from simple_rag import column_names, get_data_summary
    logger = logging.getLogger(__name__)
    logger.info("Using simple_rag for column names in groq_api")
except ImportError:
    from rag import column_names, get_data_summary
    logger = logging.getLogger(__name__)
    logger.info("Using full rag for column names in groq_api")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama2-70b-4096")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Define system prompt with dataset context
SYSTEM_PROMPT = """
You are a helpful AI assistant integrated with a custom data analysis system.

You have two modes of operation:
1. DATA MODE: When users ask about specific data points, metrics, or analysis related to the user activity data you have access to.
2. GENERAL MODE: When answering general knowledge questions unrelated to the data.

The system has determined that the user's question {is_data_related}.

{data_summary}

When answering:
- Be concise but informative
- If data is referenced, explain what the metrics mean
- If you don't know something, say so clearly
- Format numeric data nicely when presenting statistics

Reply in a helpful, conversational tone.
"""

def answer_from_groq(user_question: str, is_data_question: bool = False, data_answer: Optional[str] = None) -> str:
    """
    Get an answer from the Groq API.
    
    Args:
        user_question: The user's question
        is_data_question: Whether this was identified as a data question
        data_answer: The answer from the data system, if available
    
    Returns:
        The formatted response from Groq
    """
    if not GROQ_API_KEY:
        logger.error("GROQ API key is missing. Please check your .env file.")
        return "I'm not properly configured. The API key is missing. Please check the .env file."

    # Determine the context to provide to the model
    data_context = ""
    data_related_text = "is related to your data" if is_data_question else "is NOT related to your data"
    
    # Include data summary if available
    data_summary = f"Data columns available: {', '.join(column_names)}" if column_names else "No data is currently loaded."
    
    # If we have a data answer, include it
    if data_answer:
        data_context = f"Here is the information retrieved from your data:\n\n{data_answer}\n\nPlease incorporate this into your answer."
    
    # Format system prompt
    system_content = SYSTEM_PROMPT.format(
        is_data_related=data_related_text,
        data_summary=data_summary
    )
    
    # Build the query prompt
    query_prompt = (
        f"USER QUESTION: {user_question}\n\n"
        f"{data_context}\n\n"
        "Please provide a helpful response based on the information above."
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        logger.info(f"Sending request to Groq API using model {GROQ_MODEL}")
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        logger.info("Successfully received response from Groq API")
        
        return answer

    except requests.exceptions.RequestException as e:
        logger.error(f"Groq API Request Error: {str(e)}")
        if 'response' in locals() and response is not None:
            logger.error(f"Response Status: {response.status_code}")
            logger.error(f"Response Body: {response.text}")
        
        return "I'm having trouble connecting to my knowledge source right now. Please try again in a moment."
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return "An unexpected error occurred while processing your question."