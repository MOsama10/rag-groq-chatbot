"""
A simplified version of the RAG system that doesn't depend on sentence-transformers.
Instead, it uses simple keyword matching and TF-IDF for query matching.
"""

import os
import pandas as pd
import numpy as np
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
DATA_FILE = os.getenv("DATA_FILE", "data.csv")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))  # Lower threshold for simple matching

# Initialize data
df = None
column_names = []
tfidf_vectorizer = None
column_vectors = None
data_loaded = False

def preprocess_text(text):
    """Clean and preprocess text for matching"""
    # Convert to lowercase
    text = text.lower()
    # Replace underscores and hyphens with spaces
    text = re.sub(r'[_-]', ' ', text)
    # Remove other special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

def load_data() -> bool:
    """Load dataset and prepare for matching"""
    global df, column_names, tfidf_vectorizer, column_vectors, data_loaded
    
    try:
        # Load dataset
        df = pd.read_csv(DATA_FILE)
        logger.info(f"Loaded dataset from {DATA_FILE} with {len(df)} rows and {len(df.columns)} columns")
        
        # Prepare column names
        column_names = list(df.columns)
        
        # Prepare column descriptions for better matching
        column_descriptions = [preprocess_text(col) for col in column_names]
        
        # Create TF-IDF vectorizer and fit it on column descriptions
        tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        column_vectors = tfidf_vectorizer.fit_transform(column_descriptions)
        
        data_loaded = True
        return True
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        data_loaded = False
        return False

# Try to load data at module import
load_data()

def get_column_similarities(query: str) -> List[Tuple[str, float]]:
    """Return column names and their similarity scores to the query."""
    if not data_loaded:
        logger.warning("Data not loaded, cannot compute similarities")
        return []
    
    # Preprocess query
    processed_query = preprocess_text(query)
    
    # Convert query to vector using same vectorizer
    query_vector = tfidf_vectorizer.transform([processed_query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, column_vectors)[0]
    
    # Return column names and scores sorted by similarity
    column_scores = [(column_names[i], float(similarities[i])) for i in range(len(column_names))]
    
    # Also do direct keyword matching to boost scores
    for i, col in enumerate(column_names):
        col_lower = col.lower()
        # If column name appears directly in query, boost its score
        if col_lower in processed_query:
            # Apply a boost (add 0.3 to the similarity score)
            column_scores[i] = (column_scores[i][0], min(1.0, column_scores[i][1] + 0.3))
    
    return sorted(column_scores, key=lambda x: x[1], reverse=True)

def is_data_question(query: str, threshold: float = None) -> bool:
    """Determine if the query is about the dataset using column similarity."""
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
        
    if not data_loaded:
        return False
        
    similarities = get_column_similarities(query)
    return any(score >= threshold for _, score in similarities)

def answer_from_data(query: str, threshold: float = None) -> Optional[str]:
    """
    Answer using the dataset if the query matches known column concepts.
    Returns None if data is not available or no relevant columns found.
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
        
    if not data_loaded or df.empty:
        return "I don't have access to the data at the moment."

    similarities = get_column_similarities(query)
    relevant_columns = [(col, score) for col, score in similarities if score >= threshold]
    
    if not relevant_columns:
        return None

    # Get statistics about the relevant columns
    result_parts = []
    for col_name, score in relevant_columns:
        try:
            if df[col_name].dtype in ['int64', 'float64']:
                # For numeric columns, show statistics
                stats = {
                    "latest": df[col_name].iloc[-1],
                    "mean": df[col_name].mean(),
                    "min": df[col_name].min(),
                    "max": df[col_name].max()
                }
                result_parts.append(f"{col_name} (match: {score:.2f}):\n"
                                    f"  - Latest value: {stats['latest']}\n"
                                    f"  - Average: {stats['mean']:.2f}\n"
                                    f"  - Range: {stats['min']} to {stats['max']}")
            else:
                # For categorical columns, show latest value and top values
                latest = df[col_name].iloc[-1]
                value_counts = df[col_name].value_counts().head(3)
                top_values = ", ".join([f"{val} ({count})" for val, count in value_counts.items()])
                
                result_parts.append(f"{col_name} (match: {score:.2f}):\n"
                                    f"  - Latest value: {latest}\n"
                                    f"  - Top values: {top_values}")
        except Exception as e:
            logger.error(f"Error processing column {col_name}: {str(e)}")
            result_parts.append(f"{col_name}: Error processing this column")

    # Add context about what columns were matched
    matched_cols = ", ".join([col for col, _ in relevant_columns])
    intro = f"Based on your question, I found relevant information in these columns: {matched_cols}\n\n"
    
    return intro + "\n\n".join(result_parts)

def get_data_summary() -> str:
    """Return a summary of the loaded dataset"""
    if not data_loaded or df.empty:
        return "No data is currently loaded."
    
    return (f"Dataset summary:\n"
            f"- {len(df)} records\n"
            f"- {len(df.columns)} columns: {', '.join(column_names)}\n"
            f"- Date range: {df.index.min()} to {df.index.max()} (if applicable)")

# Make sure these are accessible to other modules
__all__ = ['is_data_question', 'answer_from_data', 'get_data_summary', 
           'column_names', 'load_data', 'get_column_similarities']