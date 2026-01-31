"""
Data Preprocessing Module
=========================
Handles text cleaning, tokenization, and preprocessing for the intent classification model.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
def download_nltk_data():
    """Download required NLTK data packages."""
    required_packages = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {package}: {e}")


def clean_text(text: str) -> str:
    """
    Clean raw text by removing special characters and normalizing.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters, keep only letters, numbers, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Preprocess text for model input.
    
    Args:
        text: Raw input text
        remove_stopwords: Whether to remove stopwords (default: False)
        
    Returns:
        Preprocessed text ready for model
    """
    # Clean the text first
    text = clean_text(text)
    
    if not text:
        return ""
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except LookupError:
        download_nltk_data()
        tokens = word_tokenize(text)
    
    # Optionally remove stopwords
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if w not in stop_words]
        except LookupError:
            download_nltk_data()
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)


def load_and_preprocess_data(file_path: str, query_column: str = 'query', 
                              intent_column: str = 'intent') -> pd.DataFrame:
    """
    Load dataset and preprocess all queries.
    
    Args:
        file_path: Path to CSV file
        query_column: Name of the query column
        intent_column: Name of the intent column
        
    Returns:
        DataFrame with preprocessed queries
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Validate columns
    if query_column not in df.columns:
        raise ValueError(f"Column '{query_column}' not found in dataset")
    if intent_column not in df.columns:
        raise ValueError(f"Column '{intent_column}' not found in dataset")
    
    # Preprocess queries
    print(f"Preprocessing {len(df)} queries...")
    df['processed_query'] = df[query_column].apply(preprocess_text)
    
    # Remove empty queries
    initial_count = len(df)
    df = df[df['processed_query'].str.len() > 0]
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} empty queries after preprocessing")
    
    print(f"Preprocessing complete. {len(df)} queries ready for training.")
    
    return df


def get_data_statistics(df: pd.DataFrame, intent_column: str = 'intent') -> dict:
    """
    Get statistics about the dataset.
    
    Args:
        df: DataFrame with preprocessed data
        intent_column: Name of the intent column
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_samples': len(df),
        'num_intents': df[intent_column].nunique(),
        'intent_distribution': df[intent_column].value_counts().to_dict(),
        'avg_query_length': df['processed_query'].str.split().str.len().mean(),
        'min_query_length': df['processed_query'].str.split().str.len().min(),
        'max_query_length': df['processed_query'].str.split().str.len().max(),
    }
    
    return stats


if __name__ == "__main__":
    # Test preprocessing
    download_nltk_data()
    
    test_queries = [
        "I want to cancel my order!!!",
        "Where is my package?",
        "PAYMENT didn't work :(",
        "Can I return this item please?",
        "What's the status of order #12345?"
    ]
    
    print("Testing preprocessing:")
    print("-" * 50)
    
    for query in test_queries:
        processed = preprocess_text(query)
        print(f"Original: {query}")
        print(f"Processed: {processed}")
        print()
