import openai
from openai import OpenAI
import logging
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, MAX_TOKENS, TEMPERATURE

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text):
    """
    Get embeddings for a given text using OpenAI's embedding model.
    """
    try:
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        raise

def get_completion(prompt, system_prompt=None):
    """
    Get completion from OpenAI's GPT-4 model.
    """
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error getting completion: {e}")
        raise

def batch_get_embeddings(texts):
    """
    Get embeddings for a batch of texts.
    """
    try:
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        logging.error(f"Error getting batch embeddings: {e}")
        raise

# Function to analyze control compliance
def analyze_control_compliance(control_text, context):
    """
    Analyze control compliance using GPT-4.
    """
    system_prompt = """You are an expert in SOC 2 compliance analysis. Your task is to analyze control descriptions 
    and determine their compliance status based on the provided context. Be thorough and precise in your analysis."""
    
    prompt = f"""
    Based on the following control and context, analyze the compliance status:
    
    Control: {control_text}
    
    Context: {context}
    
    Please provide:
    1. Compliance Score (0-100)
    2. Detailed Analysis
    3. Control Status (Fully Met/Partially Met/Not Met)
    """
    
    response = get_completion(prompt, system_prompt)
    return response

# Function to analyze qualifiers
def analyze_qualifier(question, context):
    """
    Analyze SOC report qualifiers using GPT-4.
    """
    system_prompt = """You are an expert in SOC 2 report analysis. Your task is to analyze the report's qualifiers 
    and provide accurate assessments based on the provided context."""
    
    prompt = f"""
    Based on the following question and context from the SOC report, provide a detailed analysis:
    
    Question: {question}
    
    Context: {context}
    
    Please provide a clear Yes/No answer followed by a detailed explanation.
    """
    
    response = get_completion(prompt, system_prompt)
    return response 