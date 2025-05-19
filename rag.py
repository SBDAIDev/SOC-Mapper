# rag.py

import os
import logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import re
from llm_utils import batch_get_embeddings, analyze_control_compliance

# If parser.py is in the same folder, make sure to reference it correctly
from parser import (
    extract_text_from_pdf,
    chunk_text_by_multiple_patterns,
    generate_regex_from_sample,
    concatenate_domain_control
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_embeddings(df_chunks, model_name='all-mpnet-base-v2'):
    logger.info("Generating embeddings for text chunks.")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df_chunks["Content"].tolist(), show_progress_bar=True)
    df_chunks["embedding"] = embeddings.tolist()
    logger.info("Embeddings generated.")
    return df_chunks, model

def create_faiss_index(df_chunks):
    logger.info("Creating FAISS index.")
    embedding_matrix = np.vstack(df_chunks["embedding"].values).astype('float32')
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    logger.info("FAISS index created and embeddings added.")
    return index, embedding_matrix

def save_faiss_index(index, index_path):
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index saved to '{index_path}'.")

def load_faiss_index(index_path):
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    logger.info("FAISS index loaded successfully.")
    return index

def save_chunks_dataframe(df_chunks, df_chunks_path):
    logger.info(f"Saving chunks DataFrame to {df_chunks_path}")
    df_chunks_copy = df_chunks.copy()
    # Convert embedding arrays to comma-separated strings
    df_chunks_copy['embedding'] = df_chunks_copy['embedding'].apply(
        lambda x: ",".join(map(str, x))
    )
    df_chunks_copy.to_csv(df_chunks_path, index=False)
    logger.info(f"Chunks DataFrame saved to '{df_chunks_path}'.")

def load_chunks_dataframe(df_chunks_path):
    logger.info(f"Loading chunks DataFrame from {df_chunks_path}")
    df_chunks = pd.read_csv(df_chunks_path)
    # Convert comma-separated embedding strings back to float arrays
    df_chunks['embedding'] = df_chunks['embedding'].apply(
        lambda x: np.array([float(i) for i in x.split(",")])
    )
    logger.info("Chunks DataFrame loaded successfully.")
    return df_chunks

def embed_query(query, model):
    logger.debug(f"Embedding query: {query}")
    query_embedding = model.encode([query], show_progress_bar=False)
    return query_embedding.astype('float32')

def search_faiss(index, query_embedding, top_k=5):
    logger.debug(f"Searching FAISS index for top {top_k} results.")
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

def retrieve_answers(df_chunks, distances, indices):
    """
    Grab the matching chunks from df_chunks using the
    indices from FAISS, plus add a 'distance' column.
    """
    results = df_chunks.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    logger.debug("Retrieved answers from FAISS index search.")
    return results

def chunk_text_without_patterns(text, chunk_size=1000):
    """
    Splits the text into fixed-size chunks without using any patterns.
    """
    logger.info("Chunking text without patterns.")
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def build_rag_system_with_parser(pdf_path, start_page, end_page, control_patterns,
                               output_text_path, df_chunks_path, faiss_index_path, chunk_size):
    """
    Build a RAG system using OpenAI embeddings.
    """
    try:
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page, output_text_path)
        
        # Create chunks based on control patterns
        if control_patterns:
            df_chunks = chunk_text_by_multiple_patterns(extracted_text, control_patterns)
        else:
            # If no control patterns provided, chunk by size
            chunks = [extracted_text[i:i+chunk_size] for i in range(0, len(extracted_text), chunk_size)]
            df_chunks = pd.DataFrame({"Content": chunks})
        
        # Save chunks to CSV
        df_chunks.to_csv(df_chunks_path, index=False)
        
        # Get embeddings for all chunks
        texts = df_chunks["Content"].tolist()
        embeddings = np.array(batch_get_embeddings(texts)).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save FAISS index
        faiss.write_index(index, faiss_index_path)
        
        logging.info(f"RAG system built successfully. Index saved to {faiss_index_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error building RAG system: {e}", exc_info=True)
        raise

def retrieve_answers_for_controls(
    df,
    model,
    index,
    df_chunks,
    top_k=3
):
    """
    For each control in df['Control'], retrieve the top_k matching chunks.
    Store them as Answer_1, Answer_2, ..., plus their corresponding Control ID.
    
    NOTE: We have removed the 'Page' field references to rely purely on
    'Control ID' and 'Content'.
    """
    logger.info("Retrieving answers for each control.")
    # Prepare the columns where answers will go
    for i in range(1, top_k + 1):
        df[f'Answer_{i}'] = None
        df[f'Answer_{i}_Control_ID'] = None

    # Iterate through each control in the DataFrame
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Controls"):
        query = row['Control']
        logger.debug(f"Processing Control: {query}")

        # 1) Embed the query
        query_emb = embed_query(query, model)

        # 2) Search the FAISS index
        distances, indices_search = search_faiss(index, query_emb, top_k)

        # 3) Retrieve results
        retrieved = retrieve_answers(df_chunks, distances, indices_search)
        
        # 4) Write the retrieved results to the DataFrame
        if retrieved.empty:
            logger.warning(f"No chunks retrieved for Control: {query}")
            continue
        
        for i in range(top_k):
            if i < len(retrieved):
                answer = retrieved.iloc[i]['Content']
                control_id = retrieved.iloc[i].get('Control ID', 'N/A')
                
                df.at[idx, f'Answer_{i+1}'] = answer
                df.at[idx, f'Answer_{i+1}_Control_ID'] = control_id

                logger.debug(
                    f"Retrieved Answer_{i+1} for Control '{query}': "
                    f"{answer[:50]}... Control_ID: {control_id}"
                )
            else:
                logger.debug(
                    f"No more answers available for Control '{query}' "
                    f"beyond Answer_{i+1}."
                )
    logger.info("Answers retrieved for all controls.")
    return df

def process_cybersecurity_framework_with_rag(excel_input_path, output_path, faiss_index_path,
                                           df_chunks_path, top_k=3):
    """
    Process cybersecurity framework using RAG system with OpenAI embeddings.
    """
    try:
        # Load the framework Excel file
        df = pd.read_excel(excel_input_path)
        
        # Load chunks and FAISS index
        df_chunks = pd.read_csv(df_chunks_path)
        index = faiss.read_index(faiss_index_path)
        
        results = []
        for _, row in df.iterrows():
            # Get control text
            control_text = str(row['User Org Control Statement'])
            
            # Get embedding for the control
            control_embedding = np.array([batch_get_embeddings([control_text])[0]]).astype('float32')
            
            # Search similar contexts
            D, I = index.search(control_embedding, top_k)
            
            # Get relevant chunks
            relevant_chunks = df_chunks.iloc[I[0]]['Content'].tolist()
            context = "\n".join(relevant_chunks)
            
            # Analyze compliance using GPT-4
            analysis = analyze_control_compliance(control_text, context)
            
            # Store results
            results.append({
                'Control': control_text,
                'Analysis': analysis,
                'Context': context
            })
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Save to Excel
        output_df.to_excel(output_path, index=False)
        logging.info(f"Framework processed successfully. Results saved to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing framework: {e}", exc_info=True)
        raise

def save_updated_framework(df, output_path):
    """
    Save the final DataFrame (with answers) to Excel or CSV, 
    sanitizing non-ASCII characters if needed.
    """
    logger.info(f"Saving updated framework to {output_path}.")
    
    def sanitize_text(text):
        if isinstance(text, str):
            # Remove control characters, etc.
            return re.sub(r'[^ -\x7E]', ' ', text)
        return text

    df = df.applymap(sanitize_text)

    if output_path.lower().endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    logger.info(f"Updated DataFrame saved to '{output_path}'.")
