# qualifiers.py

import os
import logging
import pandas as pd
from rag import retrieve_answers_for_controls, load_faiss_index
from llm_analysis import call_ollama_api
from sentence_transformers import SentenceTransformer
import datetime
import re
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.styles import Font, Alignment, PatternFill
import numpy as np
import faiss
from llm_utils import batch_get_embeddings, analyze_qualifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def search_context(query, df, model, index, top_k=3):
    """
    Search for relevant context using OpenAI embeddings.
    """
    query_embedding = np.array([batch_get_embeddings([query])[0]]).astype('float32')
    D, I = index.search(query_embedding, top_k)
    relevant_chunks = df.iloc[I[0]]['Content'].tolist()
    return "\n".join(relevant_chunks)


def is_report_latest(df, model, index, top_k=3):
    query = "What is the report date and audit period? Is this report less than 12 months old?"
    context = search_context(query, df, model, index, top_k)
    return analyze_qualifier(query, context)


def are_trust_principles_covered(df, model, index, top_k=3):
    query = "What trust principles (Security, Availability, Confidentiality) are covered in this SOC 2 Type 2 report?"
    context = search_context(query, df, model, index, top_k)
    return analyze_qualifier(query, context)


def is_audit_period_sufficient(df, model, index, top_k=3):
    query = "What is the audit period duration? Is it at least 9 months?"
    context = search_context(query, df, model, index, top_k)
    return analyze_qualifier(query, context)


def has_invalid_observations(df, model, index, top_k=3):
    query = "Are there any significant observations or issues in the independent auditor's opinion that would invalidate this report?"
    context = search_context(query, df, model, index, top_k)
    return analyze_qualifier(query, context)


def is_report_qualified(df, model, index, top_k=3):
    query = "Is this a qualified report? Are there any qualifications in the auditor's opinion?"
    context = search_context(query, df, model, index, top_k)
    return analyze_qualifier(query, context)


def describe_scope_of_services(df, model, index, top_k=3):
    query = "What is the scope of services covered in this SOC 2 Type 2 report? Please describe the services in detail."
    context = search_context(query, df, model, index, top_k)
    return analyze_qualifier(query, context)


def qualify_soc_report(pdf_path, df_chunks_path, faiss_index_path, excel_output_path):
    """
    Perform all qualifier checks and save results to Excel.
    """
    try:
        # Load chunks and index
        df_chunks = pd.read_csv(df_chunks_path)
        index = faiss.read_index(faiss_index_path)
        
        # Perform all qualifier checks
        latest_result = is_report_latest(df_chunks, None, index)
        trust_principles_result = are_trust_principles_covered(df_chunks, None, index)
        audit_period_result = is_audit_period_sufficient(df_chunks, None, index)
        invalid_obs_result = has_invalid_observations(df_chunks, None, index)
        qualified_result = is_report_qualified(df_chunks, None, index)
        scope_result = describe_scope_of_services(df_chunks, None, index)
        
        # Create results DataFrame
        results = [
            ["Is the SOC 2 Type 2 Report latest?", latest_result],
            ["Are all Trust Principles covered?", trust_principles_result],
            ["Is the audit period sufficient?", audit_period_result],
            ["Are there invalid observations?", invalid_obs_result],
            ["Is the report qualified?", qualified_result],
            ["Scope of Services", scope_result]
        ]
        
        df_results = pd.DataFrame(results, columns=["Question", "Answer"])
        
        # Load existing Excel file
        with pd.ExcelWriter(excel_output_path, engine='openpyxl', mode='a') as writer:
            df_results.to_excel(writer, sheet_name='Qualifying Questions', index=False)
        
        logging.info("Qualifier checks completed and saved to Excel.")
        return True
        
    except Exception as e:
        logging.error(f"Error in qualify_soc_report: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Example usage (debug/test):
    pdf_path = "path/to/soc2_report.pdf"
    df_chunks_path = "path/to/df_qualifier_chunks.csv"
    faiss_index_path = "path/to/faiss_index_qualifiers.idx"
    excel_output_path = "path/to/output.xlsx"

    qualify_soc_report(pdf_path, df_chunks_path, faiss_index_path, excel_output_path)
