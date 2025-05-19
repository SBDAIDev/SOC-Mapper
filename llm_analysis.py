# llm_analysis.py

import os
import re
import uuid
import json
import time
import logging
import requests
import pandas as pd
from functools import wraps
from tqdm import tqdm
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import column_index_from_string
from llm_utils import analyze_control_compliance

# Configure logging with enhanced setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def retry(exceptions, tries=3, delay=2, backoff=2):
    """
    Decorator for retrying a function call with exponential backoff.
    """
    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"{e}, Retrying in {_delay} seconds...")
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            return func(*args, **kwargs)
        return wrapper_retry
    return decorator_retry

# Configuration for the Ollama API
OLLAMA_API_URL = "http://localhost:11434"   # Update if different
PHI4_MODEL_NAME = "phi4"  # Fixed model now set to phi4 only
MAX_TOKENS = 1024       # Increased from 512 to 1024

@retry((requests.exceptions.RequestException, json.JSONDecodeError), tries=3, delay=2, backoff=2)
def call_ollama_api(prompt, model=PHI4_MODEL_NAME, max_tokens=MAX_TOKENS):
    """
    Calls the Ollama API with the given prompt and returns the generated text.
    """
    logger.info("Calling Ollama API for LLM analysis.")
    url = f"{OLLAMA_API_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    session_id = str(uuid.uuid4())
    payload = {
        "model": model,
        "prompt": prompt,
        "session_id": session_id,
        "num_ctx": 2048,
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()  # Raise an exception for HTTP errors
        logger.info(f"Ollama API response status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        raise

    try:
        generated_text = ''
        for line in response.iter_lines():
            if line:
                line_decoded = line.decode('utf-8').strip()
                try:
                    line_json = json.loads(line_decoded)
                    token = line_json.get('response', '')
                    generated_text += token
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line_decoded}")
        logger.info("Ollama API call succeeded.")
        return generated_text.strip()
    except Exception as e:
        logger.error(f"Unexpected error during API response processing: {e}")
        raise

def load_responses(excel_path):
    """
    Load responses from the processed framework Excel file.
    """
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        logging.error(f"Error loading responses from {excel_path}: {e}")
        raise

def generate_analysis_prompt(control_description, answers):
    """
    Generates a prompt for the LLM to analyze control compliance based on retrieved responses,
    with strict instructions to avoid markdown, extra characters, or additional commentary.
    """
    logger.debug("Generating analysis prompt for LLM.")
    prompt = f"""You are an expert in cybersecurity compliance and controls. Your task is to analyze whether specific controls are met based on the provided responses from a document. Provide your responses in plain text only, without any markdown or formatting characters.
    
Control Description:
{control_description}

Retrieved Responses:"""
    for idx, answer in enumerate(answers, 1):
        if pd.notna(answer['text']) and answer['text']:
            prompt += f"\n{idx}. {answer['text']} (Control ID: {answer['control_id']})"

    # Strict instructions and examples
    prompt += """
    
Instructions:
1. Assess Each Response: For each retrieved response, determine the level of compliance (Fully Met, Partially Met, or Not Met).
2. Format the Output: For each response, provide an analysis in the exact format:
   1. Per review of [Retrieved Response] (Control ID: [Control_ID]), [Control Description] is Fully Met/Partially Met/Not Met. [Reasoning]

3. Final Conclusion: After analyzing all responses, provide a single-sentence conclusion starting with:
   Overall, the control "[Control Description]" is [Fully Met/Partially Met/Not Met].

Important:
- Do not include any markdown characters like asterisks (**), underscores, or backticks.
- Do not provide any additional commentary or formatting beyond the lines and single final conclusion sentence.
- If multiple responses are "Fully Met" and some are "Not Met", the overall rating may be Partially Met or Fully Met depending on the significance of the 'Not Met' gap. Use your judgment, but the final line must strictly begin with "Overall, the control ...".

Example:

Control Description:
(IDAM-01): Access to sensitive data is restricted to authorized personnel only.

Retrieved Responses:
1. The organization has implemented role-based access controls, ensuring only authorized employees can access sensitive data. (Control ID: SC-01)
2. There is no record of recent access reviews or audits. (Control ID: SC-02)

Output:
1. Per review of The organization has implemented role-based access controls, ensuring only authorized employees can access sensitive data. (Control ID: SC-01), (IDAM-01): Access to sensitive data is restricted to authorized personnel only. is Fully Met. Reasoning: Access controls are effectively implemented to restrict data access.
2. Per review of There is no record of recent access reviews or audits. (Control ID: SC-02), (IDAM-01): Access to sensitive data is restricted to authorized personnel only. is Not Met. Reasoning: Lack of recent access reviews indicates potential gaps in data security.

Follow this prompt strictly and without fail.
---
"""

    return prompt

def generate_final_conclusion_prompt(control_description, llama_analysis):
    """
    Generates a prompt for the LLM to provide a final conclusion based on the existing analysis.
    """
    prompt = f"""You are an expert in cybersecurity compliance and controls. Based on the following analysis, provide a conclusive response on whether the control is Fully Met, Partially Met, or Not Met. Respond in plain text without any markdown or formatting.

Control Description:
{control_description}

Existing Analysis:
{llama_analysis}

Instructions:
- Analyze the existing analysis based on the following rules:
  1. Fully Met: If at least one of the per-response analyses is "Fully Met", the overall conclusion should generally be "Fully Met" unless there's a critical "Not Met".
  2. Partially Met: If there are no "Fully Met" responses but one or more "Partially Met" responses, decide if they collectively fulfill the control requirements.
  3. Not Met: If no responses indicate coverage of the control or are insufficient.

- Provide a single sentence conclusion starting with "Final Conclusion:" followed by the status.
- Do not include any additional information or reasoning.

Example:
Final Conclusion: Partially Met.
"""
    return prompt

def clean_llm_output(response, control_description):
    """
    Cleans and formats the LLM output to ensure consistency.

    Parameters:
    - response (str): The raw response from the LLM.
    - control_description (str): The description of the control being analyzed.

    Returns:
    - cleaned_response (str): The formatted and cleaned response (per-response analyses).
    - individual_statuses (list): List of individual assessment statuses (Fully Met, Partially Met, Not Met).
    - final_conclusion (str): The final conclusion extracted from the LLM response.
    """
    cleaned_lines = []
    individual_statuses = []
    final_conclusion = ''

    # Split the response into lines
    lines = response.strip().split('\n')

    logger.debug(f"Raw LLM Response:\n{response}")

    for line in lines:
        # Remove leading/trailing asterisks or other unwanted characters
        line = line.strip('*').strip()
        if not line:
            continue

        # Regex to match per-response analysis (more flexible)
        per_response_match = re.match(
            r"^\d+\.\s+Per review of\s+(.+?)\s+\(Control ID:\s*([A-Za-z0-9._\- ]+)\),\s*(.+?)\s+is\s+(Fully Met|Partially Met|Not Met)\.\s*(.*)",
            line,
            re.IGNORECASE
        )
        if per_response_match:
            retrieved_response = per_response_match.group(1).strip()
            control_id = per_response_match.group(2).strip()
            control_desc = per_response_match.group(3).strip()
            status = per_response_match.group(4).strip()
            reasoning = per_response_match.group(5).strip()

            # Remove duplicate "Reasoning:" if present
            if reasoning.lower().startswith("reasoning:"):
                reasoning = reasoning[len("reasoning:"):].strip()

            # Reconstruct the line to ensure consistency without formatting
            formatted_line = (
                f"Per review of {control_id} {retrieved_response}; {control_desc} is {status}.\n"
                f"Reasoning: {reasoning}"
            )
            cleaned_lines.append(formatted_line)
            individual_statuses.append(status)
            continue

        # Attempt to extract the final conclusion
        final_match = re.match(
            r"(?:Overall,\s?the\s?control\s?)\"?.+?\"?\s?(?:is)\s?(Fully Met|Partially Met|Not Met)",
            line,
            re.IGNORECASE
        )
        if final_match:
            final_conclusion = final_match.group(1).strip()
            logger.info(f"Extracted Final Conclusion: {final_conclusion}")
            continue

        # If we see "Final Conclusion:" explicitly, try to parse status from that line
        if "Final Conclusion:" in line:
            conclusion_search = re.search(
                r"Final Conclusion:\s*(Fully Met|Partially Met|Not Met)[\.\s]*",
                line,
                re.IGNORECASE
            )
            if conclusion_search:
                final_conclusion = conclusion_search.group(1).strip()
                logger.info(f"Extracted Final Conclusion from label: {final_conclusion}")
            else:
                logger.warning(f"Could not parse final conclusion from line: {line}")
            continue

        logger.warning(f"Unexpected response format: {line}")

    cleaned_response = '\n\n'.join(cleaned_lines)
    cleaned_response = remove_duplicate_control_ids(cleaned_response)

    return cleaned_response, individual_statuses, final_conclusion

def remove_duplicate_control_ids(detailed_analysis):
    """
    Removes duplicate Control IDs in the same line of the Detailed Analysis Explanation.
    """
    lines = detailed_analysis.split('\n\n')
    cleaned_lines = []

    for line in lines:
        cleaned_line = re.sub(r"(\b[A-Za-z0-9.-]+\b)\s+\1", r"\1", line)
        cleaned_lines.append(cleaned_line)

    cleaned_analysis = '\n\n'.join(cleaned_lines)
    return cleaned_analysis

def determine_final_conclusion(assessments):
    """
    Determines the final conclusion based on individual assessments.
    """
    counts = {
        'Fully Met': assessments.count('Fully Met'),
        'Partially Met': assessments.count('Partially Met'),
        'Not Met': assessments.count('Not Met'),
        'Error': assessments.count('Error')
    }

    if counts['Error'] > 0:
        return 'Error in Analysis'

    if counts['Fully Met'] > 0 and (counts['Not Met'] == 0):
        return 'Fully Met'
    elif counts['Fully Met'] > 0 and counts['Not Met'] > 0:
        return 'Partially Met'
    elif counts['Partially Met'] > 0:
        return 'Partially Met'
    else:
        return 'Not Met'

def process_controls(df):
    """
    Process each control using OpenAI's GPT-4 model.
    """
    try:
        results = []
        for _, row in df.iterrows():
            control = row['Control']
            context = row['Context']
            
            # Get analysis from GPT-4
            analysis = analyze_control_compliance(control, context)
            
            # Parse the analysis to extract score and status
            score = None
            status = None
            detailed_analysis = analysis
            
            # Simple parsing (you might want to make this more robust)
            lines = analysis.split('\n')
            for line in lines:
                if line.startswith('1.') and 'Score' in line:
                    try:
                        score = int(line.split(':')[-1].strip().rstrip('%'))
                    except:
                        score = 0
                elif line.startswith('3.') and 'Status' in line:
                    status = line.split(':')[-1].strip()
            
            results.append({
                'Control': control,
                'Compliance Score': score,
                'Control Status': status,
                'Detailed Analysis': detailed_analysis
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        logging.error(f"Error processing controls: {e}")
        raise

def process_final_conclusions(df, max_retries=1, socketio=None):
    """
    Processes each control's Llama Analysis to generate a final conclusion using the LLM.
    """
    logger.info("Processing final conclusions with LLM.")

    if 'Final Conclusion' not in df.columns:
        df['Final Conclusion'] = ''

    total_controls = df.shape[0]
    for idx, row in tqdm(df.iterrows(), total=total_controls, desc="Finalizing Control Status"):
        control_description = row.get('User Org Control Statement', '')
        llama_analysis = row.get('Detailed Analysis Explanation', '')

        if pd.isna(llama_analysis) or not str(llama_analysis).strip():
            logger.warning(f"No Llama Analysis found for control at index {idx}. Skipping final conclusion.")
            df.at[idx, 'Final Conclusion'] = 'Error in Analysis'
            df.at[idx, 'Control Status'] = 'Error in Analysis'
            if socketio:
                progress = ((idx + 1) / total_controls) * 100
                socketio.emit('progress', {'progress': progress}, broadcast=True)
            continue

        prompt = generate_final_conclusion_prompt(control_description, llama_analysis)
        retries = 0
        while retries <= max_retries:
            try:
                response = call_ollama_api(prompt)
                final_conclusion_match = re.match(r"Final Conclusion:\s*(Fully Met|Partially Met|Not Met)\.?$", response, re.IGNORECASE)
                if final_conclusion_match:
                    final_status = final_conclusion_match.group(1).strip()
                    df.at[idx, 'Final Conclusion'] = final_status
                    df.at[idx, 'Control Status'] = final_status
                    logger.info(f"Final Control Status for index {idx}: {final_status}")
                else:
                    logger.warning(f"Unexpected final conclusion format for control at index {idx}: {response}")
                    df.at[idx, 'Final Conclusion'] = 'Error in Analysis'
                    df.at[idx, 'Control Status'] = 'Error in Analysis'

                if socketio:
                    progress = ((idx + 1) / total_controls) * 100
                    socketio.emit('progress', {'progress': progress}, broadcast=True)
                break

            except Exception as e:
                logger.error(f"Error generating final conclusion for control at index {idx}: {e}")
                retries += 1
                if retries > max_retries:
                    logger.error(f"Max retries exceeded for final conclusion at index {idx}. Marking as 'Error in Analysis'.")
                    df.at[idx, 'Final Conclusion'] = 'Error in Analysis'
                    df.at[idx, 'Control Status'] = 'Error in Analysis'
                    if socketio:
                        progress = ((idx + 1) / total_controls) * 100
                        socketio.emit('progress', {'progress': progress}, broadcast=True)
                else:
                    logger.info(f"Retrying final conclusion for control at index {idx} (Attempt {retries}/{max_retries})...")

    logger.info("Final conclusions processed for all controls.")
    return df

def load_excel_file(file_path):
    """Load Excel file into DataFrame."""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        logging.error(f"Error loading Excel file {file_path}: {e}")
        raise

def map_columns_by_position(df):
    """Map DataFrame columns based on position."""
    try:
        required_columns = [
            'Sr. No.',
            'User Org Control Domain',
            'User Org Control Sub-Domain',
            'User Org Control Statement'
        ]
        
        if len(df.columns) < len(required_columns):
            return None, "Not enough columns in the framework file"
        
        df.columns = required_columns + list(df.columns[len(required_columns):])
        return df, None
    except Exception as e:
        return None, str(e)

def merge_dataframes(framework_df, analysis_df):
    """Merge framework and analysis DataFrames."""
    try:
        return pd.merge(
            framework_df,
            analysis_df,
            left_on='User Org Control Statement',
            right_on='Control',
            how='left'
        )
    except Exception as e:
        logging.error(f"Error merging DataFrames: {e}")
        raise

def create_final_dataframe(merged_df):
    """Create final DataFrame with required columns."""
    try:
        final_columns = [
            'Sr. No.',
            'User Org Control Domain',
            'User Org Control Sub-Domain',
            'User Org Control Statement',
            'Service Org Control IDs',
            'Service Org Controls',
            'Compliance Score',
            'Detailed Analysis',
            'Control Status'
        ]
        
        # Initialize empty columns if they don't exist
        for col in final_columns:
            if col not in merged_df.columns:
                merged_df[col] = None
        
        return merged_df[final_columns], None
    except Exception as e:
        return None, str(e)

def remove_not_met_controls(df):
    """Remove controls with 'Not Met' status."""
    try:
        return df[df['Control Status'] != 'Not Met']
    except Exception as e:
        logging.error(f"Error removing Not Met controls: {e}")
        raise

def save_to_excel(df, output_path):
    """Save DataFrame to Excel."""
    try:
        df.to_excel(output_path, index=False)
    except Exception as e:
        logging.error(f"Error saving to Excel {output_path}: {e}")
        raise

# Example usage (You can remove or comment out this part if integrating into a larger system)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Analysis Script")
    parser.add_argument('--framework', required=True, help="Path to the framework Excel file")
    parser.add_argument('--responses', required=True, help="Path to the responses Excel file")
    parser.add_argument('--output', required=True, help="Path to save the output Excel file")
    parser.add_argument('--top_k', type=int, default=3, help="Number of top responses to consider")
    args = parser.parse_args()

    try:
        framework_df = load_excel_file(args.framework)
        framework_df, error = map_columns_by_position(framework_df)
        if error:
            logger.error(error)
            exit(1)

        analysis_df = load_responses(args.responses)
        merged_df = merge_dataframes(framework_df, analysis_df)
        final_df, error = create_final_dataframe(merged_df)
        if error:
            logger.error(error)
            exit(1)

        processed_df = process_controls(final_df)
        processed_df = process_final_conclusions(processed_df)

        save_to_excel(processed_df, args.output)

        logger.info("LLM Analysis completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during LLM Analysis: {e}")
        exit(1)
