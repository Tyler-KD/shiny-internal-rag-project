import os
import json
import logging
import pickle
import chardet
import pandas as pd
import numpy as np
import faiss
from typing import List, Tuple, Set
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import io
from unstructured_client.models import operations, shared

# Constants
PROCESSED_FILES_TRACKER = 'processed_files.json'
FAISS_INDICES_FILE = 'faiss_indices.pkl'
CHUNK_MAPPING_FILE = 'chunk_mapping.pkl'
SENSITIVE_FILES_JSON = 'sensitive_files.json'

# Initialize the tokenizer with a specific encoding
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize the SentenceTransformer model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to initialize a FAISS index with a specific dimension (384 in this case)
def initialize_faiss_index():
    return faiss.IndexFlatL2(384)

# Load FAISS indices and chunk mappings
def load_indices_and_mappings():
    if os.path.exists(FAISS_INDICES_FILE) and os.path.exists(CHUNK_MAPPING_FILE):
        with open(FAISS_INDICES_FILE, 'rb') as f:
            indices = pickle.load(f)
        with open(CHUNK_MAPPING_FILE, 'rb') as f:
            chunk_mapping = pickle.load(f)
    else:
        indices = {}
        chunk_mapping = {}
    return indices, chunk_mapping

# Load processed files from the tracker file
def load_processed_files() -> Set[str]:
    if os.path.exists(PROCESSED_FILES_TRACKER):
        with open(PROCESSED_FILES_TRACKER, 'r') as f:
            return set(json.load(f))
    return set()

# Save the processed files to the tracker file
def save_processed_files(processed_files: Set[str]):
    with open(PROCESSED_FILES_TRACKER, 'w') as f:
        json.dump(list(processed_files), f)

# Function to count the number of tokens in a string
def num_tokens_from_string(string: str) -> int:
    return len(tokenizer.encode(string))

# Sanitize filenames by replacing non-alphanumeric characters with underscores
def sanitize_filename(filename: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in filename)

def process_excel_or_csv_file(file_path: str) -> str:
    try:
        # Determine if the file is CSV or Excel and read it accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:  # for .xlsx and .xls files
            df = pd.read_excel(file_path)
        
        # Convert all column names to strings
        df.columns = df.columns.astype(str)
        
        # Function to format values
        def format_value(val):
            if pd.isna(val):
                return "N/A"
            elif isinstance(val, (int, float)):
                if abs(val) < 1 and val != 0:
                    return f"{val:.2%}"
                elif isinstance(val, int) or val.is_integer():
                    return f"{val:,}"
                else:
                    return f"{val:,.2f}"
            elif isinstance(val, str):
                # Try to convert percentage strings to floats
                if val.endswith('%'):
                    try:
                        return f"{float(val.rstrip('%')) / 100:.2%}"
                    except ValueError:
                        pass
            return str(val)
        
        # Apply formatting to all cells and ensure all values are strings
        formatted_df = df.applymap(lambda x: str(format_value(x)))
        
        # Create a list of strings, each containing a row of data with column names
        data_strings = []
        for index, row in formatted_df.iterrows():
            row_string = ""
            for column, value in row.items():
                row_string += f"{column}: {value}\n"
            data_strings.append(row_string)
        
        # Join all the data strings
        all_data = "\n\n".join(data_strings)
        
        # Get column names
        columns = ", ".join(df.columns)
        
        # Log DataFrame info
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        logging.info(f"DataFrame Info:\n{df_info}")
        
        # Log column names and types
        column_info = "\n".join([f"{col} ({df[col].dtype})" for col in df.columns])
        logging.info(f"Columns and their types:\n{column_info}")
        
        # Combine all information
        content = f"File Information:\nTotal Columns: {len(df.columns)}\nColumns: {columns}\n\nData:\n{all_data}"
        
        return content
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}", exc_info=True)
        return f"Error processing file: {str(e)}"

def process_file_with_unstructured_api(file_path: str, client) -> str:
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 25 * 1024 * 1024:  # 25 MB limit
            return f"File is too large ({file_size / (1024 * 1024):.2f} MB). Maximum size is 25 MB."

        # Check if the file is an Excel file
        if file_path.endswith(('.xlsx', '.xls', '.csv')):
            return process_excel_or_csv_file(file_path)

        with open(file_path, "rb") as f:
            raw_data = f.read()

        # Detect the file encoding
        detected = chardet.detect(raw_data)
        file_encoding = detected['encoding'] if detected['encoding'] else 'utf-8'

        # Decode the content using the detected encoding
        try:
            data = raw_data.decode(file_encoding)
        except UnicodeDecodeError:
            # If decoding fails, try UTF-8 as a fallback
            data = raw_data.decode('utf-8', errors='replace')

        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=raw_data,
                    file_name=os.path.basename(file_path),
                ),
                strategy=shared.Strategy.AUTO,
                languages=['eng'],
            ),
        )

        try:
            res = client.general.partition(request=req)
            extracted_text = [str(elem) for elem in res.elements]
            if not extracted_text:
                return f"No content extracted from {os.path.basename(file_path)}"
            return "\n".join(extracted_text)
        except Exception as e:
            logging.error(f"Error in Unstructured API request: {str(e)}")
            return f"Error processing file: {str(e)}"

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return f"Unexpected error processing file: {str(e)}"

# Split a document into chunks based on a maximum token count
def chunk_document(text: str, max_tokens: int = 2000) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=200,  # Reduced overlap between chunks
        length_function=num_tokens_from_string,
    )
    return text_splitter.split_text(text)

# Load sensitive files from JSON
def load_sensitive_files() -> Set[str]:
    if os.path.exists(SENSITIVE_FILES_JSON):
        with open(SENSITIVE_FILES_JSON, 'r') as f:
            return set(json.load(f))
    return set()

# Save sensitive file to JSON
def save_sensitive_file(file_name: str):
    sensitive_files = load_sensitive_files()
    sensitive_files.add(file_name)
    with open(SENSITIVE_FILES_JSON, 'w') as f:
        json.dump(list(sensitive_files), f)

# Function to split content into smaller parts
def split_content(content: str, max_tokens: int = 4000) -> List[str]:
    tokens = tokenizer.encode(content)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks