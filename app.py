# Import necessary modules and classes from various libraries
from shiny import App, render, ui, reactive
import os
import shutil
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import pytz
from styles import styles_app
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
import unstructured_client
from unstructured_client.models import operations, shared
import tiktoken
from sentence_transformers import SentenceTransformer
import json
import gc
import faiss
import numpy as np
import pickle
from typing import List
import pandas as pd
import io
import random
import asyncio
from functools import lru_cache
from tqdm import tqdm
import aiohttp
import chardet

# Set up logging with a specific format
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Disable logs from httpx and httpcore
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Reduce logs from OpenAI's library
logging.getLogger('openai').setLevel(logging.WARNING)

# Load environment variables from a .env file
load_dotenv()

# Ensure the 'uploaded_files' directory exists
if not os.path.exists('uploaded_files'):
    os.makedirs('uploaded_files')

# Initialize the OpenAI client using the API key from the environment variables
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set the Unstructured Serverless API key from environment variables
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")

# Initialize the Unstructured client with the API key and server URL
client = unstructured_client.UnstructuredClient(
    api_key_auth=UNSTRUCTURED_API_KEY,
    server_url="https://api.unstructuredapp.io/",
)

# Initialize the tokenizer with a specific encoding
tokenizer = tiktoken.get_encoding("cl100k_base")

# Track processed files using a JSON file
PROCESSED_FILES_TRACKER = 'processed_files.json'

# Initialize the SentenceTransformer model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Files to store FAISS indices and chunk mappings
FAISS_INDICES_FILE = 'faiss_indices.pkl'
CHUNK_MAPPING_FILE = 'chunk_mapping.pkl'

# Function to initialize a FAISS index with a specific dimension (384 in this case)
def initialize_faiss_index():
    return faiss.IndexFlatL2(384)

# Load FAISS indices and chunk mappings if they exist, otherwise initialize empty dictionaries
if os.path.exists(FAISS_INDICES_FILE) and os.path.exists(CHUNK_MAPPING_FILE):
    with open(FAISS_INDICES_FILE, 'rb') as f:
        indices = pickle.load(f)
    with open(CHUNK_MAPPING_FILE, 'rb') as f:
        chunk_mapping = pickle.load(f)
else:
    indices = {}
    chunk_mapping = {}

# Load processed files from the tracker file
def load_processed_files():
    if os.path.exists(PROCESSED_FILES_TRACKER):
        with open(PROCESSED_FILES_TRACKER, 'r') as f:
            return set(json.load(f))
    return set()

# Save the processed files to the tracker file
def save_processed_files(processed_files):
    with open(PROCESSED_FILES_TRACKER, 'w') as f:
        json.dump(list(processed_files), f)

# Initialize the set of processed files
processed_files = load_processed_files()

# Function to count the number of tokens in a string
def num_tokens_from_string(string: str) -> int:
    return len(tokenizer.encode(string))

# Sanitize filenames by replacing non-alphanumeric characters with underscores
def sanitize_filename(filename):
    return "".join(c if c.isalnum() else "_" for c in filename)

# Get the current time for a specific timezone
def get_time_for_location(timezone):
    return datetime.now(pytz.timezone(timezone)).strftime("%H:%M")

def process_excel_or_csv_file(file_path):
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

def process_file_with_unstructured_api(file_path):
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
def chunk_document(text, max_tokens=2000):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=200,  # Reduced overlap between chunks
        length_function=num_tokens_from_string,
    )
    return text_splitter.split_text(text)

# Process a file: extract content, chunk it, and add embeddings to FAISS index
def process_file(file_name):
    global indices, chunk_mapping
    
    if file_name in processed_files:
        return f"File {file_name} has already been processed."
    
    file_path = os.path.join('uploaded_files', file_name)
    content = process_file_with_unstructured_api(file_path)
    if not content.startswith("Error"):
        chunks = chunk_document(content)
        embeddings = model.encode(chunks)
        
        index = initialize_faiss_index()
        file_chunk_mapping = {}
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            index.add(np.array([embedding]))
            chunk_id = index.ntotal - 1
            file_chunk_mapping[chunk_id] = chunk
        
        indices[file_name] = index
        chunk_mapping[file_name] = file_chunk_mapping
        
        processed_files.add(file_name)
        save_processed_files(processed_files)
        
        # Save updated indices and mapping
        with open(FAISS_INDICES_FILE, 'wb') as f:
            pickle.dump(indices, f)
        with open(CHUNK_MAPPING_FILE, 'wb') as f:
            pickle.dump(chunk_mapping, f)
        
        logging.info(f"Processed file: {file_name}. Index size: {index.ntotal}, Chunk mapping size: {len(file_chunk_mapping)}")
        return f"File processed: {file_name}"
    return content

def retrieve_relevant_chunks(question, selected_files, top_k=5, max_context_length=3000):
    try:
        question_embedding = model.encode([question])
        logging.info(f"Question embedding shape: {question_embedding.shape}")
        
        all_relevant_chunks = []
        all_relevant_files = []
        
        for file_name in selected_files:
            if file_name in indices:
                index = indices[file_name]
                file_chunk_mapping = chunk_mapping[file_name]
                
                logging.info(f"FAISS index size for {file_name}: {index.ntotal}")
                
                D, I = index.search(question_embedding, top_k)
                logging.info(f"FAISS search results for {file_name} - Distances: {D}, Indices: {I}")
                
                relevant_chunks = [file_chunk_mapping[i] for i in I[0] if i in file_chunk_mapping]
                
                all_relevant_chunks.extend(relevant_chunks)
                all_relevant_files.extend([file_name] * len(relevant_chunks))
        
        # Limit the context length to the specified max_context_length
        context = "\n\n".join(all_relevant_chunks)
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        logging.info(f"Retrieved {len(all_relevant_chunks)} relevant chunks from files: {set(all_relevant_files)}")
        return all_relevant_chunks, all_relevant_files
    except Exception as e:
        logging.error(f"Error retrieving relevant chunks: {str(e)}")
        return [], []

# Modify the summarize_document function
async def summarize_document(content: str, max_tokens: int = 4000) -> str:
    chunks = chunk_document(content, max_tokens)
    summaries = []

    async with aiohttp.ClientSession() as session:
        chat = ChatOpenAI(temperature=0, max_tokens=500)
        summary_prompt = PromptTemplate(
            input_variables=["content"],
            template="Summarize the following content in a concise manner, capturing the main points and key information:\n\n{content}\n\nSummary:"
        )
        summary_chain = LLMChain(llm=chat, prompt=summary_prompt)

        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(summarize_chunk(session, summary_chain, chunk))
            tasks.append(task)

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing chunks"):
            try:
                summary = await task
                summaries.append(summary.strip())
            except Exception as e:
                logging.error(f"Error summarizing chunk: {str(e)}")
                summaries.append(f"Error summarizing chunk: {str(e)}")

    combined_summary = "\n\n".join(summaries)

    try:
        final_summary = await summary_chain.arun(content=combined_summary)
        return final_summary.strip()
    except Exception as e:
        logging.error(f"Error creating final summary: {str(e)}")
        return f"Error creating final summary: {str(e)}"

async def summarize_chunk(session, summary_chain, chunk):
    retries = 3
    for attempt in range(retries):
        try:
            return await summary_chain.arun(content=chunk)
        except Exception as e:
            if "429 Too Many Requests" in str(e) and attempt < retries - 1:
                            wait_time = 2 ** attempt + random.uniform(0, 1)
                            logging.info(f"Rate limit hit, retrying in {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)
            else:
                raise

@lru_cache(maxsize=100)
def get_cached_summary(file_path: str) -> str:
    with open(file_path, 'r') as f:
        content = f.read()
    return asyncio.run(summarize_document(content))

# Function to split content into smaller parts
def split_content(content: str, max_tokens: int = 4000) -> List[str]:
    tokens = tokenizer.encode(content)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

# Define the UI for the Shiny app
app_ui = ui.page_fluid(
    ui.tags.style(styles_app),  # Apply custom styles
    ui.row(
        ui.column(4,
            ui.div({"class": "sidebar"},
                ui.h2("File Upload", style="color: #0E4878;"),  # Sidebar header with custom color
                ui.input_file("file_upload", "Upload a file by clicking Browse below", multiple=True),  # File upload input
                ui.output_text_verbatim("file_info"),  # Display uploaded file info
                ui.markdown("### Uploaded files:"),  # Display markdown text
                ui.output_ui("uploaded_files_list"),  # Display list of uploaded files
                ui.input_action_button("process_button", "Process Selected Files", class_="btn-primary mt-2"),  # Button to process files
                ui.input_action_button("summarize_button", "Summarize Selected Files", class_="btn-primary mt-2"),  # Button to summarize files
                ui.input_action_button("delete_button", "Delete Selected Files", class_="btn-danger mt-2"),  # Button to delete files
                ui.div(
                    ui.input_text_area("user_question", "Ask a question about the file(s):"),  # Text area to ask questions
                    class_="mt-3"
                ),
                ui.input_action_button("submit_question", "Submit Question", class_="btn-primary mt-2"),  # Button to submit questions
            )
        ),
        ui.column(8,
            ui.div({"class": "main-content"},
                ui.div({"class": "time-display"},
                    ui.output_text("time_display"),  # Display time for different locations
                ),
                ui.h1({"class": "greeting"}, "Welcome!  Bienvenue!  !أهلا وسهلا"),  # Display multilingual greeting
                ui.h3("Conversation", class_="section-header"),  # Section header
                ui.output_text_verbatim("process_output"),  # Display processing output
                ui.output_text_verbatim("api_key_info"),  # Display API key info
                ui.output_text_verbatim("progress_output"),  # Display progress output
            )
        )
    )
)

# Define the server logic for the Shiny app
def server(input, output, session):
    file_list = reactive.Value([])  # Reactive value to store the list of files
    processed_files_reactive = reactive.Value(set())  # Reactive value to store processed files
    process_output_value = reactive.Value("")  # Reactive value to store process output
    progress_output_value = reactive.Value("")  # Reactive value to store progress output
    conversation_history = reactive.Value([])  # Reactive value to store conversation history
    
    @reactive.Effect
    def _():
        file_list.set(os.listdir('uploaded_files'))
        processed_files_reactive.set(processed_files)
    
    @output
    @render.text
    def time_display():
        dc_time = get_time_for_location("US/Eastern")
        morocco_time = get_time_for_location("Africa/Casablanca")
        return f"⏲D.C.: {dc_time} 🕰Morocco: {morocco_time}"
    
    @output
    @render.text
    def file_info():
        if input.file_upload() is None:
            return "No new files uploaded this session."
        
        files = input.file_upload()
        file_details = []
        
        for file in files:
            filename = file["name"]
            file_path = file["datapath"]
            save_path = os.path.join("uploaded_files", filename)
            shutil.move(file_path, save_path)
            file_details.append(f"File uploaded: {filename}")
        
        file_list.set(os.listdir('uploaded_files'))
        return "\n".join(file_details)
    
    @output
    @render.ui
    def uploaded_files_list():
        current_files = file_list()
        current_processed = processed_files_reactive()
        if not current_files:
            return ui.p("No files available.")
        
        checkboxes = [ui.div(
            ui.input_checkbox(f"select_{sanitize_filename(file)}", file, value=True),
            ui.span(" (Processed)" if file in current_processed else " (Not processed)", style="font-size: 0.8em;"),
            class_="mb-2"
        ) for file in current_files]
        return ui.div(*checkboxes)
    
    @output
    @render.text
    def api_key_info():
        if UNSTRUCTURED_API_KEY:
            masked_key = UNSTRUCTURED_API_KEY[:4] + '*' * (len(UNSTRUCTURED_API_KEY) - 8) + UNSTRUCTURED_API_KEY[-4:]
            return f"Unstructured API Key: {masked_key}"
        else:
            return "Unstructured API Key is not set in the .env file"
    
    @output
    @render.text
    def progress_output():
        return progress_output_value()
    
    @reactive.Effect
    @reactive.event(input.process_button)
    def process_selected_files():
        current_files = file_list()
        selected_files = [file for file in current_files if input[f"select_{sanitize_filename(file)}"]()]
        
        if not selected_files:
            process_output_value.set("No files selected for processing.")
            return
        
        try:
            results = []
            for file in selected_files:
                process_output_value.set(f"Processing file: {file}... Please wait.")
                result = process_file(file)
                results.append(result)
                processed_files_reactive.set(processed_files)
                gc.collect()  # Force garbage collection after each file
            
            process_output_value.set("\n".join(results))
        except Exception as e:
            logging.error(f"Error in process_selected_files: {str(e)}")
            process_output_value.set(f"Error processing files: {str(e)}")
        finally:
            gc.collect()  # Force garbage collection after processing
    
    @reactive.Effect
    @reactive.event(input.delete_button)
    def delete_selected_files():
        global indices, chunk_mapping
        current_files = file_list()
        for file in current_files:
            checkbox_id = f"select_{sanitize_filename(file)}"
            if input[checkbox_id]():
                try:
                    os.remove(os.path.join('uploaded_files', file))
                    if file in processed_files:
                        processed_files.remove(file)
                        save_processed_files(processed_files)
                        
                        # Remove index and chunks related to the deleted file
                        if file in indices:
                            del indices[file]
                        if file in chunk_mapping:
                            del chunk_mapping[file]
                        
                        # Save updated indices and mapping
                        with open(FAISS_INDICES_FILE, 'wb') as f:
                            pickle.dump(indices, f)
                        with open(CHUNK_MAPPING_FILE, 'wb') as f:
                            pickle.dump(chunk_mapping, f)
                        
                except Exception as e:
                    logging.error(f"Error deleting {file}: {str(e)}")
        
        file_list.set(os.listdir('uploaded_files'))
        processed_files_reactive.set(processed_files)
        gc.collect()  # Force garbage collection after deletion

    @output
    @render.text
    def process_output():
        return process_output_value()

    @reactive.Effect
    @reactive.event(input.summarize_button)
    async def summarize_selected_files():
        current_files = file_list()
        selected_files = [file for file in current_files if input[f"select_{sanitize_filename(file)}"]()]
        
        if not selected_files:
            process_output_value.set("No files selected for summarization.")
            return
        
        try:
            process_output_value.set("Summarizing files... Please wait.")
            summaries = []
            for file in tqdm(selected_files, desc="Summarizing files"):
                if file not in processed_files_reactive():
                    summaries.append(f"Cannot summarize {file}: File not processed yet.")
                    continue
                file_path = os.path.join('uploaded_files', file)
                content = process_file_with_unstructured_api(file_path)
                if not content.startswith("Error"):
                    try:
                        summary = await summarize_document(content)
                        summaries.append(f"Summary of {file}:\n{summary}")
                    except Exception as e:
                        logging.error(f"Error summarizing {file}: {str(e)}")
                        summaries.append(f"Error summarizing {file}: {str(e)}")
                else:
                    summaries.append(f"Unable to summarize {file}: {content}")
                await asyncio.sleep(0.1)  # Allow UI updates
                gc.collect()  # Force garbage collection after each file
            
            process_output_value.set("\n\n".join(summaries))
        except Exception as e:
            logging.error(f"Error in summarize_selected_files: {str(e)}")
            process_output_value.set(f"Error summarizing files: {str(e)}")
        finally:
            gc.collect()  # Force garbage collection after summarization

    @reactive.Effect
    @reactive.event(input.submit_question)
    def handle_question():
        question = input.user_question()
        if not question:
            process_output_value.set("Please enter a question.")
            return

        try:
            process_output_value.set("Generating response... Please wait.")
            
            current_files = file_list()
            selected_files = [file for file in current_files if input[f"select_{sanitize_filename(file)}"]()]
            
            relevant_chunks, relevant_files = retrieve_relevant_chunks(question, selected_files)
            logging.info(f"Retrieved chunks from files: {set(relevant_files)}")
            
            if not relevant_chunks:
                process_output_value.set("No relevant information found. Please make sure you've processed some files and selected them for searching.")
                return

            context = "\n\n".join(relevant_chunks)
            logging.info(f"Context being used: {context}")
            
            chat = ChatOpenAI(temperature=0, max_tokens=300)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Always strive to find and report exact data points from the context. If the context doesn't contain relevant information, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nPlease answer the question based on the given context. Report exact numbers or percentages if they are present in the context. If the context doesn't provide relevant information, state that clearly."}
            ]
            
            response = chat.invoke(messages)
            answer = response.content

            conversation_history.set(conversation_history.get() + [(question, answer)])

            output = f"Question: {question}\n\nAnswer: {answer}\n\nDebug Info:\nRelevant files: {set(relevant_files)}\nRetrieved Chunks: {relevant_chunks}\n\nContext Used: {context}"
            process_output_value.set(output)
        except Exception as e:
            logging.error(f"Error in handle_question: {str(e)}")
            process_output_value.set(f"An error occurred: {str(e)}")
        finally:
            gc.collect()  # Force garbage collection after handling the question

# Create and run the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()