from shiny import App, render, ui, reactive
import os
import shutil
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import pytz
from styles import styles_app
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
import unstructured_client
import numpy as np
import pickle
from typing import Tuple, List, Set
import json
import gc
import random
import asyncio
from tqdm import tqdm
import aiohttp
from pathlib import Path


# Import functions from utils.py
from utils import (
    load_indices_and_mappings, load_processed_files, save_processed_files,
    sanitize_filename, process_file_with_unstructured_api, chunk_document,
    load_sensitive_files, save_sensitive_file, split_content, model,
    FAISS_INDICES_FILE, CHUNK_MAPPING_FILE, SENSITIVE_FILES_JSON, initialize_faiss_index
)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Used to path for images
here = Path(__file__).parent

# Ensure the 'uploaded_files' directory exists
if not os.path.exists('uploaded_files'):
    os.makedirs('uploaded_files')

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
client = unstructured_client.UnstructuredClient(
    api_key_auth=UNSTRUCTURED_API_KEY,
    server_url="https://api.unstructuredapp.io/",
)

# Load indices and mappings
indices, chunk_mapping = load_indices_and_mappings()

# Initialize processed files
processed_files = load_processed_files()

# Initialize ChatOpenAI with GPT-3.5-Turbo
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=300
)

# Process a file: extract content, chunk it, and add embeddings to FAISS index
def process_file(file_name) -> Tuple[str, bool]:
    global indices, chunk_mapping
    
    if file_name in processed_files:
        is_sensitive = file_name in load_sensitive_files()
        return f"File {file_name} has already been processed.", is_sensitive
    
    file_path = os.path.join('uploaded_files', file_name)
    content = process_file_with_unstructured_api(file_path, client)
    
    # Check if the file name contains the keyword "admin"
    is_sensitive = "admin" in file_name.lower()
    
    if not content.startswith("Error"):
        chunks = chunk_document(content)
        embeddings = model.encode(chunks)
        
        index = initialize_faiss_index()  # This line should now work
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
        
        # Save sensitive file information
        if is_sensitive:
            save_sensitive_file(file_name)
        
        logging.info(f"Processed file: {file_name}. Index size: {index.ntotal}, Chunk mapping size: {len(file_chunk_mapping)}, Sensitive: {is_sensitive}")
        return f"File processed: {file_name}", is_sensitive
    return content, False

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

# Define the UI for the Shiny app
app_ui = ui.page_fluid(
    ui.tags.style(styles_app),  # Apply custom styles
    ui.output_image("blueGlobeBG"),  # Display background image
    ui.page_sidebar(
        ui.sidebar(
            ui.h2("File Upload", style="color: #0E4878;"),  # Sidebar header with custom color
            ui.input_file("file_upload", "Upload a file by clicking Browse below", multiple=True),  # File upload input
            ui.output_text_verbatim("file_info"),  # Display uploaded file info
            ui.accordion(
                ui.accordion_panel(
                    ui.markdown("### Uploaded files:"),  # Display markdown text
                    ui.output_ui("uploaded_files_list"),  # Display list of uploaded files
                )
            ),
            ui.input_action_button("process_button", "Process Selected Files", class_="btn-primary mt-2 PSDbtn"),  # Button to process files
            ui.input_action_button("summarize_button", "Summarize Selected Files", class_="btn-primary mt-2 PSDbtn"),  # Button to summarize files
            ui.input_action_button("delete_button", "Delete Selected Files", class_="btn-danger mt-2 PSDbtn"),  # Button to delete files
            ui.div(
                ui.input_text_area("user_question", "Ask a question about the file(s):"),  # Text area to ask questions
                class_="mt-3"
            ),
            ui.input_action_button("submit_question", "Submit Question", class_="btn-primary mt-2"),  # Button to submit questions
        open="open"),
        ui.div ({"class": "logoWelcome"}, 
            ui.output_image("logo_transparent", inline=True),
            ui.div({"class": "time-display-column"},
                ui.output_text("time_display"),  # Display time for different locations
                ui.h1({"class": "greeting"}, "Welcome!  Bienvenue!  !أهلا وسهلا"),  # Display multilingual greeting
            ),
        ),  # Display version
        ui.div (            
            ui.h3("Instructions:"),
                ui.tags.ul(
                ui.tags.li("In the sidebar to the left, click 'Browse' to select your files."),
                ui.tags.li("Click 'Process Selected Files' to initiate processing."),
                ui.tags.li("Select files and click 'Summarize Selected Files' or 'Submit Question.'"),
                ui.tags.li("Output will be displayed below.")
                ),
            ui.h3("Output:", class_="section-header"),  # Section header
            ui.output_text_verbatim("process_output"),  # Display processing output
        ),
        ui.output_text_verbatim("api_key_info"),  # Display API key info
        ui.output_text_verbatim("progress_output"),  # Display progress output
    )
)

# Define the server logic for the Shiny app
def server(input, output, session):
    file_list = reactive.Value([])
    processed_files_reactive = reactive.Value(set())
    sensitive_files = reactive.Value(load_sensitive_files())
    process_output_value = reactive.Value("")
    progress_output_value = reactive.Value("")
    conversation_history = reactive.Value([])

    @render.image  
    def logo_transparent():
        img = {"src": here / "www/logo-transparent.png", "width": "300px"}  
        return img
    
    @render.image  
    def blueGlobeBG():
        img = {"src": here / "www/blueGlobeBG.png"}  
        return img

    @reactive.Effect
    def _():
        file_list.set(os.listdir('uploaded_files'))
        processed_files_reactive.set(processed_files)

    @output
    @render.text
    def time_display():
        dc_time = datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M")
        morocco_time = datetime.now(pytz.timezone("Africa/Casablanca")).strftime("%H:%M")
        return f"⏲D.C.: {dc_time} 🕰Morocco: {morocco_time}"

    @output
    @render.ui
    def uploaded_files_list():
        current_files = file_list()
        current_processed = processed_files_reactive()
        current_sensitive = sensitive_files.get()

        if not current_files:
            return ui.p("No files available.")
        
        master_checkbox = ui.div(
            ui.input_checkbox("select_all_files", "Select All", value=False),
            class_="mb-3"
        )
        
        checkboxes = [ui.div(
            ui.input_checkbox(f"select_{sanitize_filename(file)}", file, value=True),
            ui.span(" (Processed)" if file in current_processed else " (Not processed)", style="font-size: 0.8em; margin-left: 28px"),
            ui.span(" [SENSITIVE]" if file in current_sensitive else "", style="color: red; font-weight: bold; font-size: 0.8em;"),
            class_="mb-2"
        ) for file in current_files]
        
        return ui.div(master_checkbox, *checkboxes)


    @reactive.Effect
    @reactive.event(input.select_all_files)
    def toggle_all_checkboxes():
        current_files = file_list()  # Retrieve the current list of files
        select_all = input.select_all_files()

        for file in current_files:
            checkbox_id = f"select_{sanitize_filename(file)}"
            ui.update_checkbox(checkbox_id, value=select_all)


    @reactive.Effect
    def update_select_all():
        current_files = file_list()  # Retrieve the current list of files
        all_checked = all([input[f"select_{sanitize_filename(file)}"] for file in current_files])

        ui.update_checkbox("select_all_files", value=all_checked)

    # @output
    # @render.text
    # def api_key_info():
    #     if UNSTRUCTURED_API_KEY:
    #         masked_key = UNSTRUCTURED_API_KEY[:4] + '*' * (len(UNSTRUCTURED_API_KEY) - 8) + UNSTRUCTURED_API_KEY[-4:]
    #         return f"Unstructured API Key: {masked_key}"
    #     else:
    #         return "Unstructured API Key is not set in the .env file"

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
                result, is_sensitive = process_file(file)
                results.append(result)
                if is_sensitive:
                    current_sensitive = sensitive_files.get()
                    current_sensitive.add(file)
                    sensitive_files.set(current_sensitive)
                processed_files_reactive.set(processed_files)
                file_list.set(os.listdir('uploaded_files'))
                ui.update_checkbox(f"select_{sanitize_filename(file)}", value=True)
                gc.collect()
            
            process_output_value.set("\n".join(results))
        except Exception as e:
            logging.error(f"Error in process_selected_files: {str(e)}")
            process_output_value.set(f"Error processing files: {str(e)}")
        finally:
            gc.collect()

    @reactive.Effect
    @reactive.event(input.delete_button)
    def delete_selected_files():
        global indices, chunk_mapping
        current_files = file_list()
        sensitive_files_set = load_sensitive_files()
        files_deleted = []

        for file in current_files:
            checkbox_id = f"select_{sanitize_filename(file)}"
            if input[checkbox_id]():
                try:
                    file_path = os.path.join('uploaded_files', file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    if file in processed_files:
                        processed_files.remove(file)
                        save_processed_files(processed_files)
                    
                    if file in indices:
                        del indices[file]
                    if file in chunk_mapping:
                        del chunk_mapping[file]
                    
                    # Remove from sensitive files if present
                    if file in sensitive_files_set:
                        sensitive_files_set.remove(file)
                    
                    files_deleted.append(file)
                    
                except Exception as e:
                    logging.error(f"Error deleting {file}: {str(e)}")
        
        # Update FAISS indices and chunk mapping files
        with open(FAISS_INDICES_FILE, 'wb') as f:
            pickle.dump(indices, f)
        with open(CHUNK_MAPPING_FILE, 'wb') as f:
            pickle.dump(chunk_mapping, f)
        
        # Update sensitive files JSON
        with open(SENSITIVE_FILES_JSON, 'w') as f:
            json.dump(list(sensitive_files_set), f)
        
        # Update reactive values
        file_list.set(os.listdir('uploaded_files'))
        processed_files_reactive.set(processed_files)
        sensitive_files.set(sensitive_files_set)
        
        # Log the deleted files
        if files_deleted:
            logging.info(f"Deleted files: {', '.join(files_deleted)}")
        
        gc.collect()

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
                content = process_file_with_unstructured_api(file_path, client)
                if not content.startswith("Error"):
                    try:
                        summary = await summarize_document(content)
                        summaries.append(f"Summary of {file}:\n{summary}")
                    except Exception as e:
                        logging.error(f"Error summarizing {file}: {str(e)}")
                        summaries.append(f"Error summarizing {file}: {str(e)}")
                else:
                    summaries.append(f"Unable to summarize {file}: {content}")
                await asyncio.sleep(0.1)
                gc.collect()
            
            process_output_value.set("\n\n".join(summaries))
        except Exception as e:
            logging.error(f"Error in summarize_selected_files: {str(e)}")
            process_output_value.set(f"Error summarizing files: {str(e)}")
        finally:
            gc.collect()

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

            output = f"Question: {question}\n\nAnswer: {answer}\nRelevant files: {set(relevant_files)}"
            process_output_value.set(output)
        except Exception as e:
            logging.error(f"Error in handle_question: {str(e)}")
            process_output_value.set(f"An error occurred: {str(e)}")
        finally:
            gc.collect()

    @reactive.Effect
    @reactive.event(input.file_upload)
    def handle_file_upload():
        uploaded_files = input.file_upload()
        if uploaded_files:
            for file in uploaded_files:
                filename = file["name"]
                file_path = file["datapath"]
                save_path = os.path.join("uploaded_files", filename)
                shutil.move(file_path, save_path)
            file_list.set(os.listdir('uploaded_files'))

# Create and run the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()