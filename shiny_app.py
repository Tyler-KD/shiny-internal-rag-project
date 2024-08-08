import shiny
from shiny import ui, reactive, render
import os
from utils import process_documents, query_chatgpt
import requests
from io import BytesIO

# Ensure the 'documents' directory exists
if not os.path.exists("documents"):
    os.makedirs("documents")

app_ui = ui.page_fluid(
    ui.h2("RAG Search Pipeline - Shiny for Python"),
    ui.input_text("api_key", "Enter your OpenAI API Key"),
    ui.output_text_verbatim("api_key_status"),
    ui.input_file("uploaded_files", "Upload documents", multiple=True, accept=[".docx", ".pdf", ".html", ".htm", ".txt", ".png", ".jpeg", ".jpg"]),
    ui.output_text_verbatim("status"),
    ui.h3("Processed Documents"),
    ui.output_text_verbatim("processed_docs"),
    ui.input_text("query", "Ask a question about the documents"),
    ui.output_text_verbatim("response"),
    ui.HTML(
        """
        <script>
        document.getElementById('api_key').type = 'password';
        </script>
        """
    ),
)

def server(input, output, session):
    db = reactive.Value(None)  # Store the database reference globally
    status_message = reactive.Value("")
    api_key_status_message = reactive.Value("")
    processed_documents = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.api_key)
    def verify_api_key():
        api_key = input.api_key()
        if api_key:
            status_message.set("Verifying API key...")
            print("Verifying API key...")
            try:
                response = requests.get("https://api.openai.com/v1/engines", headers={"Authorization": f"Bearer {api_key}"})
                if response.status_code == 200:
                    api_key_status_message.set("API key verified successfully! ✅")
                    status_message.set("API key verified successfully!")
                    print("API key verified successfully!")
                else:
                    api_key_status_message.set("Invalid API key ❌")
                    status_message.set(f"Invalid API key: {response.text}")
                    print(f"Invalid API key: {response.text}")
            except Exception as e:
                api_key_status_message.set("Error verifying API key ❌")
                status_message.set(f"Error verifying API key: {str(e)}")
                print(f"Error verifying API key: {str(e)}")

    @reactive.Effect
    @reactive.event(input.uploaded_files, input.api_key)
    def process_files():
        status_message.set("Processing documents...")
        print("Processing documents...")
        if not input.api_key() or not input.uploaded_files():
            status_message.set("Please upload documents and enter your API key.")
            print("Please upload documents and enter your API key.")
            return None

        api_key = input.api_key()
        uploaded_files = input.uploaded_files()

        files = []
        file_names = []
        for file_info in uploaded_files:
            file_name = file_info["name"]
            file_path = file_info["datapath"]
            with open(file_path, "rb") as f:
                file_content = f.read()
                file_like_object = BytesIO(file_content)
                file_like_object.name = file_name
                files.append(file_like_object)
                file_names.append(file_name)
                print(f"File {file_name} processed and added to list")

        try:
            # Show processing indicator
            status_message.set("Processing documents... Please wait.")
            print("Processing documents... Please wait.")
            db_result = process_documents(files, api_key)
            db.set(db_result)
            processed_documents.set(file_names)
            status_message.set("Documents processed successfully! You can now ask a question.")
            print("Documents processed successfully!")
        except Exception as e:
            status_message.set(f"Error processing documents: {str(e)}")
            print(f"Error processing documents: {str(e)}")

    @reactive.Calc
    def generate_response():
        query = input.query()
        if query and db.get() and input.api_key():
            status_message.set("Generating response from OpenAI...")
            print("Generating response from OpenAI...")
            try:
                response = query_chatgpt(db.get(), query, input.api_key())
                status_message.set("Response generated successfully.")
                print("Response generated successfully.")
                return response
            except Exception as e:
                status_message.set(f"Error generating response: {str(e)}")
                print(f"Error generating response: {str(e)}")
                return f"Error generating response: {str(e)}"
        return ""

    @output
    @render.text
    def api_key_status():
        return api_key_status_message.get()

    @output
    @render.text
    def status():
        return status_message.get()

    @output
    @render.text
    def response():
        return generate_response()

    @output
    @render.text
    def processed_docs():
        docs = processed_documents.get()
        return "\n".join(docs) if docs else "No documents processed yet."

app = shiny.App(app_ui, server)

# To run the app locally:
# shiny run --host 0.0.0.0 --port 8502 shiny_app.py