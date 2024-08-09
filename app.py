from shiny import App, render, ui, reactive
import os
import shutil
from openai import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
import fitz  # PyMuPDF

load_dotenv()

# Ensures the uploaded_files directory exists
if not os.path.exists('uploaded_files'):
    os.makedirs('uploaded_files')

# Set OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# used so that any files uploaded have the correct syntax
def sanitize_filename(filename):
    return "".join(c if c.isalnum() else "_" for c in filename)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

app_ui = ui.page_fluid(
    ui.page_auto(
        ui.tags.style("""
            # body {
            #     background-color: #ffeaea;
            # }
                      
            #user_question {
                width: 50vw;
            }
            #process_output {
                white-space: pre-wrap;
            }
        """)
    ),
    ui.h1("File Upload and Local Storage"),
    ui.input_file("file_upload", "Upload a file", multiple=True),
    ui.output_text_verbatim("file_info"), # fix this so that the files uploaded go straight to the list of "uploaded files"
    ui.markdown("Uploaded files"),
    ui.output_ui("uploaded_files_list"),
    ui.input_action_button("delete_button", "Delete Selected Files"),
    ui.input_action_button("process_button", "Process Selected Files"),
    ui.input_text("user_question", "Ask a question about the document(s):"),
    ui.input_action_button("submit_question", "Submit Question"),
    ui.output_text_verbatim("process_output")
)

# parent server
def server(input, output, session):
    # defined so that no errors occur
    delete_trigger = reactive.Value(0)
    process_output_value = reactive.Value("")
    conversation_history = reactive.Value([])
    
    @output
    @render.text
    def file_info():
        if input.file_upload() is None:
            return "No file uploaded this session."
        
        files = input.file_upload()
        file_details = []
        
        # a for loop that moves the files to the uploaded_files directory - could be used to move file_info from where its currently displayed in the ui.
        for file in files:
            filename = file["name"]
            file_path = file["datapath"]
            save_path = os.path.join("uploaded_files", filename)
            shutil.move(file_path, save_path)
            file_details.append(f"File uploaded: {filename}")
        
        # Trigger update for the file list UI
        delete_trigger.set(delete_trigger() + 1)
        return "\n".join(file_details)
    
    @output
    @render.ui
    def uploaded_files_list():
        delete_trigger()
        # checks if there are any files in the directory
        file_list = os.listdir('uploaded_files')
        if not file_list:
            return ui.p("No files available.")
        
        checkboxes = [ui.tags.li(
            ui.input_checkbox(f"select_{sanitize_filename(file)}", file, value=False),
        ) for file in file_list]
        #  makes an unordered list of the files
        return ui.tags.ul(*checkboxes)

    @reactive.Effect
    @reactive.event(input.delete_button)
    def delete_selected_files():
        file_list = os.listdir('uploaded_files')
        for file in file_list:
            checkbox_id = f"select_{sanitize_filename(file)}"
            if input[checkbox_id]():
                try:
                    os.remove(os.path.join('uploaded_files', file))
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
        
        delete_trigger.set(delete_trigger() + 1)

    @output
    @render.text
    def process_output():
        return process_output_value()

    @reactive.Effect
    @reactive.event(input.process_button)
    def process_selected_files():
        file_list = os.listdir('uploaded_files')
        selected_files = []
        for file in file_list:
            checkbox_id = f"select_{sanitize_filename(file)}"
            if input[checkbox_id]():
                selected_files.append(os.path.join('uploaded_files', file))
        
        if not selected_files:
            process_output_value.set("No files selected for processing.")
            return
        
        output_list = []
        for file_path in selected_files:
            if file_path.endswith('.pdf'):
                content = extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r') as f:
                    content = f.read()
            
            document = Document(page_content=content)
            # Process the document using LangChain and OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                    {"role": "user", "content": f"Summarize the following document: {content}"}
                ],
                max_tokens=150
            )
            summary = response.choices[0].message.content
            output_list.append(f"File: {os.path.basename(file_path)}\nSummary: {summary.strip()}")

        process_output_value.set("\n\n".join(output_list))

    @reactive.Effect
    @reactive.event(input.submit_question)
    def handle_question():
        question = input.user_question()
        if not question:
            process_output_value.set("Please enter a question.")
            return

        file_list = os.listdir('uploaded_files')
        selected_files = []
        for file in file_list:
            checkbox_id = f"select_{sanitize_filename(file)}"
            if input[checkbox_id]():
                selected_files.append(os.path.join('uploaded_files', file))

        if not selected_files:
            process_output_value.set("No files selected. Please select at least one file.")
            return

        document_contents = []
        for file_path in selected_files:
            if file_path.endswith('.pdf'):
                content = extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r') as f:
                    content = f.read()
            document_contents.append(f"File: {os.path.basename(file_path)}\nContent: {content}")

        combined_content = "\n\n".join(document_contents)

        history = conversation_history.get()

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."},
            {"role": "user", "content": f"Here are the documents:\n\n{combined_content}\n\nPlease answer the following question based on these documents: {question}"}
        ]

        messages.extend(history)

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300
            )
            answer = response.choices[0].message.content

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            conversation_history.set(history)

            output = f"Question: {question}\n\nAnswer: {answer}"
            process_output_value.set(output)
        except Exception as e:
            process_output_value.set(f"An error occurred: {str(e)}")

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()