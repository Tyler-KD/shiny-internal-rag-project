import os
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.partition.image import partition_image

def process_documents(files, api_key):
    processed_docs = []
    loaders = {
        '.docx': partition_docx,
        '.pdf': partition_pdf,
        '.html': partition_html,
        '.htm': partition_html,
        '.txt': partition_text,
        '.png': partition_image,
        '.jpeg': partition_image,
        '.jpg': partition_image,
    }

    def load_file(file):
        try:
            file_path = os.path.join("documents", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            _, file_extension = os.path.splitext(file.name)
            loader_function = loaders.get(file_extension.lower())
            if not loader_function:
                raise ValueError(f"Unsupported file type: {file_extension}")

            print(f"Loading {file.name} with extension {file_extension}")
            elements = loader_function(file_path)
            text = "\n".join([element.text for element in elements if element.text])
            print(f"Processed file {file.name} successfully.")
            return text
        except Exception as e:
            print(f"Error processing file {file.name}: {str(e)}")
            raise e

    max_workers = min(32, (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(load_file, files)
        for result in results:
            processed_docs.append(result)

    # Split the documents into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = []
    for doc in processed_docs:
        texts.extend(text_splitter.split_text(doc))

    # Create embeddings and store them in Chroma
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = Chroma.from_texts(texts, embeddings)

    return db

def query_chatgpt(db, query, api_key):
    # Set up the ChatGPT model
    chat = ChatOpenAI(openai_api_key=api_key, temperature=0, max_tokens=300)
    # Create a ConversationalRetrievalChain for querying
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        max_tokens_limit=1000
    )
    result = qa.invoke({"question": query, "chat_history": []})
    return result["answer"]

# Ensure the 'documents' directory exists
if not os.path.exists("documents"):
    os.makedirs("documents")