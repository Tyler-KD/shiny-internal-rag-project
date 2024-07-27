# app.py
from shiny import App, render, ui
import requests
import pandas as pd

app_ui = ui.page_fluid(
    ui.h1("RAG Search Engine"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.file_input("file", "Upload Document"),
            ui.input_text("query", "Enter search query"),
            ui.input_action_button("search_button", "Search"),
        ),
        ui.panel_main(
            ui.output_text("upload_result"),
            ui.output_table("search_result"),
        )
    )
)

def server(input, output, session):
    base_url = "http://localhost:8000"  # Update this with your FastAPI backend URL

    @output
    @render.text
    def upload_result():
        if input.file():
            files = {'file': input.file()}
            response = requests.post(f"{base_url}/upload", files=files)
            return f"Upload response: {response.json()['message']}"

    @output
    @render.table
    def search_result():
        if input.search_button():
            response = requests.post(f"{base_url}/search", json={'query': input.query()})
            results = response.json()['results']
            return pd.DataFrame(results)

app = App(app_ui, server)
