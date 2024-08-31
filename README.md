# shiny-internal-rag-project

![GitHub License](https://img.shields.io/badge/license-MIT-default.svg)

## Description

Shiny Internal RAG Project is a RAG (retrieval-augmented generation) application that leverages AI technologies and provides innovation by seamlessly summarizing, retrieving, and performing key concept extraction on various LixCap documentation. This project was built to create a competitive advantage against industry peers, increase operational efficiency for the project management and research teams, and result in informed decision making across the company. It solves the problem of needing informed decision-making with AI-generated summaries and recommendations for fast retrieval, further reading, and related topics. Some things learned throughout this project were seeing the benefits of a low code UI layer like Shiny for seemlessly integrating AI models, Unstructured Serverless API effortlessly extracts and transforms complex data for use with every major vector database and LLM (Large Language Model) framework, understanding the importance of vector databases to store embeddings for easy retrieval, and LLMs can give vastly different contextual summaries and answers depending on the model, tokenization, and chunk-overlapping.

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Video](#video)
* [Deployed Site](#deployed-site)
* [Credits](#credits)
* [Contributing](#contributing)
* [License](#license)
* [Tests](#tests)
* [Features](#features)
* [Questions](#questions)

## Installation

Within the main directory, install dependencies by running "pip install -r requirements.txt" within the terminal of the main directory. This will install all the necessary dependencies needed for Shiny, Unstructured, FAISS vector database, and OpenAI to run. Then, connect to the server with the command "shiny run app.py."

## Usage

To run this application from the terminal, enter "shiny run app.py" to start the server. The message "Uvicorn running on http://127.0.0.1:8000" will display within the console. Click the link to open up the application in the browser. Once the application is loaded, a homepage will be displayed welcoming the user containing a collapsable side pannel, a drop down menu, and an output area near the center.

In the side pannel to the left, click "Browse" to upload your files or drag and drop them into the "Browse" area. Accepted file types are txt, doc, docx, pdf, xlsx, csv, and pptx. Select all or any number of uploaded files. Click "Process Selected Files" to initiate processing. This will extract content, chunk it, and add embeddings to FAISS index. Enter your OpenAI API key to use GPT-3.5-Turbo for summarization and answering questions. Select files and click "Summarize Selected Files" or "Submit Question." "Summarize Selected Files" will summarize the following content in a concise manner, capturing the main points and key information. "Submit Question" will answer the question based on the given context, report exact numbers or percentages if they are present in the context, and indicate if the context doesn't provide relevant information. Output will be displayed below in the format of question, answer, relevant files, and model used.

If a sensitive file with the keyword "admin" is processed, it will be saved in the sensitive_files.json and appear with the word SENSITIVE in red to denote sensitive data. To delete a file, select one or multiple files from the Uploaded files and click "Delete Selected Files." This will delete the file(s) from the FAISS vector database, FAISS index, and sensitive_files.json.

For usage instructions with the deployed application using Shiny, follow the previous instructions after the application is loaded.

**Attached is a screenshot of the application Homepage:**

![Homepage](www/Shiny%20Internal%20RAG%20Project%20-%20Homepage.png)

**Attached is a screenshot of the summarization function:**

![Summarization](www/Shiny%20Internal%20RAG%20Project%20-%20Summarization.png)

**Attached is a screenshot of the submit question function:**

![Submit Question](www/Shiny%20Internal%20RAG%20Project%20-%20Submit%20Question.png)

## Video

Attached is a demo of the app in use, featuring functionality of "Summarize Selected Files", "Submit Question", and "Delete Files" buttons with corresponding output displayed in the "Output" text area:

[Demo](https://drive.google.com/file/d/1sDZRD6lRGl12DdQxakLJdLplt-T9JYVs/view?usp=sharing)

## Deployed Site

N/A

## Credits

Collaborators: [Tyler Knapp](https://github.com/Tyler-KD), [Shelby Wentz](https://github.com/kobrakitty), [Josh Hill](https://github.com/JoshHill1)

[Lonely Octopus](https://www.lonelyoctopus.com/)

[shinyapps.io](https://www.shinyapps.io/)

[shinyapps.io Documentation](https://docs.posit.co/shinyapps.io/guide/index.html)

[Unstructured Serverless API](https://docs.unstructured.io/api-reference/api-services/saas-api-development-guide)

[Unstructured IO / Unstructured API](https://github.com/Unstructured-IO/unstructured-api)

[FAISS](https://ai.meta.com/tools/faiss/)

[facebookresearch / FAISS](https://github.com/facebookresearch/faiss)

[Optimizing LLM Accuracy](https://platform.openai.com/docs/guides/optimizing-llm-accuracy)

[How-to guides | Langchain](https://python.langchain.com/v0.2/docs/how_to/#llms)

## Contributing

[Lonely Octopus](https://www.lonelyoctopus.com/)

## License

MIT License

Copyright (c) 2024 Tyler-KD

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

* (https://choosealicense.com/licenses/mit/)

## Tests

N/A

## Features

python 3.12.4, Shiny, Unstructured Serverless API, FAISS vector database, Langchain, OpenAI ChatGPT 3.5-Turbo, sentence_transformers, pandas, numpy

## Questions

If you have any questions, please visit [GitHub/Tyler-KD](https://github.com/Tyler-KD), [GitHub/kobrakitty](https://github.com/kobrakitty), [GitHub/JoshHill1](https://github.com/JoshHill1) or submit questions to tyler.kd.knapp@gmail.com, shelbywentz@gmail.com, jjhi11240@gmail.com.