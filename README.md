# Document AI Chatbot

A PDF question-answering application built with Streamlit,
Google Gemini, embeddings, and FAISS vector search.

## Features

- Upload a PDF document
- Extract and split document text into chunks
- Generate embeddings for semantic search
- Retrieve relevant passages using FAISS
- Generate answers with Google Gemini
- Display source references

## Technology Stack

- Python
- Streamlit
- Google Gemini API
- Google GenAI SDK
- FAISS
- PyPDF

## Run Locally

1. Install the dependencies:

   pip install -r requirements.txt

2. Configure the GEMINI_API_KEY environment variable.

3. Start the application:

   streamlit run app.py

## Security

Never commit API keys, authentication tokens,
private documents, or secret configuration files.

## Project Status

Educational prototype. Further testing and deployment
hardening are recommended before production use.
