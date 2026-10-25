Mutual Fund Insight Chatbot Using LLMs

Project Overview :

  The Mutual Fund Insight Chatbot is an AI-powered application developed using Python, LangChain, ChromaDB, and Streamlit. It allows users to interact with mutual fund reports and fact sheets by asking questions in natural language.

The system uses Agentic Retrieval-Augmented Generation (Agentic RAG) to retrieve relevant information from uploaded PDF documents and generate context-aware responses using a Large Language Model (LLM).

Objectives:

 To develop an intelligent chatbot that helps users retrieve and understand information from mutual fund documents through natural language queries.

Key Features:

* PDF Document Upload: Upload mutual fund reports and fact sheets.

* Natural Language Q&A: Ask questions about uploaded documents.

* Semantic Search: Retrieve relevant document content using vector embeddings.

* Agentic RAG: Use a LangChain ReAct agent to select and execute document-search tools.

* Context-Aware Responses: Generate answers using retrieved PDF content and an LLM.

* Interactive Interface: Provide a user-friendly chatbot interface using Streamlit.

Technology

1. Python
2. LangChain
3. Azure OpenAI
4. ChromaDB
5. Streamlit
6. pdfplumber
7. Pandas and NumPy

How It Works

* Document Upload: Users upload a mutual fund PDF document through Streamlit.

* Text Extraction: pdfplumber extracts text from the uploaded PDF.

* Text Chunking: LangChain splits the extracted text into smaller chunks.

* Embedding Generation: Azure OpenAI converts text chunks into vector embeddings.

* Vector Storage: ChromaDB stores embeddings and document metadata.

* Semantic Retrieval: The system retrieves relevant content based on the user's question.

* Agent Processing: A LangChain ReAct agent uses the document-search tool when needed.

* Answer Generation: The LLM generates a response using the retrieved information.


