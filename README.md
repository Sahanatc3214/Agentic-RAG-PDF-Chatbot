                                Mutual Fund Insight Chatbot Using LLMs


<img width="854" height="479" alt="Screenshot 2026-10-25 192910" src="https://github.com/user-attachments/assets/55d1933f-2572-4f77-816e-30a5fc3ac17c" />






[Mutual_Fund_Monthly_Report_November_2025.pdf](https://github.com/user-attachments/files/33922373/Mutual_Fund_Monthly_Report_November_2025.pdf)




<img width="926" height="504" alt="Screenshot 2026-10-25 192744" src="https://github.com/user-attachments/assets/b981e30d-c9a8-4f3c-830d-2ca42d5f862b" />





Project Overview :

* The Mutual Fund Insight Chatbot is an AI-powered application developed using Python, LangChain, ChromaDB, and Streamlit. It allows users to interact with mutual fund reports and fact sheets by asking questions in natural language.

* The system uses Agentic Retrieval-Augmented Generation (Agentic RAG) to retrieve relevant information from uploaded PDF documents and generate context-aware responses using a Large Language Model (LLM).

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


