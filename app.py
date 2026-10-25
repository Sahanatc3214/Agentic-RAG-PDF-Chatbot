import pdfplumber 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
import os
from dotenv import load_dotenv
load_dotenv()


def extract_for_rag(pdf_file):
    documents = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text(
                layout=False,
                x_tolerance=2
            ) or ""

            if text.strip():
                documents.append({
                    "page": page.page_number,
                    "text": text
                })

    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for doc in documents:
        splits = text_splitter.split_text(doc["text"])

        for i in splits:
            chunks.append({
                "page": doc["page"],
                "text": i
            })

    return chunks


def get_vectorstore(chunks):
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_deployment="text-embedding-3-small",
        chunk_size=2
    )

    texts = [c["text"] for c in chunks]
    metadatas = [{'page': c['page']} for c in chunks]

    # Remove duplicate chunks
    unique = list(dict.fromkeys(
        (c["page"], c["text"]) for c in chunks
    ))

    texts = [text for page, text in unique]
    metadatas = [{"page": page} for page, text in unique]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore


if __name__ == "__main__":
    st.title("PDF Chatbot using AI Agent")
    pdf_docs = st.file_uploader("Upload your PDF",
                                type="pdf",
                                )
    if pdf_docs is not None:
        import hashlib

        pdf_hash = hashlib.sha256(
            pdf_docs.getvalue()
        ).hexdigest()

        if st.session_state.get("pdf_hash") != pdf_hash:
            raw_text = extract_for_rag(pdf_docs)
            text_chunks = get_text_chunks(raw_text)

            st.session_state.pop("vectorstore", None)
            st.session_state.pdf_pages = raw_text

            if text_chunks:
                st.session_state.vectorstore = get_vectorstore(
                    text_chunks
                )

            st.session_state.pdf_hash = pdf_hash
REACT_PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers questions
from uploaded PDF reports.

Available tools:
{tools}

Tool names: {tool_names}

Rules:
1. For general questions, answer directly.
2. For PDF questions, use search_pdf_content.
3. For numerical questions, identify the correct
   row and column before answering.
4. The "No. of Schemes" column represents the
   number of schemes, not the number of folios.
5. When a question asks for a total, find all
   relevant subtotals before calculating.
6. Do not count the number of listed categories
   as the number of schemes.
7. PDF extraction may insert spaces inside numbers.
   For example, "5 44" can represent 544 when
   verified against the table structure.
8. Do not confuse open-ended, close-ended,
   interval, debt, and equity schemes.
9. Never guess missing numbers.
10. If the retrieved information is incomplete,
    search again using a more specific query.
11. Mention the page number in the final answer.
12. Always follow the ReAct format.

For general questions:
Thought: This is a general question.
Final Answer: Your answer.

For PDF questions:
Thought: I need to search the PDF.
Action: search_pdf_content
Action Input: relevant search query

After receiving results, decide whether another
search is required.

When sufficient information is available:
Thought: I have verified the relevant table values.
Final Answer: Your answer with calculation and page.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
@tool
def search_pdf_content(query: str) -> str:
    """Search uploaded PDF for relevant information."""

    if "vectorstore" not in st.session_state:
        return "Please upload a PDF first."

    vectorstore = st.session_state.vectorstore

    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=12,
        fetch_k=40,
        lambda_mult=0.5
    )

    if not docs:
        return "No relevant information found."

    results = []
    seen = set()

    for doc in docs:
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()

        key = (page, content)

        if key in seen:
            continue

        seen.add(key)

        results.append(
            f"Page: {page}\nContent: {content}"
        )

    return "\n\n".join(results)

@tool
def read_pdf_page(page_number: int) -> str:
    """Read the complete text of a specific PDF page.
    Useful for numerical questions, tables, and subtotals.
    """

    if "pdf_pages" not in st.session_state:
        return "Please upload a PDF first."

    for page in st.session_state.pdf_pages:
        if page["page"] == page_number:
            return (
                f"Page: {page_number}\n"
                f"Content:\n{page['text']}"
            )

    return f"Page {page_number} not found."


llm = AzureChatOpenAI(
     model_name="gpt-4.1-nano",
     azure_deployment="gpt41nano",
     azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
     openai_api_key=os.getenv('OPENAI_API_KEY'),
     api_version=os.getenv('AZURE_OPENAI_API_VERSION')
     )

prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
tools = [search_pdf_content, read_pdf_page]
agent = create_react_agent(
    llm,
    tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=7,
    max_execution_time=90,
    handle_parsing_errors=True
)


input_text = st.chat_input(
    "Ask any question",
    key="pdf_chat_input"
)

if input_text is not None and input_text.strip():

    st.write("Your Question:", input_text)

    with st.spinner("Generating answer..."):
        try:
            response = agent_executor.invoke({
                "input": input_text.strip()
            })

            st.write("Agent Output:", response["output"])

        except Exception as e:
            st.error(f"Error: {str(e)}")

