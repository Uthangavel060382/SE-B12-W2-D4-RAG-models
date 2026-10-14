
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# ----------------------------------------
# 1. CONFIGURATION
# ----------------------------------------

load_dotenv()

PDF_DIR = Path("data")
INDEX_DIR = Path("indexes/faiss")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


# ----------------------------------------
# 2. INITIALIZE MODELS
# ----------------------------------------

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )


# ----------------------------------------
# 3. LOAD PDF DOCUMENTS
# ----------------------------------------

def load_documents():
    documents = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            "No PDF files found in data folder."
        )

    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_file.name

        documents.extend(pages)

    return documents


# ----------------------------------------
# 4. SPLIT DOCUMENTS
# ----------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_documents(documents)


# ----------------------------------------
# 5. BUILD LOCAL FAISS DATABASE
# ----------------------------------------

def build_vector_database():

    documents = load_documents()

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "No extractable text found in the PDFs."
        )

    embeddings = get_embeddings()

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_db.save_local(str(INDEX_DIR))

    return len(documents), len(chunks)


# ----------------------------------------
# 6. LOAD EXISTING FAISS DATABASE
# ----------------------------------------

def load_vector_database():

    if not (INDEX_DIR / "index.faiss").exists():
        return None

    embeddings = get_embeddings()

    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )


# ----------------------------------------
# 7. RETRIEVE RELEVANT CHUNKS
# ----------------------------------------

def retrieve_documents(query, vector_db):

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    return retriever.invoke(query)


# ----------------------------------------
# 8. GENERATE RAG ANSWER
# ----------------------------------------

def generate_answer(query, documents):

    context_parts = []

    for doc in documents:
        source = doc.metadata.get(
            "source", "Unknown"
        )

        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"Source: {source}, Page: {page}\n"
            f"Content: {doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful technical assistant.

            Answer using only the provided context.

            If the context does not contain the answer,
            say: "I could not find this information
            in the uploaded documents."

            Be accurate and concise.
            Do not invent information.

            Cite source filenames and page numbers.

            Context:
            {context}
            """
        ),
        (
            "human",
            "{question}"
        )
    ])

    chain = prompt | get_llm()

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response.content


# ----------------------------------------
# 9. STREAMLIT WEB INTERFACE
# ----------------------------------------

def main():

    st.set_page_config(
        page_title="AI RAG Search",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 AI Document Search")
    st.caption(
        "LangChain + FAISS + OpenAI + Streamlit"
    )

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing.")
        st.stop()

    PDF_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # SIDEBAR

    with st.sidebar:

        st.header("Document Management")

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:

            for uploaded_file in uploaded_files:

                safe_name = Path(
                    uploaded_file.name
                ).name

                file_path = PDF_DIR / safe_name

                file_path.write_bytes(
                    uploaded_file.getbuffer()
                )

            st.success(
                f"{len(uploaded_files)} file(s) saved."
            )

        if st.button(
            "Build FAISS Index",
            type="primary"
        ):

            with st.spinner("Indexing documents..."):

                try:
                    pages, chunks = (
                        build_vector_database()
                    )

                    st.success("Index created!")

                    st.metric(
                        "Pages Loaded",
                        pages
                    )

                    st.metric(
                        "Chunks Indexed",
                        chunks
                    )

                except Exception as error:
                    st.error(str(error))

    # MAIN SEARCH AREA

    st.subheader("Ask Questions")

    query = st.text_input(
        "Enter your question",
        placeholder=(
            "Explain the AWS web hosting architecture"
        )
    )

    if st.button("Search Documents"):

        if not query.strip():
            st.warning("Enter a question.")
            return

        with st.spinner("Searching FAISS..."):

            try:

                vector_db = load_vector_database()

                if vector_db is None:
                    st.error(
                        "Build the FAISS index first."
                    )
                    return

                retrieved_docs = retrieve_documents(
                    query,
                    vector_db
                )

                if not retrieved_docs:
                    st.info(
                        "No relevant documents found."
                    )
                    return

                answer = generate_answer(
                    query,
                    retrieved_docs
                )

                st.subheader("AI Answer")
                st.write(answer)

                st.subheader("Retrieved Sources")

                for i, doc in enumerate(
                    retrieved_docs, start=1
                ):

                    source = doc.metadata.get(
                        "source", "Unknown"
                    )

                    page = (
                        doc.metadata.get("page", 0)
                        + 1
                    )

                    with st.expander(
                        f"Source {i}: "
                        f"{source} - Page {page}"
                    ):
                        st.write(doc.page_content)

            except Exception as error:
                st.error(
                    f"Search failed: {error}"
                )


if __name__ == "__main__":
    main()
