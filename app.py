import streamlit as st
import os
import time
import tempfile

# LangChain components (same structure)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    WebBaseLoader,
)
from langchain.schema import Document



# ----------------------------
# Streamlit UI
# ----------------------------
st.title("RAG Document Q&A — OpenAI Only")

with st.sidebar:
    st.subheader("🔑 OpenAI Settings")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    model_name = st.text_input(
        "Chat Model",
        value=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        help="e.g., gpt-4o-mini, gpt-4o, gpt-4.1, gpt-3.5-turbo"
    )
    embed_model = st.text_input(
        "Embedding Model",
        value=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    )

# Make sure LangChain/OpenAI clients see the key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# ----------------------------
# Prompt (unchanged structure)
# ----------------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

# ----------------------------
# Helpers (file saving & loading)
# ----------------------------
def save_uploads_to_temp_dir(uploaded_files):
    temp_dir = tempfile.mkdtemp(prefix="rag_uploads_")
    paths = []
    for f in uploaded_files:
        out_path = os.path.join(temp_dir, f.name)
        with open(out_path, "wb") as w:
            w.write(f.read())
        paths.append(out_path)
    return temp_dir, paths

def load_docs_from_paths(file_paths):
    docs = []
    for p in file_paths:
        p_lower = p.lower()
        try:
            if p_lower.endswith(".pdf"):
                loader = PyPDFLoader(p)
                docs.extend(loader.load())
            elif p_lower.endswith(".txt"):
                loader = TextLoader(p, encoding="utf-8")
                docs.extend(loader.load())
            else:
                # Fallback for docx, pptx, etc. (requires unstructured deps if used)
                loader = UnstructuredFileLoader(p)
                docs.extend(loader.load())
        except Exception as e:
            docs.append(
                Document(
                    page_content=f"[[Failed to load {os.path.basename(p)}: {e}]]",
                    metadata={"source": p},
                )
            )
    return docs

def load_docs_from_urls(url_list):
    urls = [u.strip() for u in url_list.splitlines() if u.strip()]
    if not urls:
        return []
    loader = WebBaseLoader(urls)
    return loader.load()

# ----------------------------
# Uploads & Links (same flow)
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload documents (PDF/TXT/other common formats). Multiple allowed.",
    type=None,
    accept_multiple_files=True
)

url_block = st.text_area(
    "Or paste one or more URLs (one per line) to include as context",
    value="",
    height=100
)

if st.button("Document Embedding"):
    if not uploaded_files and not url_block.strip():
        st.warning("Please upload at least one document or provide at least one URL before embedding.")
    else:
        if "vectors" not in st.session_state:
            # Load from uploads
            all_docs = []
            if uploaded_files:
                temp_dir, paths = save_uploads_to_temp_dir(uploaded_files)
                all_docs.extend(load_docs_from_paths(paths))

            # Load from URLs
            if url_block.strip():
                try:
                    all_docs.extend(load_docs_from_urls(url_block))
                except Exception as e:
                    st.warning(f"Failed to load one or more URLs: {e}")

            # Split (same params)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_docs = text_splitter.split_documents(all_docs)

            # Embeddings (OpenAI only)
            embeddings = OpenAIEmbeddings(model=embed_model)

            # Vector store
            st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
            st.session_state.text_splitter = text_splitter
            st.session_state.final_documents = final_docs

        st.write("Vector Database is ready")

# ----------------------------
# Query (same chain structure)
# ----------------------------
user_prompt = st.text_input("Enter your query from the uploaded documents / URLs")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' after adding documents/links, then ask your question.")
    else:
        llm = ChatOpenAI(model=model_name, temperature=0)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        print(f"Response time : {time.process_time() - start:.2f}s")  # same pattern as your original

        # Main answer
        st.write(response['answer'])

        # Similarity context (same expander & separator)
        with st.expander("Document similarity Search"):
            ctx = response.get('context', [])
            for i, doc in enumerate(ctx):
                st.write(doc.page_content)
                st.write('------------------------')
