import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

# LangChain components (same structure)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Document loaders (files only)
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.schema import Document

# ---- Load .env if present
load_dotenv()
st.set_page_config(page_title="RAG Document Q&A — OpenAI", page_icon="📄")

# ----------------------------
# Sidebar: OpenAI settings + status
# ----------------------------
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
        help="e.g., gpt-4o-mini, gpt-4o, gpt-4.1"
    )
    embed_model = st.text_input(
        "Embedding Model",
        value=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    )

    st.markdown("---")
    if st.session_state.get("vectors_ready"):
        st.success("✅ Vector Database is ready")
    else:
        st.info("ℹ️ Vector DB will be built on your first query")

# Make sure OpenAI clients see the key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

st.title("RAG Document Q&A — OpenAI Only")

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
                # Fallback for docx/pptx/etc. (requires unstructured deps if used)
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

def ensure_vectors_built(uploaded_files, embed_model_name: str):
    """
    Build vector store if it doesn't exist yet. Called automatically on first query.
    """
    if "vectors" in st.session_state:
        return True  # already built

    if not uploaded_files:
        # No files uploaded; can't build
        return False

    # Build fresh from current uploads
    temp_dir, paths = save_uploads_to_temp_dir(uploaded_files)
    raw_docs = load_docs_from_paths(paths)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(model=embed_model_name)
    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
    st.session_state.text_splitter = text_splitter
    st.session_state.final_documents = final_docs
    st.session_state.vectors_ready = True
    return True

# ----------------------------
# Uploads (no URL option)
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload documents (PDF/TXT/other common formats). Multiple allowed.",
    type=None,
    accept_multiple_files=True
)

# ----------------------------
# Chat state & rendering
# ----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role": "user"/"assistant", "content": str}

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ----------------------------
# Chat input & retrieval
# ----------------------------
user_prompt = st.chat_input("Ask about your uploaded documents...")
if user_prompt:
    # Append user message and render immediately
    st.session_state.chat.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Ensure vectors exist (auto-build on first query)
    vectors_ok = ensure_vectors_built(uploaded_files, embed_model)
    with st.chat_message("assistant"):
        if not vectors_ok:
            st.warning("Please upload at least one document, then ask your question again.")
            assistant_text = "I need documents to build the vector index. Upload files first."
            st.session_state.chat.append({"role": "assistant", "content": assistant_text})
        else:
            llm = ChatOpenAI(model=model_name, temperature=0)
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            elapsed = time.process_time() - start

            answer = response.get('answer', '').strip() or "_No answer returned._"
            st.markdown(answer)
            st.caption(f"_Response time: {elapsed:.2f}s_")
            st.session_state.chat.append({"role": "assistant", "content": answer})

            with st.expander("Document similarity Search"):
                for doc in response.get('context', []):
                    st.write(doc.page_content)
                    st.write('------------------------')

    # --- Auto-scroll to bottom after each turn
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        const bottom = document.createElement('div');
        bottom.id = '___bottom';
        document.body.appendChild(bottom);
        bottom.scrollIntoView({behavior: 'smooth', block: 'end'});
        </script>
        """,
        height=0,
    )
