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
# Sidebar: OpenAI settings
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

# ----------------------------
# Uploads & Build vectors (same flow)
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload documents (PDF/TXT/other common formats). Multiple allowed.",
    type=None,
    accept_multiple_files=True
)

colA, colB = st.columns([1, 2])
with colA:
    if st.button("Document Embedding"):
        if not uploaded_files:
            st.warning("Please upload at least one document before embedding.")
        else:
            # Build embeddings + vectors (same structure)
            # Always rebuild when pressed to reflect new uploads
            temp_dir, paths = save_uploads_to_temp_dir(uploaded_files)
            raw_docs = load_docs_from_paths(paths)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_docs = text_splitter.split_documents(raw_docs)

            embeddings = OpenAIEmbeddings(model=embed_model)
            st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
            st.session_state.text_splitter = text_splitter
            st.session_state.final_documents = final_docs

            st.success("Vector Database is ready")

with colB:
    st.caption("After embedding, ask questions in the chat below.")

# ----------------------------
# Chat state & rendering
# ----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role": "user"/"assistant", "content": str}

# Render history
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

    # Answer
    with st.chat_message("assistant"):
        if "vectors" not in st.session_state:
            st.warning("Please click 'Document Embedding' after adding documents, then ask your question.")
            assistant_text = "I need the vector index first. Please embed your documents."
            st.session_state.chat.append({"role": "assistant", "content": assistant_text})
        else:
            llm = ChatOpenAI(model=model_name, temperature=0)
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            elapsed = time.process_time() - start

            answer = response.get('answer', '').strip()
            if not answer:
                answer = "_No answer returned._"

            st.markdown(answer)
            st.caption(f"_Response time: {elapsed:.2f}s_")
            st.session_state.chat.append({"role": "assistant", "content": answer})

            # Similarity context for the latest answer
            with st.expander("Document similarity Search"):
                ctx = response.get('context', [])
                for i, doc in enumerate(ctx):
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
