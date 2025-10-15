import streamlit as st
import time
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.schema import Document

st.set_page_config(page_title="RAG Document Q&A — OpenAI", page_icon="📄")
st.title("RAG Document Q&A — OpenAI Only")

# --------- API key required every run (no env, no session) ----------
with st.sidebar:
    st.subheader("🔑 OpenAI Settings (not stored)")
    user_api_key = st.text_input("OpenAI API Key", type="password", value="")
    model_name = st.text_input("Chat Model", value="gpt-4o-mini")
    embed_model = st.text_input("Embedding Model", value="text-embedding-3-small")
    st.markdown("---")
    if st.session_state.get("vectors_ready"):
        st.success("✅ Vector Database is ready")
    else:
        st.info("ℹ️ Vector DB will be built on your first query")

# If no key, do not proceed further (prevents accidental leaks)
if not user_api_key:
    st.warning("Enter your OpenAI API key in the sidebar to use the app. The key is NOT stored.")
    st.stop()

# --------- Prompt (same as before) ----------
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

# --------- File helpers ----------
def save_uploads_to_temp_dir(uploaded_files):
    import os
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
                loader = UnstructuredFileLoader(p)  # docx/pptx/etc. if deps available
                docs.extend(loader.load())
        except Exception as e:
            docs.append(
                Document(
                    page_content=f"[[Failed to load {p}: {e}]]",
                    metadata={"source": p},
                )
            )
    return docs

def ensure_vectors_built(uploaded_files, embed_model_name: str, api_key: str):
    """Auto-build vector store on first query. Never stores the API key."""
    if "vectors" in st.session_state:
        return True
    if not uploaded_files:
        return False

    temp_dir, paths = save_uploads_to_temp_dir(uploaded_files)
    raw_docs = load_docs_from_paths(paths)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(raw_docs)

    # Pass api_key directly (no env/session storage)
    embeddings = OpenAIEmbeddings(model=embed_model_name, api_key=api_key)
    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
    st.session_state.text_splitter = text_splitter
    st.session_state.final_documents = final_docs
    st.session_state.vectors_ready = True
    return True

# --------- Uploads (no URL option) ----------
uploaded_files = st.file_uploader(
    "Upload documents (PDF/TXT/other common formats). Multiple allowed.",
    type=None,
    accept_multiple_files=True
)

# --------- Chat state ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --------- Chat input & retrieval ----------
user_prompt = st.chat_input("Ask about your uploaded documents...")
if user_prompt:
    st.session_state.chat.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    vectors_ok = ensure_vectors_built(uploaded_files, embed_model, user_api_key)
    with st.chat_message("assistant"):
        if not vectors_ok:
            msg = "Please upload at least one document, then ask your question again."
            st.warning(msg)
            st.session_state.chat.append({"role": "assistant", "content": msg})
        else:
            # Pass api_key directly to ChatOpenAI (no env/session)
            llm = ChatOpenAI(model=model_name, temperature=0, api_key=user_api_key)
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

    # Auto-scroll to bottom
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
