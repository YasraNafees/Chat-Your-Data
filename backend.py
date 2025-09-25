import os
import json
import hashlib
from dotenv import load_dotenv
from pathlib import Path

from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Download HF model weights
hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="pytorch_model.bin")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
api_base = os.getenv("OPENROUTER_API_BASE")



PDF_PATH = "AI Foundations of Computational Agents 3rd Ed.pdf"
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ----------------- Helper Functions -----------------

def load_pdf(path: str):
    return  PyMuPDFLoader(path).load()

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def build_vectorstore(splits, embed_model_name: str):
    emb = HuggingFaceEmbeddings(model_name=embed_model_name)
    return FAISS.from_documents(splits, emb)

def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {'sha256': h.hexdigest(), "size": p.stat().st_size, 'mtime': int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1"
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

def load_index_run(index_dir: Path, embed_model_name: str):
    emb = HuggingFaceEmbeddings(model_name=embed_model_name)
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )

def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits, embed_model_name)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
    }, indent=2))
    return vs

def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    force_rebuild: bool = False
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        return load_index_run(index_dir, embed_model_name)
    else:
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

# ----------------- LLM Setup -----------------

llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base=api_base,
    model="deepseek/deepseek-chat-v3.1:free",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def setup_pipeline(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    force_rebuild: bool = False
):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild
    )

def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    force_rebuild: bool = False
):
    vectorstore = setup_pipeline(pdf_path, chunk_size, chunk_overlap, embed_model_name, force_rebuild)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | llm
    return chain.stream(
        question
    )
