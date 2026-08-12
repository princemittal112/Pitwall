import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

os.makedirs("data", exist_ok=True)
os.makedirs("cache/faiss", exist_ok=True)


def load_and_chunk_regulations(pdf_path: str) -> list:
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", "Article", " "],
    )
    chunks = splitter.split_documents(pages)
    print(f"  Split into {len(chunks)} chunks")
    return chunks


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def build_faiss_store(chunks: list) -> FAISS:
    print("Building FAISS vector store...")
    store = FAISS.from_documents(chunks, _get_embeddings())
    store.save_local("cache/faiss")
    print("  FAISS store built and saved to cache/faiss")
    return store


def load_faiss_store() -> FAISS:
    return FAISS.load_local(
        "cache/faiss",
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def get_or_build_faiss_store(chunks: list) -> FAISS:
    """Loads from cache if available, builds and saves if not."""
    if os.path.exists("cache/faiss/index.faiss"):
        print("Loading FAISS store from cache...")
        return load_faiss_store()
    return build_faiss_store(chunks)


def build_ensemble_retriever(chunks: list, faiss_store: FAISS) -> EnsembleRetriever:
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 4})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.6, 0.4],
    )


def query_regulations(retriever: EnsembleRetriever, query: str) -> None:
    print(f"\nQuery: '{query}'")
    print("-" * 60)
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        print(f"\n[Result {i+1} — Page {page}]")
        print(doc.page_content[:400])
        print("...")


if __name__ == "__main__":
    chunks = load_and_chunk_regulations("data/fia_regulations.pdf")
    faiss_store = get_or_build_faiss_store(chunks)
    retriever = build_ensemble_retriever(chunks, faiss_store)

    query_regulations(retriever, "mandatory tyre compound rule during the race")
    query_regulations(retriever, "Article 28.6")
    query_regulations(retriever, "pit stop during safety car period")