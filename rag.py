from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

CHROMA_DIR = "chroma_store"

load_dotenv(Path(__file__).with_name(".env"))

EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def build_library(docs_dir="data/docs"):
    # Imported here, not at module scope: langchain_text_splitters' __init__ eagerly
    # imports SentenceTransformersTokenTextSplitter, which pulls torch (+868 MB RSS).
    # Only the build step needs a splitter; serving the app does not.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Read the files directly rather than via langchain_community's DirectoryLoader:
    # importing langchain_community.document_loaders drags in torch (+741 MB RSS).
    docs = [
        Document(page_content=p.read_text(), metadata={"source": str(p)})
        for p in sorted(Path(docs_dir).glob("**/*")) if p.is_file()
    ]
    if not docs:
        raise SystemExit(f"No documents found in {docs_dir!r} - add some .txt/.md files and re-run.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    Chroma.from_documents(chunks, EMBEDDINGS, persist_directory=CHROMA_DIR)  # saves to disk
    print(f"Indexed {len(chunks)} chunks into ChromaDB")


def get_retriever(k=4):
    store = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)
    return store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    build_library()   # run once: python rag.py