import argparse
from pathlib import Path
from urllib.request import Request, urlopen

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from appening_rag.config import get_settings
from appening_rag.vector_store import create_vector_store


def download_pdf(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "AppeningRAG/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        output.write(response.read())
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Downloaded PDF is empty: {url}")
    return destination


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf_path), "page": page_number},
                )
            )
    if not pages:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")
    return pages


def split_pages(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index
    return chunks


def ingest_pdf(
    vector_store: VectorStore,
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    url: str | None = None,
) -> int:
    if not pdf_path.exists():
        if not url:
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        download_pdf(url, pdf_path)
    chunks = split_pages(load_pdf_pages(pdf_path), chunk_size, chunk_overlap)
    vector_store.add_documents(chunks)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into the configured vector store")
    parser.add_argument("--force-download", action="store_true", help="Download the configured PDF again")
    args = parser.parse_args()

    settings = get_settings()
    if args.force_download:
        settings.pdf_path.unlink(missing_ok=True)
    store = create_vector_store(settings)
    count = ingest_pdf(
        store,
        settings.pdf_path,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.pdf_url,
    )
    print(f"Indexed {count} chunks from {settings.pdf_path}")


if __name__ == "__main__":
    main()
