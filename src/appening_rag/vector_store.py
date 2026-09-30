from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from appening_rag.config import Settings
from appening_rag.embeddings import FeatureHashEmbeddings


def create_vector_store(settings: Settings) -> VectorStore:
    settings.validate_runtime()
    if settings.vector_store_backend == "local":
        from langchain_chroma import Chroma

        return Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=FeatureHashEmbeddings(),
            persist_directory=str(settings.chroma_persist_directory),
        )

    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    embeddings: Embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        namespace=settings.pinecone_namespace,
    )
