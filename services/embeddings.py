# services/embeddings.py
import os
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)

CHROMA_DIR = "data/vector_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize embeddings model
logger.info("Initializing OpenAI embedding model: text-embedding-3-small")
emb = OpenAIEmbeddings(model="text-embedding-3-small")


def index_text(docs, namespace="default"):
    """
    Index a list of {'id','text'} into Chroma namespace.
    Logs all major steps for observability.
    """
    if not docs:
        logger.warning(f"[EMBEDDINGS] No documents provided for namespace '{namespace}'. Skipping indexing.")
        return 0

    logger.info(f"[EMBEDDINGS] Starting indexing for namespace '{namespace}' with {len(docs)} documents.")
    try:
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=emb,
            collection_name=namespace
        )

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_texts, metadatas, ids = [], [], []

        for d in docs:
            if not isinstance(d, dict) or "text" not in d or "id" not in d:
                logger.warning(f"[EMBEDDINGS] Skipping invalid doc entry: {d}")
                continue
            chunks = splitter.split_text(d["text"])
            for i, ch in enumerate(chunks):
                all_texts.append(ch)
                ids.append(f"{d['id']}_chunk_{i}")
                metadatas.append({"source": d["id"]})

        if not all_texts:
            logger.warning(f"[EMBEDDINGS] No text chunks generated for namespace '{namespace}'.")
            return 0

        vectordb.add_texts(texts=all_texts, metadatas=metadatas, ids=ids)
        #vectordb.persist()
        logger.info(f"[EMBEDDINGS] Successfully indexed {len(all_texts)} text chunks into '{namespace}'.")
        return len(all_texts)

    except Exception as e:
        logger.error(f"[EMBEDDINGS] Error while indexing namespace '{namespace}': {e}", exc_info=True)
        return 0


def get_retrievers():
    """
    Returns retrievers for all namespaces (news, social, misc).
    RCA agents can call each sequentially.
    Logs which retrievers were successfully initialized.
    """
    retrievers = {}
    try:
        for ns in ["news_data", "social_data", "misc_data"]:
            try:
                vectordb = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=emb,
                    collection_name=ns
                )
                retrievers[ns] = vectordb.as_retriever(search_kwargs={"k": 5})
                logger.info(f"[RETRIEVER] Initialized retriever for namespace '{ns}'.")
            except Exception as e:
                logger.warning(f"[RETRIEVER] Failed to initialize retriever for '{ns}': {e}")
        logger.info(f"[RETRIEVER] Total retrievers ready: {len(retrievers)}")
    except Exception as e:
        logger.error(f"[RETRIEVER] Error building retrievers: {e}", exc_info=True)

    return retrievers
