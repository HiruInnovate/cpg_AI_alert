import os
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# -------------------------------------------------
#  CONFIGURABLE SETTINGS
# -------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/vector_db")
USE_AZURE_OPENAI = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

# ✅ Ensure vector DB directory exists
os.makedirs(CHROMA_DIR, exist_ok=True)

# ✅ TikToken Cache
TIKTOKEN_CACHE_DIR = os.getenv("TIKTOKEN_CACHE_DIR", "./tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR

# -------------------------------------------------
#  EMBEDDING INITIALIZATION FACTORY
# -------------------------------------------------
def create_embedding_model():
    """Create and return an embedding model instance based on environment config.

    Supports both Azure OpenAI and standard OpenAI setups using env vars:
    - USE_AZURE_OPENAI: toggles Azure mode
    - AZURE_OPENAI_ENDPOINT: full base URL (e.g., https://genailab.tcs.in)
    - AZURE_OPENAI_API_KEY: API key for Azure endpoint
    - AZURE_EMBED_MODEL: embedding model name for Azure
    - OPENAI_API_KEY: API key for regular OpenAI
    - OPENAI_EMBED_MODEL: model name for regular OpenAI
    - TIKTOKEN_CACHE_DIR: cache location for tokenizer
    """
    try:
        if USE_AZURE_OPENAI:
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "https://genailab.tcs.in")
            api_key = os.getenv("AZURE_OPENAI_API_KEY", "your_azure_api_key")
            model = os.getenv("AZURE_EMBED_MODEL", "azure/genailab-maas-text-embedding-3-large")
            logger.info(f"[EMBEDDINGS] Using Azure OpenAI embedding model: {model}")
        else:
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
            model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            logger.info(f"[EMBEDDINGS] Using OpenAI embedding model: {model}")

        return OpenAIEmbeddings(
            base_url=base_url,
            model=model,
            api_key=api_key
        )
    except Exception as e:
        logger.error(f"[EMBEDDINGS] Failed to initialize embeddings: {e}")
        raise e


# Global embedding model instance
emb = create_embedding_model()

# -------------------------------------------------
#  TEXT INDEXING AND RETRIEVAL
# -------------------------------------------------
def index_text(docs, namespace="default"):
    """Index a list of {'id','text'} documents into a Chroma namespace.

    Steps:
      - Split text into overlapping chunks
      - Create vector embeddings
      - Store them persistently in Chroma under the given namespace

    Args:
        docs (list[dict]): List of documents containing 'id' and 'text'
        namespace (str): Chroma namespace to store the embeddings under
    Returns:
        int: Number of text chunks indexed
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
                metadatas.append({"source": d['id']})

        if not all_texts:
            logger.warning(f"[EMBEDDINGS] No text chunks generated for namespace '{namespace}'.")
            return 0

        vectordb.add_texts(texts=all_texts, metadatas=metadatas, ids=ids)
        logger.info(f"[EMBEDDINGS] Successfully indexed {len(all_texts)} text chunks into '{namespace}'.")
        return len(all_texts)

    except Exception as e:
        logger.error(f"[EMBEDDINGS] Error while indexing namespace '{namespace}': {e}", exc_info=True)
        return 0


def get_retrievers():
    """Return Chroma retrievers for all default namespaces (news, social, misc).

    Returns:
        dict: A dictionary mapping namespace → retriever instance.
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
