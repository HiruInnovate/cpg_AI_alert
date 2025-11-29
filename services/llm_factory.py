import os

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from services.logger_config import get_logger

client = httpx.Client(verify=False)


# -------------------------------------------------
#  LOAD ENV AND LOGGER
# -------------------------------------------------
load_dotenv()
logger = get_logger(__name__)

# -------------------------------------------------
#  FACTORY CONFIG
# -------------------------------------------------
USE_AZURE_OPENAI = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://genailab.tcs.in")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your_azure_api_key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")

# ✅ Ensure local TikToken cache
TIKTOKEN_CACHE_DIR = os.getenv("TIKTOKEN_CACHE_DIR", "./tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR


# -------------------------------------------------
#  CHAT MODEL FACTORY
# -------------------------------------------------
def create_chat_model(model: str = 'gpt-4o-mini', temperature: float = 0.2, client: httpx.Client = None,
                      is_agent: bool = False):
    """Create and return a ChatOpenAI model for either Azure or OpenAI.

    Args:
        model (str): Model name (if None, use env default)
        temperature (float): Temperature for creativity control
        client (httpx.Client): Optional custom client for SSL bypass or logging
        is_agent (bool): for indicating it is agent or not

    Returns:
        ChatOpenAI: Configured chat LLM client
    """
    if USE_AZURE_OPENAI:
        base_url = AZURE_ENDPOINT
        api_key = AZURE_API_KEY
        model_name = model or os.getenv("AZURE_CHAT_MODEL", "azure/genailab-maas-gpt-35-turbo")
        logger.info(f"[LLM_FACTORY] Using Azure OpenAI chat model: {model_name}")
    else:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = OPENAI_API_KEY
        model_name = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        logger.info(f"[LLM_FACTORY] Using OpenAI chat model: {model_name}")

    if is_agent:
        return ChatOpenAI(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            stop=["\nObservation", "Observation"],
            http_client=client or httpx.Client(verify=False)
        )

    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        http_client=client or httpx.Client(verify=False),
        max_tokens=2048
    )


# -------------------------------------------------
#  EMBEDDING MODEL FACTORY
# -------------------------------------------------
def create_embedding_model(model: str = None):
    """Create and return an OpenAIEmbeddings model for either Azure or OpenAI.

    Args:
        model (str): Model name (if None, uses default env vars)

    Returns:
        OpenAIEmbeddings: Configured embedding model instance
    """
    if USE_AZURE_OPENAI:
        base_url = AZURE_ENDPOINT
        api_key = AZURE_API_KEY
        model_name = model or os.getenv("AZURE_EMBED_MODEL", "azure/genailab-maas-text-embedding-3-large")
        logger.info(f"[LLM_FACTORY] Using Azure OpenAI embedding model: {model_name}")
    else:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = OPENAI_API_KEY
        model_name = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        logger.info(f"[LLM_FACTORY] Using OpenAI embedding model: {model_name}")

    return OpenAIEmbeddings(
        base_url=base_url,
        model=model_name,
        api_key=api_key,
        http_client=client
    )


# -------------------------------------------------
#  TEST UTILITIES (Optional)
# -------------------------------------------------
def test_llm_factory():
    """Quick self-test to verify configuration and connectivity with actual LLM call."""
    try:
        chat = create_chat_model()
        logger.info(f"✅ Chat model ready: {chat}")

        # Send a simple prompt to verify the LLM works
        response = chat.invoke("Hello! This is a connection test from the CPG AI system.")
        logger.info(f"🤖 LLM Response: {response.content if hasattr(response, 'content') else response}")

        emb = create_embedding_model()
        logger.info(f"✅ Embedding model ready: {emb}")

        return True
    except Exception as e:
        logger.error(f"❌ LLM Factory Test Failed: {e}", exc_info=True)
        return False
