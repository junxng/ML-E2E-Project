from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams
import os
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

_vector_store = None
_client = None
_embeddings = None

def get_qdrant_client():
    """
    Get or initialize the Qdrant client
    """
    global _client
    if _client is None:
        try:
            logger.info(f"Connecting to Qdrant at http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            _client = QdrantClient(
                url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}", 
                timeout=10.0
            )

            _client.get_collections()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise RuntimeError(f"Could not connect to Qdrant: {str(e)}")
    return _client

def get_embeddings():
    """
    Get embeddings model with consistent dimensions
    """
    global _embeddings
    
    if _embeddings is None:
        try:
            logger.info("Initializing OpenAI embeddings model")
                
            _embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.OPENAI_API_KEY,
                dimensions=1536,
                request_timeout=15.0
            )
            
            logger.info("OpenAI embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings model: {str(e)}")
            raise RuntimeError(f"OpenAI API error: {str(e)}. Check your API key.")
    
    return _embeddings

def get_vector_store():
    """
    Get or create the vector store using Qdrant with consistent dimensions
    """
    global _vector_store
    
    if _vector_store is None:
        try:
            logger.info("Setting up vector store")
                
            embeddings = get_embeddings()
            client = get_qdrant_client()
            collection_name = settings.QDRANT_COLLECTION_NAME

            try:
                logger.info(f"Checking if collection {collection_name} exists")
                collections = client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if collection_name in collection_names:
                    logger.info(f"Collection {collection_name} exists, checking vector dimensions")
                    collection_info = client.get_collection(collection_name=collection_name)
                    current_dim = collection_info.config.params.vectors.size

                    if current_dim != 1536:
                        logger.info(f"Recreating collection {collection_name} with correct dimensions (1536)")
                        client.delete_collection(collection_name=collection_name)
                        client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=1536,
                                distance=Distance.COSINE
                            ),
                        )
                        logger.info(f"Collection {collection_name} recreated successfully")
                else:
                    logger.info(f"Creating new collection {collection_name}")
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=1536,
                            distance=Distance.COSINE
                        ),
                    )
                    logger.info(f"Collection {collection_name} created successfully")
 
                logger.info("Creating Qdrant vector store instance")
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    _vector_store = Qdrant(
                        client=client,
                        collection_name=collection_name,
                        embeddings=embeddings,
                    )
                    
                logger.info("Qdrant vector store instance created successfully")
                
            except UnexpectedResponse as e:
                logger.error(f"Qdrant API error: {str(e)}")
                raise RuntimeError(f"Qdrant API error: {str(e)}")
            
        except Exception as e:
            logger.exception(f"Failed to initialize vector store: {str(e)}")
            raise RuntimeError(f"Vector store initialization error: {str(e)}")
    
    return _vector_store
