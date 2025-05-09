import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.rag.embeddings import get_vector_store

# Get logger but don't configure it (main module handles configuration)
logger = logging.getLogger(__name__)

def get_knowledge_base() -> List[Dict[str, Any]]:
    """
    Get the knowledge base entries (list of processed PDFs)
    """
    if not os.path.exists(settings.KNOWLEDGE_BASE_FILE):
        return []
    
    try:
        with open(settings.KNOWLEDGE_BASE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_knowledge_base(knowledge_base: List[Dict[str, Any]]) -> None:
    """
    Save the knowledge base entries to file
    """
    with open(settings.KNOWLEDGE_BASE_FILE, 'w') as f:
        json.dump(knowledge_base, f)

def add_to_knowledge_base(file_path: str) -> Dict[str, Any]:
    """
    Add a processed PDF to the knowledge base
    """
    # Get file info
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    date_added = datetime.now().isoformat()
    
    # Create entry
    entry = {
        "id": abs(hash(file_path + date_added)) % (10**10),  # Simple unique ID
        "file_name": file_name,
        "file_path": file_path,
        "file_size_bytes": file_size,
        "date_added": date_added
    }
    
    # Add to knowledge base
    knowledge_base = get_knowledge_base()
    knowledge_base.append(entry)
    save_knowledge_base(knowledge_base)
    
    return entry

def process_pdf(file_path: str) -> Dict[str, Any]:
    """
    Process a PDF file and add it to the vector store
    Returns a dictionary with success status and error message if any
    """
    result = {"success": False, "error": None}
    
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            result["error"] = f"File not found: {file_path}"
            return result
        
        logger.info(f"Processing PDF: {file_path}")
        
        # Load PDF with error handling
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
        except Exception as e:
            logger.error(f"Failed to read PDF file: {str(e)}")
            result["error"] = f"Failed to read PDF file: {str(e)}"
            return result
        
        if not documents:
            result["error"] = "PDF loaded but no content extracted"
            return result
            
        # Add metadata to documents
        file_name = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["source"] = file_name
            if "page" in doc.metadata:
                doc.metadata["page_number"] = doc.metadata.get("page")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks to reduce potential embedding errors
            chunk_overlap=150
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(splits)} chunks")
        
        if not splits:
            result["error"] = "No content extracted from PDF to embed"
            return result
        
        # Process in small batches with error handling
        success = False
        try:
            vector_store = get_vector_store()
            
            # Process in smaller batches to avoid errors
            batch_size = 5
            logger.info(f"Adding {len(splits)} chunks to vector store in batches of {batch_size}")
            
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i+batch_size]
                batch_end = min(i+batch_size, len(splits))
                logger.debug(f"Processing batch {i//batch_size + 1}, chunks {i+1}-{batch_end}")
                vector_store.add_documents(documents=batch)
            
            success = True
            logger.info("Successfully added all chunks to vector store")
                
        except Exception as e:
            # In case of failure, try recreating the vector store
            if not success:
                error_msg = str(e)
                logger.error(f"Error during embedding: {error_msg}")
                    
                if "dimension error" in error_msg.lower():
                    # Clear vector store instance to force recreation
                    logger.info("Detected dimension error, attempting to recreate vector store")
                    import app.rag.embeddings
                    app.rag.embeddings._vector_store = None
                    
                    # Try again with new vector store
                    try:
                        logger.info("Retrying with new vector store")
                        vector_store = get_vector_store()
                        for i in range(0, len(splits), batch_size):
                            batch = splits[i:i+batch_size]
                            batch_end = min(i+batch_size, len(splits))
                            logger.debug(f"Re-processing batch {i//batch_size + 1}, chunks {i+1}-{batch_end}")
                            vector_store.add_documents(documents=batch)
                        success = True
                        logger.info("Successfully recovered from dimension error")
                    except Exception as e2:
                        logger.error(f"Second embedding attempt failed: {str(e2)}")
                        result["error"] = f"Failed to add documents after retry: {str(e2)}"
                        return result
                else:
                    result["error"] = f"Failed to add documents to vector store: {error_msg}"
                    return result
        
        # If successful, add to knowledge base
        if success:
            logger.info(f"Adding {file_path} to knowledge base")
            add_to_knowledge_base(file_path)
            logger.info(f"PDF processing complete: {file_path}")
            result["success"] = True
            return result
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        result["error"] = f"Error processing PDF: {str(e)}"
    
    return result

def get_all_pdf_files() -> List[str]:
    """
    Get all PDF files in the storage directory
    """
    pdf_files = []
    if os.path.exists(settings.PDF_STORAGE_PATH):
        for file in os.listdir(settings.PDF_STORAGE_PATH):
            if file.lower().endswith('.pdf'):
                pdf_files.append(file)
    return pdf_files

def remove_from_knowledge_base(file_id: int) -> bool:
    """
    Remove a PDF from the knowledge base
    """
    knowledge_base = get_knowledge_base()
    entry_to_remove = None
    
    for entry in knowledge_base:
        if entry["id"] == file_id:
            entry_to_remove = entry
            break
    
    if entry_to_remove:
        knowledge_base.remove(entry_to_remove)
        save_knowledge_base(knowledge_base)
        
        # Try to remove the file too
        try:
            file_path = entry_to_remove["file_path"]
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass  # If file removal fails, continue anyway
            
        return True
    return False
