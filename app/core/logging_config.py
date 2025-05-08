import logging
import sys

def configure_logging():
    """
    Configure logging to prevent duplication and control verbosity
    """
    # Remove default handlers to avoid duplication
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Configure a single handler for output
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    # Keep these loggers silent to reduce noise
    logging.getLogger('uvicorn.error').setLevel(logging.ERROR)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('qdrant_client').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    # Prevent propagation of logs from these modules
    for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
        logger = logging.getLogger(logger_name)
        logger.propagate = False
