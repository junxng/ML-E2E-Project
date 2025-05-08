from typing import Dict, Any, List
from langchain import hub
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.runnables.base import RunnableSequence
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from app.rag.embeddings import get_vector_store

# Get logger but don't configure it (main module handles configuration)
logger = logging.getLogger(__name__)

# Define the state for our RAG application
class State(TypedDict):
    question: str
    context: list
    answer: str

# Cache the chain to avoid recreating it on every request
_rag_chain = None

def get_rag_chain():
    """
    Create a RAG chain using LangGraph, with caching to improve performance
    """
    global _rag_chain
    
    if _rag_chain is not None:
        return _rag_chain
        
    try:
        # Setup LLM with reasonable timeout
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            request_timeout=15  # Reduced timeout to prevent hanging
        )
        
        # Define the retriever with a timeout
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3},  # Reduced from 4 to 3 for efficiency
            search_type="similarity"  # Explicitly set search type for better performance
        )
        
        # Get the RAG prompt
        prompt = hub.pull("rlm/rag-prompt")
        
        # Define the nodes for our graph
        def retrieve(state: State) -> State:
            """Retrieve relevant documents based on the question."""
            question = state["question"]
            try:
                # Add timeout to retrieval operation
                start_time = time.time()
                docs = retriever.invoke(question)
                logger.info(f"Retrieval took {time.time() - start_time:.2f} seconds")
                return {"question": question, "context": docs, "answer": ""}
            except Exception as e:
                logger.error(f"Retrieval error: {str(e)}")
                # Return empty context if retrieval fails
                return {"question": question, "context": [], "answer": ""}
    
        def generate_answer(state: State) -> State:
            """Generate an answer based on the question and retrieved context."""
            question = state["question"]
            context_docs = state["context"]
            
            if not context_docs:
                # Handle case with no context
                answer = "I couldn't find relevant information to answer your question. Please try rephrasing or ask another question."
                return {"question": question, "context": context_docs, "answer": answer}
                
            try:
                # Join context with clear separators
                context_texts = []
                for i, doc in enumerate(context_docs):
                    context_texts.append(f"Document {i+1}:\n{doc.page_content}")
                context = "\n\n".join(context_texts)
                
                # Generate answer with timeout
                start_time = time.time()
                chain = prompt | llm
                answer = chain.invoke({"context": context, "question": question})
                logger.info(f"Answer generation took {time.time() - start_time:.2f} seconds")
                return {"question": question, "context": context_docs, "answer": answer.content}
            except Exception as e:
                logger.error(f"Answer generation error: {str(e)}")
                return {
                    "question": question, 
                    "context": context_docs, 
                    "answer": "I encountered an error while generating the answer. Please try again."
                }
    
        # Define our graph
        workflow = StateGraph(State)
    
        # Add our nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate_answer", generate_answer)
    
        # Add our edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_edge("generate_answer", END)  # Use END instead of "end"
    
        # Compile the graph
        _rag_chain = workflow.compile()
        
        return _rag_chain
        
    except Exception as e:
        logger.exception(f"Error creating RAG chain: {str(e)}")
        raise RuntimeError(f"Failed to create RAG chain: {str(e)}")

def query_documents(question: str) -> Dict[str, Any]:
    """
    Query the RAG system with a question, with better error handling and
    timeouts for various components
    """
    try:
        logger.info(f"Processing question: {question[:50]}...")
        
        # Get the chain
        rag_chain = get_rag_chain()
        
        # Execute with a timeout
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            rag_chain.invoke, 
            {
                "question": question,
                "context": [],
                "answer": ""
            }
        )
        
        try:
            # Wait for up to 25 seconds for a result
            result = future.result(timeout=25)
            
            # Prepare the response with simplified sources
            response = {
                "answer": result["answer"],
                "sources": []
            }
            
            # Add source information with clean formatting
            for i, doc in enumerate(result["context"]):
                if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
                    # Extract a shorter preview of content
                    content = doc.page_content
                    preview = content[:150] + "..." if len(content) > 150 else content
                    
                    # Create a clean source entry
                    source_info = {
                        "content": preview,
                        "metadata": {
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page_number", "N/A")
                        }
                    }
                    response["sources"].append(source_info)
            
            return response
            
        except TimeoutError:
            logger.error(f"Timeout occurred while processing question: {question[:50]}...")
            return {
                "answer": "Sorry, it's taking too long to process your question. Please try a simpler question or try again later.",
                "sources": []
            }
        finally:
            executor.shutdown(wait=False)
        
    except Exception as e:
        logger.exception(f"Error in query_documents: {str(e)}")
        # Return a user-friendly error message
        return {
            "answer": "Sorry, I encountered a technical issue while processing your question. Please try again or contact support if the issue persists.",
            "sources": []
        }
