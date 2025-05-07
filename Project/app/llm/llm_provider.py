from typing import List, Dict, Any, Optional, Union
import logging
import requests
import json
from abc import ABC, abstractmethod

from ..core.config import settings

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, query: str, context: List[Dict[str, Any]], 
                         conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate a response to a query based on context and conversation history.
        
        Args:
            query: User query
            context: Retrieved context documents
            conversation_history: Optional conversation history
            
        Returns:
            Dict[str, Any]: Response with answer and citations
        """
        pass

class OllamaProvider(BaseLLMProvider):
    """LLM provider using Ollama models."""
    
    def __init__(self, model_name: str = settings.LLM_MODEL, 
                 base_url: str = "http://13.234.177.214:11434"):  
        """
        Initialize the Ollama provider.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate" 
        
        logger.info(f"Initialized Ollama provider with model: {model_name}")
    
    def generate_response(self, query: str, context: List[Dict[str, Any]], 
                         conversation_history: Optional[List[Dict[str, Any]]] = None,
                         require_citations: bool = True) -> Dict[str, Any]:
        """
        Generate a response using Ollama API.
        
        Args:
            query: User query
            context: Retrieved context documents
            conversation_history: Optional conversation history
            require_citations: Whether to include citations in the response
            
        Returns:
            Dict[str, Any]: Response with answer and citations
        """
        try:
   
            context_text = "\n\n".join([f"Document: {ctx['metadata']['source']}, Page: {ctx['metadata']['page']}\n{ctx['text']}" 
                                      for ctx in context])
            
      
            system_prompt = f"""You are a helpful AI assistant that provides accurate information based on the provided context. 
When answering questions, ALWAYS use ONLY the information from the provided context. 
If the context doesn't contain the answer, say "I don't have enough information to answer this question."
Always provide citations for your answers in the following format: [(Document Name, Page Number)].
For example: [Sample Document, p.5]

Here is the context information:
{context_text}"""
            

            conversation_text = ""
            if conversation_history:
                for message in conversation_history:
                    role = message["role"]
                    content = message["content"]
                    conversation_text += f"\n{role.upper()}: {content}"
       
            full_prompt = f"{system_prompt}\n\n{conversation_text}\n\nUSER: {query}\n\nASSISTANT:"
            
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": settings.LLM_TEMPERATURE
                }
            }
       
            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()
       
            result = response.json()
            
      
            answer_text = result.get("response", "")
            
   
            citations = self._extract_citations(answer_text, context) if require_citations else []
            
  
            response_object = {
                "answer": answer_text,
                "citations": citations
            }
            
            return response_object
            
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise
            
    def _extract_citations(self, answer: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citation information from the answer.
        This is a simplified implementation and can be improved.
        
        Args:
            answer: Generated answer text
            context: Context used for generation
            
        Returns:
            List[Dict[str, Any]]: List of citation objects
        """

        citations = []
        

        for ctx in context:
          
            if len(ctx["text"]) > 30 and any(
                segment in answer 
                for segment in [ctx["text"][:30], ctx["text"][30:60], ctx["text"][-30:]]
            ):
                citations.append({
                    "page": ctx["metadata"].get("page", 1),
                    "document_name": ctx["metadata"].get("source", "unknown")
                })
   
        unique_citations = []
        seen = set()
        
        for citation in citations:
            key = (citation["page"], citation["document_name"])
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations