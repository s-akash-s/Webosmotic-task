import uuid
from typing import Dict, List, Any, Optional
import logging
import json
import os
import time

from ..core.config import settings

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manager for conversation tracking and history."""
    
    def __init__(self, storage_dir: str = "./data/conversations"):
        """
        Initialize the conversation manager.
        
        Args:
            storage_dir: Directory to store conversation data
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.conversations = {}
        logger.info(f"Initialized conversation manager with storage directory: {storage_dir}")
    
    def create_conversation(self, document_id: str) -> str:
        """
        Create a new conversation.
        
        Args:
            document_id: Document ID associated with the conversation
            
        Returns:
            str: Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        # Create conversation object
        conversation = {
            "id": conversation_id,
            "document_id": document_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": []
        }
        
        # Store in memory and on disk
        self.conversations[conversation_id] = conversation
        self._save_conversation(conversation)
        
        logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                   response: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content
            response: Optional response data
            
        Returns:
            None
        """
        try:
            # Get or load the conversation
            conversation = self._get_conversation(conversation_id)
            
            if not conversation:
                raise ValueError(f"Conversation not found: {conversation_id}")
            
            # Create message object
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time()
            }
            
            # Add response data if provided
            if response and role == "assistant":
                message["response_data"] = response
            
            # Add to conversation
            conversation["messages"].append(message)
            conversation["updated_at"] = time.time()
            
            # Save updated conversation
            self._save_conversation(conversation)
            
            logger.info(f"Added {role} message to conversation: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error adding message to conversation: {e}")
            raise
    
    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to return
            
        Returns:
            List[Dict[str, Any]]: List of messages
        """
        try:
            # Get or load the conversation
            conversation = self._get_conversation(conversation_id)
            
            if not conversation:
                return []
            
            # Return messages (limited)
            messages = conversation.get("messages", [])
            return messages[-limit:] if limit > 0 else messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def _get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Optional[Dict[str, Any]]: Conversation data or None
        """
        # Check in-memory cache first
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        # Try to load from disk
        conversation_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
        
        if os.path.exists(conversation_path):
            try:
                with open(conversation_path, 'r') as f:
                    conversation = json.load(f)
                    
                # Cache in memory
                self.conversations[conversation_id] = conversation
                return conversation
                
            except Exception as e:
                logger.error(f"Error loading conversation from disk: {e}")
                return None
        
        return None
    
    def _save_conversation(self, conversation: Dict[str, Any]) -> None:
        """
        Save a conversation to disk.
        
        Args:
            conversation: Conversation data
            
        Returns:
            None
        """
        try:
            conversation_id = conversation["id"]
            conversation_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
            
            with open(conversation_path, 'w') as f:
                json.dump(conversation, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving conversation to disk: {e}")
            raise