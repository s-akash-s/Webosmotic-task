import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Chunks text based on semantic boundaries (paragraphs, sections, headings)
    with optional overlap between chunks.
    """
    
    def __init__(
        self, 
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap_size: int = 200,
        paragraph_separator: str = "\n\n",
        heading_patterns: List[str] = None
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters
            overlap_size: Number of characters to overlap between chunks
            paragraph_separator: String that separates paragraphs
            heading_patterns: Regex patterns to identify headings
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.paragraph_separator = paragraph_separator
        
        if heading_patterns is None:
            self.heading_patterns = [
                r"^#{1,6}\s+.+$", 
                r"^(?:Section|Chapter|Part)\s+\d+:?\s+.+$",  
                r"^---\s+Page\s+\d+\s+---$",  
                r"^[A-Z][^.!?]*(?:[.!?]|$)"  
            ]
        else:
            self.heading_patterns = heading_patterns
            
        self.heading_regex = re.compile("|".join(self.heading_patterns), re.MULTILINE)
        
    def _split_by_semantic_boundaries(self, text: str) -> List[str]:
        """
        Split text by paragraphs and headings to maintain semantic context.
        
        Args:
            text: Input text
            
        Returns:
            List of semantic units (paragraphs, sections)
        """
      
        paragraphs = text.split(self.paragraph_separator)
        
        semantic_units = []
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
        
            heading_matches = list(self.heading_regex.finditer(paragraph))
            
            if not heading_matches:
                semantic_units.append(paragraph)
                continue
                
            last_end = 0
            for match in heading_matches:
                if match.start() > last_end:
                    unit = paragraph[last_end:match.start()].strip()
                    if unit:
                        semantic_units.append(unit)
                
                if match.end() < len(paragraph):
                    unit = paragraph[match.start():].strip()
                    semantic_units.append(unit)
                else:
                    unit = paragraph[match.start():match.end()].strip()
                    semantic_units.append(unit)
                last_end = match.end()
        
        return [unit for unit in semantic_units if unit.strip()]
        
    def _merge_small_units(self, units: List[str]) -> List[str]:
        """
        Merge small semantic units to reach minimum chunk size.
        
        Args:
            units: List of semantic units
            
        Returns:
            List of merged units
        """
        merged_units = []
        current_unit = ""
        
        for unit in units:
            if len(current_unit) + len(unit) > self.max_chunk_size and current_unit:
                merged_units.append(current_unit)
                current_unit = unit
            else:
                if current_unit:
                    current_unit += self.paragraph_separator
                current_unit += unit
        
        if current_unit:
            merged_units.append(current_unit)
            
        return merged_units
    
    def create_chunks(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from text with metadata.
        
        Args:
            text: Input text
            metadata: Metadata to include with each chunk
            
        Returns:
            List of chunks with text and metadata
        """
        if metadata is None:
            metadata = {}
            
        semantic_units = self._split_by_semantic_boundaries(text)
        merged_units = self._merge_small_units(semantic_units)
        
        chunks = []
        
        for i, unit in enumerate(merged_units):
            chunk_metadata = metadata.copy()
            
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(merged_units)
            
            page_numbers = []
            page_pattern = re.compile(r"---\s+Page\s+(\d+)\s+---")
            for match in page_pattern.finditer(unit):
                page_numbers.append(int(match.group(1)))
            
            if page_numbers:
                chunk_metadata["pages"] = page_numbers
                chunk_metadata["page"] = page_numbers[0]
            
            if i > 0 and self.overlap_size > 0:
                prev_text = merged_units[i-1]
                if len(prev_text) > self.overlap_size:
                    prev_text = prev_text[-self.overlap_size:]
                unit = prev_text + self.paragraph_separator + unit
                chunk_metadata["has_previous_context"] = True
            
            if i < len(merged_units) - 1 and self.overlap_size > 0:
                next_text = merged_units[i+1]
                if len(next_text) > self.overlap_size:
                    next_text = next_text[:self.overlap_size]
                unit = unit + self.paragraph_separator + next_text
                chunk_metadata["has_next_context"] = True
                
            chunks.append({
                "text": unit,
                "metadata": chunk_metadata
            })
            
        logger.info(f"Created {len(chunks)} semantic chunks from text with {len(text)} characters")
        return chunks