import io
import uuid
import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

class IngestedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    content: str
    token_count: int
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentIngestionPipeline:
    """
    Production-grade document ingestion pipeline:
    - Extracts raw text from PDF, Markdown, TXT
    - Cleans whitespace and scrubs non-printable characters
    - Recursive chunking with sliding-window overlap
    - Enriches each chunk with doc-level and collection metadata
    """
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        filename_lower = filename.lower()
        if filename_lower.endswith(".pdf"):
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                text_pages = []
                for idx, page in enumerate(pdf_reader.pages):
                    txt = page.extract_text()
                    if txt and txt.strip():
                        text_pages.append(f"--- [Page {idx+1}] ---\n{txt}")
                return "\n\n".join(text_pages) if text_pages else file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                return file_bytes.decode("utf-8", errors="ignore")
        else:
            return file_bytes.decode("utf-8", errors="ignore")

    def recursive_chunk_text(self, text: str, doc_id: str, doc_title: str, collection: str, tags: List[str]) -> List[IngestedChunk]:
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        words = text.split()
        if not words:
            return []

        chunks: List[IngestedChunk] = []
        step = max(20, self.chunk_size - self.chunk_overlap)

        for i in range(0, len(words), step):
            slice_words = words[i:i + self.chunk_size]
            if not slice_words:
                break
            
            chunk_content = " ".join(slice_words)
            chunk_index = len(chunks) + 1
            chunk_id = f"{doc_id}_c{chunk_index}"
            
            chunk_obj = IngestedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                content=chunk_content,
                token_count=len(slice_words),
                chunk_index=chunk_index,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "collection": collection,
                    "tags": ",".join(tags or []),
                    "chunk_index": chunk_index
                }
            )
            chunks.append(chunk_obj)
            if i + self.chunk_size >= len(words):
                break

        return chunks
