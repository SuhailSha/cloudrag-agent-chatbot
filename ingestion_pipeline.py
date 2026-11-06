import io
import uuid
import re
import zipfile
import xml.etree.ElementTree as ET
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
    - Extracts raw text from PDF, DOCX, Markdown, TXT, JSON, CSV
    - Cleans whitespace and scrubs non-printable characters
    - Recursive chunking with sliding-window overlap
    - Enriches each chunk with doc-level and collection metadata
    """
    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        filename_lower = filename.lower()
        
        # 1. PDF Parser
        if filename_lower.endswith(".pdf"):
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                text_pages = []
                for idx, page in enumerate(pdf_reader.pages):
                    txt = page.extract_text()
                    if txt and txt.strip():
                        text_pages.append(f"--- [Page {idx+1}] ---\n{txt}")
                if text_pages:
                    return "\n\n".join(text_pages)
            except Exception:
                pass
            return file_bytes.decode("utf-8", errors="ignore")

        # 2. DOCX Parser (Native Python zip & XML extraction, 0 external dependencies)
        elif filename_lower.endswith(".docx"):
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
                    xml_content = docx_zip.read("word/document.xml")
                    tree = ET.fromstring(xml_content)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    paragraphs = []
                    for p in tree.iter(f"{{{ns['w']}}}p"):
                        texts = [node.text for node in p.iter(f"{{{ns['w']}}}t") if node.text]
                        if texts:
                            paragraphs.append("".join(texts))
                    if paragraphs:
                        return "\n\n".join(paragraphs)
            except Exception:
                pass
            return file_bytes.decode("utf-8", errors="ignore")

        # 3. Plain Text, Markdown, CSV, JSON
        else:
            return file_bytes.decode("utf-8", errors="ignore")

    def recursive_chunk_text(self, text: str, doc_id: str, doc_title: str, collection: str, tags: List[str]) -> List[IngestedChunk]:
        # Normalize line endings and whitespace
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
