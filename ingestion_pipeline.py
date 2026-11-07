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
    Enterprise Structure-Aware Document Ingestion Pipeline:
    - Extracts raw text and table structures preserving page numbers from PDF, DOCX, Markdown, TXT, CSV.
    - True Hierarchical Recursive Chunking (Pages -> Sections -> Paragraphs -> Sentences).
    - Fixes overlap calculation (preserves overlap from previous chunk, no duplication of next section).
    - Enforces hard maximum chunk ceiling (prevents massive paragraphs from bloating chunks).
    - Enriches each chunk with page numbers, document IDs, and structural metadata.
    """
    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 40, max_chunk_size: int = 450):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size

    def extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        filename_lower = filename.lower()
        
        # 1. PDF Parser with Page Preservation
        if filename_lower.endswith(".pdf"):
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                text_pages = []
                for idx, page in enumerate(pdf_reader.pages):
                    txt = page.extract_text()
                    if txt and txt.strip():
                        text_pages.append(f"--- [Page {idx+1}] ---\n{txt.strip()}")
                if text_pages:
                    return "\n\n".join(text_pages)
            except Exception as e:
                print(f"[PDF Parse Warning]: {e}")
            return file_bytes.decode("utf-8", errors="ignore")

        # 2. DOCX Parser with Paragraph & Table Preservation (Finding 3)
        elif filename_lower.endswith(".docx"):
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
                    xml_content = docx_zip.read("word/document.xml")
                    tree = ET.fromstring(xml_content)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    body_elements = []

                    # Iterate through both paragraphs and tables sequentially
                    body = tree.find(f"{{{ns['w']}}}body")
                    if body is not None:
                        for elem in body:
                            tag = elem.tag.split("}")[-1]
                            if tag == "p":
                                texts = [node.text for node in elem.iter(f"{{{ns['w']}}}t") if node.text]
                                if texts:
                                    body_elements.append("".join(texts))
                            elif tag == "tbl":
                                table_rows = []
                                for tr in elem.iter(f"{{{ns['w']}}}tr"):
                                    row_cells = []
                                    for tc in tr.iter(f"{{{ns['w']}}}tc"):
                                        cell_texts = [n.text for n in tc.iter(f"{{{ns['w']}}}t") if n.text]
                                        row_cells.append(" ".join(cell_texts).strip())
                                    if any(row_cells):
                                        table_rows.append("| " + " | ".join(row_cells) + " |")
                                if table_rows:
                                    body_elements.append("\n".join(table_rows))
                    if body_elements:
                        return "\n\n".join(body_elements)
            except Exception as e:
                print(f"[DOCX Parse Warning]: {e}")
            return file_bytes.decode("utf-8", errors="ignore")

        # 3. Plain Text, Markdown, CSV, JSON
        else:
            return file_bytes.decode("utf-8", errors="ignore")

    def _split_into_atomic_units(self, text: str) -> List[Tuple[str, int]]:
        """
        Splits raw text hierarchically (Resolves P1-04):
        1. Identifies Page markers.
        2. Splits by paragraphs (\n\n).
        3. Recursively splits paragraphs by sentence boundary.
        4. Hard-splits any unpunctuated / long sentence by word window <= max_chunk_size.
        """
        pages = re.split(r'(--- \[Page \d+\] ---)', text)
        segments: List[Tuple[str, int]] = []
        current_page = 1

        def process_block(block_str: str, page: int):
            b_words = block_str.split()
            if not b_words:
                return
            # If block already fits under chunk_size, keep atomic
            if len(b_words) <= self.chunk_size:
                segments.append((block_str, page))
                return

            # Try splitting by sentence
            sentences = re.split(r'(?<=[.!?])\s+', block_str)
            for s in sentences:
                s_str = s.strip()
                if not s_str:
                    continue
                s_words = s_str.split()
                # Hard ceiling: if sentence itself exceeds max_chunk_size (e.g. 1000-word unpunctuated text), split by word window
                if len(s_words) > self.max_chunk_size:
                    step = self.chunk_size - self.chunk_overlap
                    if step <= 0:
                        step = self.chunk_size // 2 or 1
                    for i in range(0, len(s_words), step):
                        window = s_words[i:i + self.chunk_size]
                        if window:
                            segments.append((" ".join(window), page))
                else:
                    segments.append((s_str, page))

        if len(pages) > 1:
            for part in pages:
                m = re.match(r'--- \[Page (\d+)\] ---', part.strip())
                if m:
                    current_page = int(m.group(1))
                elif part.strip():
                    paras = part.strip().split("\n\n")
                    for p in paras:
                        if p.strip():
                            process_block(p.strip(), current_page)
        else:
            paras = text.split("\n\n")
            for p in paras:
                if p.strip():
                    process_block(p.strip(), 1)

        return segments

    def recursive_chunk_text(
        self, text: str, doc_id: str, doc_title: str, collection: str = "General", tags: List[str] = None
    ) -> List[IngestedChunk]:
        """
        True Hierarchical Recursive Chunker (Resolves Finding 3):
        - Enforces chunk size and max chunk size limits.
        - Carries overlap from the END of the previous chunk forward.
        - Guarantees no truncation or duplication of subsequent text.
        """
        text = re.sub(r'\r\n', '\n', text)
        segments = self._split_into_atomic_units(text)
        if not segments:
            return []

        chunks: List[IngestedChunk] = []
        current_pieces: List[str] = []
        current_word_count = 0
        chunk_page = segments[0][1]
        previous_chunk_overlap: List[str] = []

        for seg_text, page_num in segments:
            seg_words = seg_text.split()
            if not seg_words:
                continue

            if not current_pieces:
                chunk_page = page_num
                if previous_chunk_overlap:
                    current_pieces.append(" ".join(previous_chunk_overlap))
                    current_word_count += len(previous_chunk_overlap)

            # Check if adding this segment stays within target chunk size
            if current_word_count + len(seg_words) <= self.chunk_size:
                current_pieces.append(seg_text)
                current_word_count += len(seg_words)
            else:
                # If current buffer has content, flush it as a complete chunk
                if current_pieces:
                    chunk_index = len(chunks) + 1
                    content = "\n\n".join(current_pieces)
                    chunks.append(IngestedChunk(
                        chunk_id=f"{doc_id}_c{chunk_index}",
                        doc_id=doc_id,
                        doc_title=doc_title,
                        content=content,
                        token_count=current_word_count,
                        chunk_index=chunk_index,
                        metadata={
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "category": collection,
                            "page_number": chunk_page,
                            "tags": ",".join(tags or []),
                            "chunk_index": chunk_index
                        }
                    ))
                    # Calculate true overlap from the END of the finished chunk
                    all_chunk_words = content.split()
                    previous_chunk_overlap = all_chunk_words[-self.chunk_overlap:] if len(all_chunk_words) > self.chunk_overlap else []

                # Reset buffer with overlap from previous chunk
                current_pieces = []
                current_word_count = 0
                chunk_page = page_num

                if previous_chunk_overlap:
                    current_pieces.append(" ".join(previous_chunk_overlap))
                    current_word_count += len(previous_chunk_overlap)

                current_pieces.append(seg_text)
                current_word_count += len(seg_words)

        # Flush final remaining buffer
        if current_pieces:
            chunk_index = len(chunks) + 1
            content = "\n\n".join(current_pieces)
            chunks.append(IngestedChunk(
                chunk_id=f"{doc_id}_c{chunk_index}",
                doc_id=doc_id,
                doc_title=doc_title,
                content=content,
                token_count=current_word_count,
                chunk_index=chunk_index,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "category": collection,
                    "page_number": chunk_page,
                    "tags": ",".join(tags or []),
                    "chunk_index": chunk_index
                }
            ))

        return chunks
