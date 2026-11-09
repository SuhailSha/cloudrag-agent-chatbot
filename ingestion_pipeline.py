import io
import csv
import uuid
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

MAX_PDF_PAGES = 150


class UnreadableDocumentError(Exception):
    """Raised when an uploaded document contains no readable text or exceeds structural safety limits."""
    pass


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
    Structure-Aware Document Ingestion Pipeline (Fixes H1, H2, H6, M3, N11):
    - Enforces 150-page PDF limit and rejects scanned/encrypted PDFs cleanly (Fixes H6).
    - Parses DOCX headers, body, and footers without duplicating nested tables (Fixes M3).
    - Word-counted Markdown table chunking with repeated headers and <= 250 word ceiling (Fixes H1).
    - Row-aware CSV chunking with repeated header row, strict .csv detection (never misdetecting comma prose),
      and conservative 85-token target / 110-token ceiling so MiniLM wordpieces stay < 256 (Fixes H2 & N11).
    - Never glues prose overlap words onto structured table/CSV chunks (Fixes H1).
    """
    def __init__(self, chunk_size: int = 180, chunk_overlap: int = 25, max_chunk_size: int = 250):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def _extract_docx_part_elements(xml_bytes: bytes, ns: Dict[str, str]) -> List[str]:
        """
        Safely parses a DOCX XML part (header, body, or footer) iterating direct children
        so nested tables are never duplicated (Fixes M3).
        """
        if b"<!ENTITY" in xml_bytes or len(xml_bytes) > 25 * 1024 * 1024:
            raise UnreadableDocumentError("XML entity expansion or oversized XML structure rejected.")

        root = ET.fromstring(xml_bytes)
        w_ns = f"{{{ns['w']}}}"
        container = root.find(f"{w_ns}body")
        if container is None:
            container = root

        elements = []
        for elem in container:
            tag = elem.tag.split("}")[-1]
            if tag == "p":
                texts = [node.text for node in elem.iter(f"{w_ns}t") if node.text]
                line_str = "".join(texts).strip()
                if line_str:
                    elements.append(line_str)
            elif tag == "tbl":
                table_rows = []
                for tr in elem.findall(f"{w_ns}tr"):
                    row_cells = []
                    for tc in tr.findall(f"{w_ns}tc"):
                        cell_texts = [n.text for n in tc.iter(f"{w_ns}t") if n.text]
                        cell_str = " ".join("".join(cell_texts).split())
                        row_cells.append(cell_str)
                    if any(row_cells):
                        table_rows.append("| " + " | ".join(row_cells) + " |")
                if len(table_rows) >= 2:
                    num_cols = table_rows[0].count("|") - 1
                    sep_row = "| " + " | ".join(["---"] * max(1, num_cols)) + " |"
                    if not re.match(r'^\|\s*[:\-]+', table_rows[1]):
                        table_rows.insert(1, sep_row)
                if table_rows:
                    elements.append("\n".join(table_rows))
        return elements

    def extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """
        Extracts structured text from PDF, DOCX, CSV, MD, and TXT with strict readability validation.
        """
        filename_lower = filename.lower()

        # 1. PDF Parser with Page Cap & OCR Detection
        if filename_lower.endswith(".pdf"):
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                if pdf_reader.is_encrypted:
                    raise UnreadableDocumentError(f"PDF '{filename}' is password-protected or encrypted. Please upload an unencrypted document.")

                if len(pdf_reader.pages) > MAX_PDF_PAGES:
                    raise UnreadableDocumentError(f"PDF '{filename}' has {len(pdf_reader.pages)} pages, exceeding the maximum limit of {MAX_PDF_PAGES} pages.")

                text_pages = []
                total_chars = 0
                for idx, page in enumerate(pdf_reader.pages):
                    txt = page.extract_text() or ""
                    clean_txt = txt.strip()
                    if clean_txt:
                        text_pages.append(f"--- [Page {idx+1}] ---\n{clean_txt}")
                        total_chars += len(re.sub(r'\s+', '', clean_txt))

                if not text_pages or total_chars < 20:
                    raise UnreadableDocumentError(f"No readable text found in '{filename}'. Scanned or image-only PDFs require OCR.")

                return "\n\n".join(text_pages)
            except UnreadableDocumentError:
                raise
            except Exception as e:
                raise UnreadableDocumentError(f"Could not read PDF '{filename}': {e}. The file may be corrupt or encrypted.")

        # 2. DOCX Parser with Header, Body, Footer & Non-Duplicated Table Preservation (Fixes M3)
        elif filename_lower.endswith(".docx"):
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
                    total_uncompressed = sum(info.file_size for info in docx_zip.infolist())
                    if total_uncompressed > 50 * 1024 * 1024:
                        raise UnreadableDocumentError(f"Word document '{filename}' exceeds uncompressed expansion limits.")

                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    namelist = docx_zip.namelist()
                    all_elements = []

                    for name in sorted(n for n in namelist if re.match(r'^word/header\d*\.xml$', n)):
                        all_elements.extend(self._extract_docx_part_elements(docx_zip.read(name), ns))

                    if "word/document.xml" in namelist:
                        all_elements.extend(self._extract_docx_part_elements(docx_zip.read("word/document.xml"), ns))

                    for name in sorted(n for n in namelist if re.match(r'^word/(?:footer\d*|footnotes|endnotes)\.xml$', n)):
                        all_elements.extend(self._extract_docx_part_elements(docx_zip.read(name), ns))

                    extracted = "\n\n".join(all_elements).strip()
                    if not extracted or len(re.sub(r'\s+', '', extracted)) < 15:
                        raise UnreadableDocumentError(f"No readable text found in Word document '{filename}'.")

                    return extracted
            except UnreadableDocumentError:
                raise
            except Exception as e:
                raise UnreadableDocumentError(f"Could not parse Word document '{filename}': {e}. File may be corrupt.")

        # 3. Plain Text, Markdown, CSV
        else:
            if b"\x00" in file_bytes[:1024]:
                raise UnreadableDocumentError(f"File '{filename}' contains binary data and cannot be read as text.")
            try:
                text_content = file_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                try:
                    text_content = file_bytes.decode("latin-1").strip()
                except Exception:
                    raise UnreadableDocumentError(f"File '{filename}' is not encoded in readable UTF-8 text.")

            if not text_content or len(re.sub(r'\s+', '', text_content)) < 10:
                raise UnreadableDocumentError(f"File '{filename}' contains empty or unreadable text.")
            return text_content

    def _chunk_markdown_table(self, lines: List[str], page: int) -> List[Tuple[str, int, bool]]:
        """
        Packs Markdown table rows by actual WORD COUNT (not row count!), repeats the table header
        and separator in every split chunk, and enforces the 250-word hard ceiling (Fixes H1).
        """
        if not lines:
            return []

        header_lines = []
        data_lines = lines
        if len(lines) >= 2 and re.match(r'^\|\s*[:\-]+', lines[1]):
            header_lines = [lines[0], lines[1]]
            data_lines = lines[2:]
        elif len(lines) >= 2:
            num_cols = max(1, lines[0].count("|") - 1)
            sep_line = "| " + " | ".join(["---"] * num_cols) + " |"
            header_lines = [lines[0], sep_line]
            data_lines = lines[1:]

        header_word_count = sum(len(h.split()) for h in header_lines)
        budget = max(60, min(self.chunk_size, self.max_chunk_size - header_word_count))

        table_segments: List[Tuple[str, int, bool]] = []
        current_rows: List[str] = list(header_lines)
        current_words = header_word_count

        for row in data_lines:
            rw_list = row.split()
            rw_count = len(rw_list)

            if header_word_count + rw_count > self.max_chunk_size:
                if len(current_rows) > len(header_lines):
                    table_segments.append(("\n".join(current_rows), page, True))
                    current_rows = list(header_lines)
                    current_words = header_word_count
                step = max(40, self.max_chunk_size - header_word_count - 4)
                for i in range(0, rw_count, step):
                    slice_str = "| " + " ".join(rw_list[i:i + step]).strip("|").strip() + " |"
                    chunk_rows = list(header_lines) + [slice_str]
                    table_segments.append(("\n".join(chunk_rows), page, True))
                continue

            if current_words + rw_count <= header_word_count + budget and current_words + rw_count <= self.max_chunk_size:
                current_rows.append(row)
                current_words += rw_count
            else:
                if len(current_rows) > len(header_lines):
                    table_segments.append(("\n".join(current_rows), page, True))
                current_rows = list(header_lines) + [row]
                current_words = header_word_count + rw_count

        if len(current_rows) > len(header_lines) or (not data_lines and current_rows):
            table_segments.append(("\n".join(current_rows), page, True))

        return table_segments

    def _chunk_csv_text(self, text: str, doc_id: str, doc_title: str, collection: str, tags: List[str]) -> List[IngestedChunk]:
        """
        Row-Aware CSV Chunker (Fixes H2 & N11):
        - Parses CSV lines cleanly so no chunk ever starts mid-row.
        - Repeats the CSV header row at the top of every chunk.
        - Uses an 85-token target and 110-token ceiling so numeric/code cells (which expand into 2-3 MiniLM
          wordpieces per token) always stay well under MiniLM's 256-wordpiece window (Fixes N11).
        """
        raw_lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not raw_lines:
            return []

        def csv_token_count(line_str: str) -> int:
            return max(1, len(re.findall(r"[a-zA-Z0-9_.\-/]+", line_str)))

        csv_target_size = min(85, self.chunk_size)
        csv_max_ceiling = min(110, self.max_chunk_size)

        header = raw_lines[0]
        data_rows = raw_lines[1:] if len(raw_lines) > 1 else []
        header_words = csv_token_count(header)
        target_limit = max(25, min(csv_target_size, csv_max_ceiling - header_words))

        chunks: List[IngestedChunk] = []
        current_lines = [header]
        current_words = header_words

        def flush_csv_chunk(lines_to_flush: List[str]):
            idx = len(chunks) + 1
            content = "\n".join(lines_to_flush)
            chunks.append(IngestedChunk(
                chunk_id=f"{doc_id}_c{idx}",
                doc_id=doc_id,
                doc_title=doc_title,
                content=content,
                token_count=csv_token_count(content),
                chunk_index=idx,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "category": collection,
                    "page_number": 1,
                    "tags": ",".join(tags or []),
                    "chunk_index": idx
                }
            ))

        for row in data_rows:
            row_words = csv_token_count(row)
            if current_words + row_words <= header_words + target_limit and current_words + row_words <= csv_max_ceiling:
                current_lines.append(row)
                current_words += row_words
            else:
                if len(current_lines) > 1:
                    flush_csv_chunk(current_lines)
                current_lines = [header, row]
                current_words = header_words + row_words

        if len(current_lines) > 1 or (not data_rows and current_lines):
            flush_csv_chunk(current_lines)

        return chunks

    def _split_into_atomic_units(self, text: str) -> List[Tuple[str, int, bool]]:
        """
        Splits text hierarchically into non-overlapping atomic segments:
        Returns list of (segment_text, page_number, is_structured_table).
        """
        pages = re.split(r'(--- \[Page \d+\] ---)', text)
        segments: List[Tuple[str, int, bool]] = []
        current_page = 1

        def process_atomic_block(block_str: str, page: int):
            lines = [l.strip() for l in block_str.splitlines() if l.strip()]
            if lines and all(l.startswith("|") and l.endswith("|") for l in lines):
                segments.extend(self._chunk_markdown_table(lines, page))
                return

            words = block_str.split()
            if not words:
                return

            if len(words) <= self.chunk_size:
                segments.append((block_str, page, False))
                return

            sentences = re.split(r'(?<=[.!?])\s+', block_str)
            for s in sentences:
                s_str = s.strip()
                if not s_str:
                    continue
                s_words = s_str.split()
                if len(s_words) > self.chunk_size:
                    for i in range(0, len(s_words), self.chunk_size):
                        window = s_words[i:i + self.chunk_size]
                        if window:
                            segments.append((" ".join(window), page, False))
                else:
                    segments.append((s_str, page, False))

        if len(pages) > 1:
            for part in pages:
                m = re.match(r'--- \[Page (\d+)\] ---', part.strip())
                if m:
                    current_page = int(m.group(1))
                elif part.strip():
                    for p in part.strip().split("\n\n"):
                        if p.strip():
                            process_atomic_block(p.strip(), current_page)
        else:
            for p in text.split("\n\n"):
                if p.strip():
                    process_atomic_block(p.strip(), 1)

        return segments

    def recursive_chunk_text(
        self, text: str, doc_id: str, doc_title: str, collection: str = "General", tags: List[str] = None
    ) -> List[IngestedChunk]:
        """
        Structure-Aware Recursive Chunker (Fixes H1, H2, N11):
        - Routes explicit .csv files to row-aware CSV chunker with header repetition (never misdetecting
          hard-wrapped comma-containing prose in .txt/.md files as CSV; Fixes N11).
        - Packs Markdown tables with repeated headers and zero prose overlap pollution.
        - Guarantees every chunk is <= self.max_chunk_size (250 words).
        """
        text = re.sub(r'\r\n', '\n', text)

        if doc_title.lower().endswith(".csv"):
            return self._chunk_csv_text(text, doc_id, doc_title, collection, tags or [])

        segments = self._split_into_atomic_units(text)
        if not segments:
            return []

        chunks: List[IngestedChunk] = []
        current_pieces: List[str] = []
        current_word_count = 0
        chunk_page = segments[0][1]
        previous_overlap_words: List[str] = []

        def emit_chunk(pieces: List[str], page: int) -> List[str]:
            chunk_index = len(chunks) + 1
            content = "\n\n".join(pieces)
            words = content.split()
            if len(words) > self.max_chunk_size:
                content = " ".join(words[:self.max_chunk_size])
                words = content.split()
            chunks.append(IngestedChunk(
                chunk_id=f"{doc_id}_c{chunk_index}",
                doc_id=doc_id,
                doc_title=doc_title,
                content=content,
                token_count=len(words),
                chunk_index=chunk_index,
                metadata={
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "category": collection,
                    "page_number": page,
                    "tags": ",".join(tags or []),
                    "chunk_index": chunk_index
                }
            ))
            return words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else []

        for seg_text, page_num, is_table in segments:
            seg_words = seg_text.split()
            if not seg_words:
                continue

            if is_table:
                if current_pieces:
                    previous_overlap_words = emit_chunk(current_pieces, chunk_page)
                    current_pieces = []
                    current_word_count = 0
                emit_chunk([seg_text], page_num)
                previous_overlap_words = []
                continue

            if not current_pieces:
                chunk_page = page_num
                if previous_overlap_words and len(previous_overlap_words) + len(seg_words) <= self.max_chunk_size:
                    current_pieces.append(" ".join(previous_overlap_words))
                    current_word_count += len(previous_overlap_words)

            if current_word_count + len(seg_words) <= self.chunk_size:
                current_pieces.append(seg_text)
                current_word_count += len(seg_words)
            else:
                if current_pieces:
                    previous_overlap_words = emit_chunk(current_pieces, chunk_page)
                current_pieces = []
                current_word_count = 0
                chunk_page = page_num

                if previous_overlap_words and len(previous_overlap_words) + len(seg_words) <= self.max_chunk_size:
                    current_pieces.append(" ".join(previous_overlap_words))
                    current_word_count += len(previous_overlap_words)

                current_pieces.append(seg_text)
                current_word_count += len(seg_words)

        if current_pieces:
            emit_chunk(current_pieces, chunk_page)

        return chunks
