import io
import re
import time
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from config import logger, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS

class RecursiveCharacterTextSplitter:
    """
    Recursively splits text into chunks using hierarchical separators:
    Markdown headers -> paragraphs -> newlines -> sentences -> words -> characters.
    """
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_CHARS,
        chunk_overlap: int = CHUNK_OVERLAP_CHARS,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n# ",
            "\n\n## ",
            "\n\n### ",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            " ",
            "",
        ]

    def split_text(self, text: str) -> List[str]:
        final_chunks: List[str] = []
        self._split_recursive(text, self.separators, final_chunks)
        
        # Merge small adjacent chunks if under chunk_size with overlap
        merged_chunks = self._merge_splits(final_chunks, self.chunk_size, self.chunk_overlap)
        return [c.strip() for c in merged_chunks if c.strip()]

    def _split_recursive(self, text: str, separators: List[str], final_chunks: List[str]):
        if not text:
            return

        if len(text) <= self.chunk_size:
            final_chunks.append(text)
            return

        # Find the best separator that exists in text
        chosen_sep = ""
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                chosen_sep = ""
                break
            if sep in text:
                chosen_sep = sep
                new_separators = separators[i + 1:]
                break

        if chosen_sep == "":
            # Split character-by-character if no separator matches
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                final_chunks.append(text[i:i + self.chunk_size])
            return

        splits = text.split(chosen_sep)
        for idx, piece in enumerate(splits):
            # Re-attach separator except for first piece if it's a markdown header
            cur_piece = piece if idx == 0 else (chosen_sep.lstrip("\n") + piece if chosen_sep.startswith("\n\n#") else piece)
            if len(cur_piece) > self.chunk_size:
                self._split_recursive(cur_piece, new_separators, final_chunks)
            elif cur_piece.strip():
                final_chunks.append(cur_piece)

    def _merge_splits(self, splits: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = len(d)
            if total + _len > chunk_size and current_doc:
                doc_text = " ".join(current_doc).strip()
                if doc_text:
                    docs.append(doc_text)
                
                # Keep overlap from the end
                overlap_doc = []
                overlap_len = 0
                for part in reversed(current_doc):
                    if overlap_len + len(part) <= chunk_overlap:
                        overlap_doc.insert(0, part)
                        overlap_len += len(part)
                    else:
                        break
                current_doc = overlap_doc
                total = overlap_len

            current_doc.append(d)
            total += _len

        if current_doc:
            doc_text = " ".join(current_doc).strip()
            if doc_text:
                docs.append(doc_text)

        return docs


def extract_text_from_file(filename: str, content_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extracts text from raw bytes based on extension.
    Returns a list of section dicts: [{'text': str, 'page_or_section': str}]
    """
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    sections: List[Dict[str, Any]] = []

    if ext == "pdf":
        try:
            reader = PdfReader(io.BytesIO(content_bytes))
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    sections.append({
                        "text": page_text.strip(),
                        "page_or_section": f"Page {i + 1}"
                    })
        except Exception as e:
            logger.error(f"Error extracting PDF {filename}: {e}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    else:
        # Text, Markdown, code, JSON, CSV
        try:
            text = content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            text = content_bytes.decode("latin-1", errors="replace")

        if not text.strip():
            raise ValueError("Document appears to be empty.")

        # If markdown or text with clear headers, extract section headers
        header_pattern = re.compile(r"^(#{1,4}\s+[^\n]+)", re.MULTILINE)
        matches = list(header_pattern.finditer(text))
        
        if matches:
            last_pos = 0
            current_section = "Introduction / Overview"
            for m in matches:
                section_text = text[last_pos:m.start()].strip()
                if section_text:
                    sections.append({
                        "text": section_text,
                        "page_or_section": current_section
                    })
                current_section = m.group(1).lstrip("#").strip()
                last_pos = m.start()
            
            remaining = text[last_pos:].strip()
            if remaining:
                sections.append({
                    "text": remaining,
                    "page_or_section": current_section
                })
        else:
            sections.append({
                "text": text.strip(),
                "page_or_section": "General Content"
            })

    return sections


def process_and_chunk_document(
    filename: str,
    content_bytes: bytes,
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[Dict[str, Any]]:
    """
    Extracts, splits, and formats chunks with metadata ready for embedding and Chroma.
    """
    sections = extract_text_from_file(filename, content_bytes)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    chunks: List[Dict[str, Any]] = []
    timestamp = int(time.time())

    for sec in sections:
        sec_text = sec["text"]
        sec_name = sec["page_or_section"]
        raw_chunks = splitter.split_text(sec_text)

        for chunk_idx, text_chunk in enumerate(raw_chunks):
            chunk_id = f"{filename}_{sec_name.replace(' ', '_')}_{chunk_idx}_{len(chunks)}"
            # Sanitize chunk_id
            chunk_id = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", chunk_id)
            
            chunks.append({
                "id": chunk_id,
                "text": text_chunk,
                "metadata": {
                    "source": filename,
                    "section": sec_name,
                    "chunk_index": len(chunks),
                    "char_count": len(text_chunk),
                    "estimated_tokens": max(1, len(text_chunk) // 4),
                    "timestamp": timestamp,
                }
            })

    logger.info(f"Processed document '{filename}' into {len(chunks)} chunks.")
    return chunks
