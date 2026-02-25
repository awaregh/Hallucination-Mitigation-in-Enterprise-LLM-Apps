"""
Document ingestion pipeline for loading, processing, and managing source documents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a single ingested document."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def from_text(cls, content: str, source: str = "", metadata: Optional[Dict] = None) -> "Document":
        """Create a Document from raw text, auto-generating an id."""
        doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(id=doc_id, content=content, source=source, metadata=metadata or {})


@dataclass
class IngestResult:
    """Result summary of an ingestion run."""

    total_files: int
    successful: int
    failed: int
    documents: List[Document]
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.successful / self.total_files


class DocumentLoader:
    """Loads documents from filesystem paths, supporting txt, md, json, and pdf stubs."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}

    def load_file(self, path: str) -> Optional[Document]:
        """Load a single file and return a Document, or None on failure."""
        p = Path(path)
        if not p.exists():
            logger.warning("File not found: %s", path)
            return None

        ext = p.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning("Unsupported extension '%s' for file: %s", ext, path)
            return None

        try:
            content = self._read_content(p, ext)
            return Document(
                id=hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:16],
                content=content,
                source=str(p),
                metadata={"filename": p.name, "extension": ext, "size_bytes": p.stat().st_size},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s", path, exc)
            return None

    def _read_content(self, path: Path, ext: str) -> str:
        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8")
        if ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2)
        if ext == ".pdf":
            # Stub — real implementation would use pypdf or pdfminer
            logger.warning("PDF loading is stubbed; returning placeholder content for %s", path)
            return f"[PDF content placeholder for {path.name}]"
        return path.read_text(encoding="utf-8")

    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        """Load all supported files from a directory."""
        docs: List[Document] = []
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.error("Not a directory: %s", directory)
            return docs

        pattern = "**/*" if recursive else "*"
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self.load_file(str(file_path))
                if doc:
                    docs.append(doc)
        return docs


class IngestionPipeline:
    """Orchestrates loading, normalizing, and storing documents."""

    def __init__(self) -> None:
        self._loader = DocumentLoader()
        self._documents: List[Document] = []

    def load(self, paths: List[str]) -> List[Document]:
        """Load documents from a list of file or directory paths."""
        docs: List[Document] = []
        for path in paths:
            if os.path.isdir(path):
                docs.extend(self._loader.load_directory(path))
            else:
                doc = self._loader.load_file(path)
                if doc:
                    docs.append(doc)
        logger.info("Loaded %d documents from %d paths", len(docs), len(paths))
        return docs

    def process(self, documents: List[Document]) -> List[Document]:
        """Apply normalization and deduplication to loaded documents."""
        seen_ids: set = set()
        processed: List[Document] = []
        for doc in documents:
            doc.content = doc.content.strip()
            if not doc.content:
                logger.debug("Skipping empty document: %s", doc.source)
                continue
            if doc.id in seen_ids:
                logger.debug("Deduplicating document id: %s", doc.id)
                continue
            seen_ids.add(doc.id)
            processed.append(doc)
        logger.info("Processed %d unique non-empty documents", len(processed))
        return processed

    def ingest(self, paths: List[str]) -> IngestResult:
        """Full ingestion: load, process, store, and return result summary."""
        errors: List[str] = []
        total = len(paths)

        raw_docs = self.load(paths)
        processed = self.process(raw_docs)
        self._documents.extend(processed)

        failed = total - len(raw_docs)
        return IngestResult(
            total_files=total,
            successful=len(processed),
            failed=failed,
            documents=processed,
            errors=errors,
        )

    @property
    def documents(self) -> List[Document]:
        """Return all ingested documents."""
        return list(self._documents)

    def clear(self) -> None:
        """Clear all ingested documents."""
        self._documents = []
        logger.info("Ingestion pipeline cleared.")
