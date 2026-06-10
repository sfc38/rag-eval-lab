"""Document loaders: extract raw text from supported file types.

PDF extraction uses ``pypdfium2`` (Apache 2.0). PyMuPDF is deliberately avoided — it is
AGPL-3.0 and conflicts with the project's license constraint.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_text(path: Path) -> str:
    """Extract raw text from a supported document.

    Args:
        path: Path to a ``.pdf``, ``.txt``, or ``.md`` file.

    Returns:
        str: The extracted raw text (uncleaned).

    Raises:
        ValueError: If the file type is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}")


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF, page by page, using pypdfium2.

    Args:
        path: Path to a PDF file.

    Returns:
        str: Concatenated page text separated by newlines.
    """
    import pypdfium2 as pdfium  # deferred: heavy import, only needed for PDFs

    doc = pdfium.PdfDocument(str(path))
    try:
        pages: list[str] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            text_page = page.get_textpage()
            pages.append(text_page.get_text_range())
        logger.info("Extracted %d page(s) from %s", len(pages), path.name)
        return "\n".join(pages)
    finally:
        doc.close()
