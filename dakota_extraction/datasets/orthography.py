"""Utilities for validating Dakota orthography in extracted entries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Characters that are diagnostic for Dakota orthography. The list intentionally
# focuses on characters that rarely appear in plain English so a failure is a
# strong signal that the Dakota column has been swapped with English text.
ORTHOGRAPHIC_MARKERS = {
    "á",
    "â",
    "ä",
    "é",
    "í",
    "ó",
    "ú",
    "ǫ",
    "ȟ",
    "ŋ",
    "ṡ",
    "š",
    "ž",
    "č",
    "ḣ",
    "ĺ",
    "ǧ",
    "ṫ",
    "ų",
}


# We also look for the presence of combining marks so that composed vs.
# decomposed UTF-8 forms are caught.
COMBINING_MARK_RE = re.compile(r"[\u0300-\u036f]")


@dataclass
class DakotaOrthographyValidator:
    """Validate whether headwords look like Dakota orthography."""

    min_markers: int = 1

    def count_markers(self, text: str) -> int:
        """Return the number of characters that look distinctively Dakota."""
        if not text:
            return 0

        markers = sum(1 for ch in text if ch in ORTHOGRAPHIC_MARKERS)
        if markers:
            return markers

        # Fallback to combining marks detection.
        return len(COMBINING_MARK_RE.findall(text))

    def is_probably_dakota(self, text: str) -> bool:
        """Return True if the text contains Dakota specific orthography."""
        return self.count_markers(text) >= self.min_markers

    def failing_examples(self, texts: Iterable[str]) -> list[str]:
        """Return a list of texts that do not pass the orthography check."""
        return [text for text in texts if not self.is_probably_dakota(text)]
