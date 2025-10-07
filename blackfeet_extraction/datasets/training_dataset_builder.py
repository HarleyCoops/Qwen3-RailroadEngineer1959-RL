"""
Simple dataset builder - creates training data from extractions.
"""

from pathlib import Path
from typing import Dict, Any


class TrainingDatasetBuilder:
    """Build training datasets from extracted dictionary pages."""

    def __init__(self, extraction_dir: str = "data/extracted", output_dir: str = "data/training_datasets"):
        self.extraction_dir = Path(extraction_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_all_datasets(self):
        """Placeholder - builds datasets."""
        print("Dataset builder initialized")
        print(f"  Extraction dir: {self.extraction_dir}")
        print(f"  Output dir: {self.output_dir}")

    def generate_statistics(self) -> Dict[str, Any]:
        """Generate basic stats."""
        extraction_files = list(self.extraction_dir.glob("page_*.json"))
        return {
            "total_pages": len(extraction_files),
            "total_entries": 0,
        }
