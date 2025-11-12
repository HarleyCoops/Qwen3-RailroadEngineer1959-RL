"""Dataset builder for Dakota dictionary extractions."""

from __future__ import annotations

import json
import random
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dakota_extraction.datasets.orthography import DakotaOrthographyValidator
from dakota_extraction.datasets.task_generator import (
    NormalizedEntry,
    build_rl_tasks,
    build_sft_example,
)


def _ensure_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    return [str(value).strip()]


@dataclass
class DatasetBuildStats:
    total_pages: int = 0
    raw_entries: int = 0
    deduplicated_entries: int = 0
    orthography_failures: int = 0
    identical_language_pairs: int = 0
    missing_fields: int = 0
    guardrail_threshold: float = 0.005
    sample_for_review: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "raw_entries": self.raw_entries,
            "deduplicated_entries": self.deduplicated_entries,
            "orthography_failures": self.orthography_failures,
            "identical_language_pairs": self.identical_language_pairs,
            "missing_fields": self.missing_fields,
            "guardrail_threshold": self.guardrail_threshold,
        }


class TrainingDatasetBuilder:
    """Build training datasets from extracted dictionary pages."""

    def __init__(
        self,
        extraction_dir: str = "data/extracted",
        output_dir: str = "data/training_datasets",
        validation_split: float = 0.05,
        guardrail_threshold: float = 0.005,
    ):
        self.extraction_dir = Path(extraction_dir)
        self.output_dir = Path(output_dir)
        self.validation_split = validation_split
        self.validator = DakotaOrthographyValidator()
        self.guardrail_threshold = guardrail_threshold
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_stats: Optional[DatasetBuildStats] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_all_datasets(self) -> DatasetBuildStats:
        """Load extractions, validate, and materialize dataset artifacts."""
        extraction_files = sorted(self.extraction_dir.glob("page_*.json"))
        if not extraction_files:
            raise FileNotFoundError(
                f"No extraction files found in {self.extraction_dir}. Run the extraction step first."
            )

        entries = self._collect_entries(extraction_files)
        deduped_entries, duplicates = self._deduplicate(entries)
        stats = DatasetBuildStats(
            total_pages=len(extraction_files),
            raw_entries=len(entries),
            deduplicated_entries=len(deduped_entries),
            guardrail_threshold=self.guardrail_threshold,
        )

        guardrail_report = self._enforce_guardrails(deduped_entries, stats)
        stats.orthography_failures = guardrail_report["orthography_failures"]
        stats.identical_language_pairs = guardrail_report["identical_language_pairs"]
        stats.missing_fields = guardrail_report["missing_fields"]

        self._write_sft_datasets(deduped_entries)
        self._write_rl_tasks(deduped_entries)
        stats.sample_for_review = self._write_spot_check(deduped_entries)
        self._write_report(stats, duplicates, guardrail_report)
        self._last_stats = stats
        return stats

    def generate_statistics(self) -> Dict[str, Any]:
        """Return the most recent build statistics."""
        if self._last_stats is None:
            extraction_files = list(self.extraction_dir.glob("page_*.json"))
            return {
                "total_pages": len(extraction_files),
                "total_entries": 0,
            }
        return self._last_stats.to_dict()

    # ------------------------------------------------------------------
    # Extraction handling
    # ------------------------------------------------------------------
    def _collect_entries(self, extraction_files: Iterable[Path]) -> List[NormalizedEntry]:
        entries: List[NormalizedEntry] = []
        for path in extraction_files:
            raw_text = Path(path).read_text(encoding="utf-8")
            if raw_text.startswith("version https://git-lfs"):
                # Skip pointer files when the large JSON payload is not available.
                continue
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}: {exc}") from exc
            page_meta = data.get("metadata", {})
            page_number = page_meta.get("page_number")
            source_image = page_meta.get("image_path")
            for idx, item in enumerate(data.get("entries", [])):
                entry_id = item.get("entry_id") or self._generate_entry_id(path, idx)
                normalized = self._to_normalized_entry(
                    entry_id=entry_id,
                    item=item,
                    page_number=page_number,
                    source_image=source_image,
                )
                entries.append(normalized)
        return entries

    def _generate_entry_id(self, path: Path, index: int) -> str:
        page_num = ''.join(ch for ch in path.stem if ch.isdigit())
        return f"page_{page_num or '000'}_entry_{index+1:03d}"

    def _to_normalized_entry(
        self,
        *,
        entry_id: str,
        item: Dict[str, Any],
        page_number: Optional[int],
        source_image: Optional[str],
    ) -> NormalizedEntry:
        dakota = (item.get("dakota") or item.get("headword") or "").strip()
        english = (item.get("english") or item.get("definition") or item.get("definition_primary") or "").strip()
        pos = item.get("pos") or item.get("part_of_speech")
        morphology = item.get("morphology") or item.get("grammatical_notes")
        examples = _ensure_list(item.get("examples") or item.get("example_phrases"))
        notes = item.get("notes") or item.get("etymology") or item.get("usage_notes")
        aliases = _ensure_list(item.get("aliases") or item.get("variants"))
        confidence = item.get("confidence")

        if dakota and dakota not in aliases:
            aliases.append(dakota)

        return NormalizedEntry(
            entry_id=entry_id,
            dakota=dakota,
            english=english,
            pos=pos,
            morphological_notes=morphology,
            examples=examples,
            notes=notes,
            page_number=page_number,
            source_image=source_image,
            aliases=aliases,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Deduplication / canonicalisation
    # ------------------------------------------------------------------
    def _canonical_key(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", text).lower()
        return "".join(ch for ch in normalized if ch.isalpha())

    def _deduplicate(self, entries: List[NormalizedEntry]) -> Tuple[List[NormalizedEntry], Dict[str, List[str]]]:
        merged: Dict[str, NormalizedEntry] = {}
        duplicates: Dict[str, List[str]] = {}
        for entry in entries:
            key = self._canonical_key(entry.dakota)
            if not key:
                key = entry.entry_id
            if key not in merged:
                merged[key] = entry
                duplicates[key] = []
                continue

            existing = merged[key]
            duplicates[key].append(entry.entry_id)

            # Keep the entry with higher confidence if available.
            if (entry.confidence or 0) > (existing.confidence or 0):
                existing.dakota = entry.dakota or existing.dakota
                existing.english = entry.english or existing.english
                existing.pos = entry.pos or existing.pos
                existing.morphological_notes = entry.morphological_notes or existing.morphological_notes
                existing.examples = entry.examples or existing.examples
                existing.notes = entry.notes or existing.notes
                existing.page_number = entry.page_number or existing.page_number
                existing.source_image = entry.source_image or existing.source_image
                existing.confidence = entry.confidence

            for alias in entry.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)

            if entry.examples:
                for example in entry.examples:
                    if example not in existing.examples:
                        existing.examples.append(example)

        return list(merged.values()), duplicates

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------
    def _enforce_guardrails(
        self,
        entries: List[NormalizedEntry],
        stats: DatasetBuildStats,
    ) -> Dict[str, Any]:
        orthography_failures = [e for e in entries if not self.validator.is_probably_dakota(e.dakota)]
        identical_pairs = [
            e
            for e in entries
            if e.dakota.strip().lower() == e.english.strip().lower() and e.dakota.strip()
        ]
        missing_fields = [e for e in entries if not e.dakota or not e.english]

        total = max(len(entries), 1)
        failure_ratio = len(orthography_failures) / total
        if failure_ratio > self.guardrail_threshold:
            samples = [e.dakota for e in orthography_failures[:5]]
            raise ValueError(
                "Orthography guardrail tripped: "
                f"{failure_ratio:.3%} of entries look non-Dakota. Samples: {samples}"
            )
        if identical_pairs:
            raise ValueError(
                "Detected entries where Dakota and English fields are identical: "
                + ", ".join(pair.dakota for pair in identical_pairs[:5])
            )
        if missing_fields:
            raise ValueError(
                "Detected entries with missing Dakota or English text. "
                f"Count: {len(missing_fields)}"
            )

        stats.orthography_failures = len(orthography_failures)
        stats.identical_language_pairs = len(identical_pairs)
        stats.missing_fields = len(missing_fields)

        return {
            "orthography_failures": len(orthography_failures),
            "identical_language_pairs": len(identical_pairs),
            "missing_fields": len(missing_fields),
            "orthography_failure_ratio": failure_ratio,
        }

    # ------------------------------------------------------------------
    # Dataset writers
    # ------------------------------------------------------------------
    def _write_sft_datasets(self, entries: List[NormalizedEntry]) -> None:
        rng = random.Random(42)
        shuffled = entries[:]
        rng.shuffle(shuffled)
        split_index = int(len(shuffled) * (1 - self.validation_split))
        train_entries = shuffled[:split_index]
        valid_entries = shuffled[split_index:]

        train_path = self.output_dir / "sft_train.jsonl"
        valid_path = self.output_dir / "sft_valid.jsonl"
        self._write_jsonl(train_path, (build_sft_example(e) for e in train_entries))
        self._write_jsonl(valid_path, (build_sft_example(e) for e in valid_entries))

    def _write_rl_tasks(self, entries: List[NormalizedEntry]) -> None:
        rl_path = self.output_dir / "rl_tasks_all.jsonl"
        tasks = (task for entry in entries for task in build_rl_tasks(entry))
        self._write_jsonl(rl_path, tasks)

    def _write_spot_check(self, entries: List[NormalizedEntry]) -> List[Dict[str, Any]]:
        if not entries:
            return []
        rng = random.Random(171)
        sample_size = max(1, round(len(entries) * 0.01))
        sampled = rng.sample(entries, k=min(sample_size, len(entries)))
        payload = [
            {
                "entry_id": e.entry_id,
                "dakota": e.dakota,
                "english": e.english,
                "pos": e.pos,
                "page_number": e.page_number,
                "source_image": e.source_image,
                "aliases": e.aliases,
            }
            for e in sampled
        ]
        path = self.output_dir / "manual_spot_check.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def _write_report(
        self,
        stats: DatasetBuildStats,
        duplicates: Dict[str, List[str]],
        guardrail_report: Dict[str, Any],
    ) -> None:
        report_path = self.output_dir / "dataset_report.json"
        report_payload = {
            "stats": stats.to_dict(),
            "duplicates": {k: v for k, v in duplicates.items() if v},
            "guardrails": guardrail_report,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _write_jsonl(self, path: Path, records: Iterable[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
