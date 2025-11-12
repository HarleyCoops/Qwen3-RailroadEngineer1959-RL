import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from dakota_extraction.datasets.training_dataset_builder import TrainingDatasetBuilder


def _write_extraction(path: Path, page_number: int, entries: List[Dict[str, Any]]) -> None:
    payload = {
        "layout": "dual-column",
        "entries": entries,
        "metadata": {
            "page_number": page_number,
            "image_path": f"page_{page_number:03d}.jpg",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_all_datasets(tmp_path: Path) -> None:
    extraction_dir = tmp_path / "extracted"
    dataset_dir = tmp_path / "datasets"
    extraction_dir.mkdir()

    _write_extraction(
        extraction_dir / "page_001.json",
        1,
        [
            {
                "entry_id": "page_001_entry_001",
                "dakota": "šúŋka",
                "english": "dog",
                "pos": "n.",
                "grammatical_notes": "Animate noun",
                "examples": ["šúŋka waŋ bluháŋ"],
                "confidence": 0.9,
            },
            {
                "entry_id": "page_001_entry_002",
                "dakota": "ȟáŋble",
                "english": "dream",
                "pos": "n.",
                "examples": ["ȟáŋble kiŋ waŋná waŋží"],
                "confidence": 0.8,
            },
        ],
    )

    builder = TrainingDatasetBuilder(
        extraction_dir=str(extraction_dir),
        output_dir=str(dataset_dir),
        validation_split=0.5,
    )
    stats = builder.build_all_datasets()

    assert stats.raw_entries == 2
    assert stats.deduplicated_entries == 2

    train_path = dataset_dir / "sft_train.jsonl"
    valid_path = dataset_dir / "sft_valid.jsonl"
    rl_path = dataset_dir / "rl_tasks_all.jsonl"
    spot_check_path = dataset_dir / "manual_spot_check.json"
    report_path = dataset_dir / "dataset_report.json"

    for target in (train_path, valid_path, rl_path, spot_check_path, report_path):
        assert target.exists(), f"expected artifact missing: {target}"

    train_records = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    valid_records = [json.loads(line) for line in valid_path.read_text(encoding="utf-8").splitlines()]
    rl_records = [json.loads(line) for line in rl_path.read_text(encoding="utf-8").splitlines()]

    # Half split → one entry each for train/valid.
    assert len(train_records) == 1
    assert len(valid_records) == 1

    # RL tasks should contain forward + backward pairs for every entry.
    assert len(rl_records) == 4
    directions = {task["metadata"]["direction"] for task in rl_records}
    assert directions == {"dakota_to_english", "english_to_dakota"}

    spot_check_payload = json.loads(spot_check_path.read_text(encoding="utf-8"))
    assert spot_check_payload, "spot check sample should not be empty"

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["stats"]["raw_entries"] == 2


def test_guardrail_identical_strings(tmp_path: Path) -> None:
    extraction_dir = tmp_path / "extracted"
    dataset_dir = tmp_path / "datasets"
    extraction_dir.mkdir()

    _write_extraction(
        extraction_dir / "page_010.json",
        10,
        [
            {
                "entry_id": "page_010_entry_001",
                "dakota": "áni",
                "english": "áni",
                "pos": "n.",
                "confidence": 0.7,
            }
        ],
    )

    builder = TrainingDatasetBuilder(
        extraction_dir=str(extraction_dir),
        output_dir=str(dataset_dir),
    )

    with pytest.raises(ValueError) as exc:
        builder.build_all_datasets()

    assert "identical" in str(exc.value)
