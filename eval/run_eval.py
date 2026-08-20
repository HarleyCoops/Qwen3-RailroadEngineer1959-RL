import argparse
import pathlib
import subprocess
from datetime import datetime, timezone

from score_extraction import score_pairs, load_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pred = load_pairs(args.pred)
    truth = load_pairs(args.truth)
    scores = score_pairs(pred, truth)

    try:
        rev = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])  # noqa: S603,S607
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - git not guaranteed
        rev = "unknown"

    md = f"""# Dakota Extraction — Small-Subset Evaluation

**Date:** {datetime.now(timezone.utc).isoformat()}  
**Commit:** `{rev}`  
**Files:** pred=`{args.pred}`, truth=`{args.truth}`

| Metric | Value |
|---|---|
| Token accuracy | {scores.token_acc:.3f} |
| Char distance (normalized, lower is better) | {scores.char_dist:.3f} |
| Diacritic preservation | {scores.diacritics:.3f} |

"""

    output = pathlib.Path(args.out)
    output.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
