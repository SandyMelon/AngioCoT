"""Format CoT files for SFT."""

import json
from pathlib import Path


# Input and output roots.
DATA_ROOT_DIR = Path("/path/to/your/structured_cot_directory").resolve()
OUTPUT_ROOT = Path("/path/to/your/dataset_cot_directory").resolve()


def convert_cot(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected a non-empty CoT string.")
    return {"cot": json.dumps({"cot": text}, ensure_ascii=False)}


def main() -> None:
    sources = sorted(DATA_ROOT_DIR.glob("*/auto/cot.json"))
    if not sources:
        raise FileNotFoundError(f"No patient auto/cot.json files in {DATA_ROOT_DIR}")

    failed = []
    for source in sources:
        patient = source.parent.parent.name
        try:
            with source.open(encoding="utf-8") as handle:
                converted = convert_cot(json.load(handle)["cot"])
            target = OUTPUT_ROOT / patient / "cot.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(converted, handle, indent=2)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failed.append(patient)
            print(f"{patient}: {error}")

    print(f"Converted {len(sources) - len(failed)}/{len(sources)} patients.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
