"""Merge translated sessions and personas back into the source dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from openpyxl import Workbook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine persona and translation outputs into the source dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tmp/source.json"),
        help="Path to the base dataset JSON (default: tmp/source.json).",
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path("tmp/personas.json"),
        help="Path to personas JSON (default: tmp/personas.json).",
    )
    parser.add_argument(
        "--translations-dir",
        type=Path,
        default=Path("tmp/translated"),
        help="Directory containing translated sessions (default: tmp/translated).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("tmp/source_translated.json"),
        help="Where to write the merged JSON (default: tmp/source_translated.json).",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path("tmp/source_translated.xlsx"),
        help="Where to write the merged XLSX (default: tmp/source_translated.xlsx).",
    )
    return parser


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_persona_map(personas: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for entry in personas:
        key_value = entry.get("KEY")
        english_name = entry.get("english_name")
        if not isinstance(key_value, str) or key_value in ("", "UNKNOWN"):
            continue
        if not isinstance(english_name, str) or not english_name.strip():
            continue
        mapping.setdefault(key_value, [])
        if english_name not in mapping[key_value]:
            mapping[key_value].append(english_name)
    return mapping


def build_translation_map(translated_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not translated_dir.exists():
        raise FileNotFoundError(f"Translations directory not found: {translated_dir}")

    for path in translated_dir.glob("*.json"):
        data = load_json(path)
        session = data.get("session", {})
        turns = session.get("turns", [])
        for turn in turns:
            key_value = turn.get("KEY")
            english_line = turn.get("english_utterance")
            if (
                isinstance(key_value, str)
                and key_value
                and isinstance(english_line, str)
                and english_line
            ):
                mapping[key_value] = english_line
    return mapping


def determine_headers(rows: List[Dict[str, Any]]) -> List[str]:
    headers: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    if "english" in headers:
        headers.append(headers.pop(headers.index("english")))
    else:
        headers.append("english")
    return headers


def write_xlsx(rows: List[Dict[str, Any]], headers: List[str], path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "translations"

    sheet.append(headers)
    for row in rows:
        sheet.append([row.get(header, "") for header in headers])

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON array.")

    personas = load_json(args.personas)
    if not isinstance(personas, list):
        raise ValueError("Personas payload must be a list.")

    persona_map = build_persona_map(personas)
    translation_map = build_translation_map(args.translations_dir)

    merged_rows: List[Dict[str, Any]] = []
    for row in dataset:
        if not isinstance(row, dict):
            continue
        new_row = dict(row)
        key_value = row.get("KEY")
        english_value = ""

        if isinstance(key_value, str) and key_value:
            if key_value in translation_map:
                english_value = translation_map[key_value]
            elif key_value in persona_map:
                english_value = ", ".join(persona_map[key_value])

        new_row["english"] = english_value
        merged_rows.append(new_row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(merged_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    headers = determine_headers(merged_rows)
    write_xlsx(merged_rows, headers, args.output_xlsx)

    print(f"Merged JSON written to {args.output_json}")
    print(f"Merged XLSX written to {args.output_xlsx}")


if __name__ == "__main__":
    main()

