"""CLI for converting the toy translator dataset from XLSX to JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an XLSX dataset into JSON, omitting empty cells."
    )
    parser.add_argument(
        "--input",
        default=Path("data/source.xlsx"),
        type=Path,
        help="Path to the source XLSX file (default: data/source.xlsx).",
    )
    parser.add_argument(
        "--output",
        default=Path("data/source.json"),
        type=Path,
        help="Where to write the JSON output (default: data/source.json).",
    )
    return parser


def _normalise_value(value: Any) -> Any | None:
    """Return cleaned value or None (for empty)."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return value


def convert_xlsx_to_json(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input XLSX not found: {input_path}")

    workbook = load_workbook(filename=input_path, read_only=True)
    worksheet = workbook.active

    rows_iter = worksheet.iter_rows(values_only=True)
    headers = next(rows_iter, None)
    if not headers:
        raise ValueError("XLSX file does not contain a header row.")

    cleaned_headers = [
        header.strip() if isinstance(header, str) else header for header in headers
    ]

    result: list[dict[str, Any]] = []
    for row in rows_iter:
        row_dict: dict[str, Any] = {}
        for header, value in zip(cleaned_headers, row):
            if not header:
                continue
            normalised = _normalise_value(value)
            if normalised is None:
                continue
            row_dict[str(header)] = normalised
        if row_dict:
            result.append(row_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    convert_xlsx_to_json(args.input, args.output)


if __name__ == "__main__":
    main()

