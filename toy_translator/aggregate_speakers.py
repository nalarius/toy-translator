"""Aggregate dialogue turns by speaker and merge character metadata."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a speaker-centric summary from Gemini output, merging dialogue turns "
            "with character metadata when available."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tmp/gemini_output.json"),
        help="Path to the Gemini response JSON (default: tmp/gemini_output.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/speakers.json"),
        help="Where to write the aggregated speaker JSON (default: tmp/speakers.json).",
    )
    return parser


def load_gemini_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Gemini output not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gemini output must be a JSON object.")
    if "characters" not in payload or "sessions" not in payload:
        raise ValueError("Gemini output must contain 'characters' and 'sessions' keys.")
    return payload


def normalise_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def split_labels(value: Any) -> List[str]:
    if value is None:
        return []
    labels = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            labels.append(part)
    return labels


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        token = item.lower()
        if token in seen:
            continue
        seen.add(token)
        ordered.append(item)
    return ordered


def build_character_indexes(
    characters: List[Dict[str, Any]]
) -> tuple[dict[str, Dict[str, Dict[str, Any]]], dict[str, Dict[str, Any]]]:
    by_key: dict[str, Dict[str, Dict[str, Any]]] = {}
    by_name: dict[str, Dict[str, Any]] = {}

    for character in characters:
        key = normalise_label(character.get("KEY"))
        name_tokens = dedupe_preserve_order(split_labels(character.get("character_name")))

        alias_tokens: List[str] = []
        for alias in character.get("aliases", []):
            alias_tokens.extend(split_labels(alias))
        alias_tokens = dedupe_preserve_order(alias_tokens)

        if not name_tokens and key:
            name_tokens = [key]

        for name in name_tokens:
            metadata = deepcopy(character)
            metadata["character_name"] = name
            if alias_tokens:
                metadata["aliases"] = alias_tokens.copy()

            lower_name = name.lower()

            if key:
                key_entry = by_key.setdefault(key, {})
                key_entry.setdefault(lower_name, metadata)
                for alias in alias_tokens:
                    key_entry.setdefault(alias.lower(), metadata)

            by_name.setdefault(lower_name, metadata)
            for alias in alias_tokens:
                by_name.setdefault(alias.lower(), metadata)

    return by_key, by_name


def merge_metadata(dest: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if key == "aliases":
            if not value:
                continue
            existing = dest.setdefault("aliases", [])
            for alias in value:
                alias_name = normalise_label(alias)
                if alias_name and alias_name not in existing:
                    existing.append(alias_name)
        elif key not in dest or dest[key] in (None, "", []):
            dest[key] = deepcopy(value)


def aggregate_speakers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    characters = payload.get("characters", [])
    sessions = payload.get("sessions", [])

    by_key, by_name = build_character_indexes(characters)

    speakers: dict[str, dict[str, Any]] = {}

    for session in sessions:
        session_id = normalise_label(session.get("session_id")) or None
        turns = session.get("turns", [])
        if not isinstance(turns, list):
            continue

        for turn in turns:
            speaker_field = normalise_label(turn.get("speaker"))
            speaker_names = dedupe_preserve_order(split_labels(speaker_field)) or ["UNKNOWN"]
            utterance = normalise_label(turn.get("utterance"))
            key = normalise_label(turn.get("KEY"))

            for speaker_name in speaker_names:
                entry = speakers.setdefault(
                    speaker_name,
                    {"speaker": speaker_name, "metadata": {}, "utterances": []},
                )
                entry["utterances"].append(utterance)

                metadata: Dict[str, Any] | None = None
                lower_name = speaker_name.lower()

                if key and key in by_key:
                    key_entry = by_key[key]
                    metadata = key_entry.get(lower_name)
                    if metadata is None and key_entry:
                        metadata = next(iter(key_entry.values()))
                if metadata is None:
                    metadata = by_name.get(lower_name)

                if metadata:
                    merge_metadata(entry["metadata"], metadata)

    speaker_list = list(speakers.values())
    speaker_list.sort(key=lambda item: item["speaker"].lower())
    return speaker_list


def save_output(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    payload = load_gemini_output(args.input)
    aggregated = aggregate_speakers(payload)
    save_output(args.output, aggregated)


if __name__ == "__main__":
    main()
