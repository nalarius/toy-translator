"""Attach personas to each dialogue session and save per-session JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine Gemini session data with generated personas and emit per-session files."
    )
    parser.add_argument(
        "--gemini-output",
        type=Path,
        default=Path("tmp/gemini_output.json"),
        help="Path to the Gemini output JSON (default: tmp/gemini_output.json).",
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path("tmp/personas.json"),
        help="Path to the persona list JSON (default: tmp/personas.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/sessions"),
        help="Directory where individual session files will be written (default: tmp/sessions).",
    )
    return parser


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_persona_index(personas: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for persona in personas:
        speaker = persona.get("speaker")
        if not isinstance(speaker, str):
            continue
        index[speaker] = persona
    return index


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe or "session"


def split_speaker_label(value: str) -> List[str]:
    parts: List[str] = []
    for part in value.split(","):
        label = part.strip()
        if label:
            parts.append(label)
    return parts


def collect_session_speakers(session: Dict[str, Any]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for turn in session.get("turns", []):
        speaker = turn.get("speaker")
        if not isinstance(speaker, str):
            continue
        for label in split_speaker_label(speaker):
            if label not in seen:
                seen.add(label)
                ordered.append(label)
    return ordered


def create_fallback_persona(speaker: str) -> Dict[str, Any]:
    """Create a basic fallback persona for speakers without persona data."""
    # Use the speaker name as english_name if it's ASCII, otherwise keep it as-is
    english_name = speaker if all(ord(c) < 128 for c in speaker) else speaker

    return {
        "speaker": speaker,
        "english_name": english_name,
        "KEY": "UNKNOWN",
        "persona": {
            "gender": "unknown",
            "age_range": "unknown",
            "occupation": "unknown",
            "speech_style": "Inferred from context. No specific style information available.",
            "personality": "No specific personality traits available. Translate naturally based on dialogue.",
            "relationships": "Unknown",
            "translation_notes": (
                f"No persona data available for '{speaker}'. Translate naturally based on "
                "context and dialogue content."
            ),
        },
    }


def attach_personas_to_session(
    session: Dict[str, Any],
    persona_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    session_copy = json.loads(json.dumps(session))
    speakers = collect_session_speakers(session_copy)

    characters: List[Dict[str, Any]] = []
    missing_personas: List[str] = []

    for speaker in speakers:
        if speaker not in persona_index:
            # Create fallback persona instead of raising error
            missing_personas.append(speaker)
            characters.append(create_fallback_persona(speaker))
        else:
            characters.append(persona_index[speaker])

    # Warn about missing personas
    if missing_personas:
        session_id = session_copy.get('session_id', 'UNKNOWN')
        print(f"  ⚠ Warning: Session '{session_id}' has {len(missing_personas)} speaker(s) without persona data:")
        for speaker in missing_personas:
            print(f"    - {speaker} (using fallback persona)")

    enriched = {
        "characters": characters,
        "session": session_copy,
    }
    return enriched


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    gemini_payload = load_json(args.gemini_output)
    personas_payload = load_json(args.personas)

    if not isinstance(gemini_payload, dict) or "sessions" not in gemini_payload:
        raise ValueError("Gemini output must be a JSON object containing a 'sessions' key.")
    sessions = gemini_payload["sessions"]
    if not isinstance(sessions, list):
        raise ValueError("'sessions' must be a list.")

    if not isinstance(personas_payload, list):
        raise ValueError("Personas payload must be a list.")
    persona_index = build_persona_index(personas_payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = len(sessions)
    for idx, session in enumerate(sessions, start=1):
        session_id = session.get("session_id") or f"session_{idx}"
        print(f"[{idx}/{total}] Processing session '{session_id}'...")

        enriched_session = attach_personas_to_session(session, persona_index)

        filename = sanitize_filename(session_id) or f"session_{idx}"
        output_path = args.output_dir / f"{filename}.json"
        output_path.write_text(
            json.dumps(enriched_session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  -> Wrote {output_path}")

    print(f"Completed writing {total} session files to {args.output_dir}")


if __name__ == "__main__":
    main()
