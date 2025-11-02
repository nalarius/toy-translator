"""Translate a dialogue session JSON using Gemini personas."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate a session JSON using persona information and Gemini."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a session file or directory containing session files (e.g., tmp/sessions).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the translated session JSON (file mode only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/translated"),
        help="Directory for translated files when --input is a directory (default: tmp/translated).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model identifier (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--unique-key",
        default="KEY",
        help="Name of the unique identifier column (default: KEY).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("tmp/column_schema.json"),
        help="Path to column schema JSON (auto-loads if exists, default: tmp/column_schema.json).",
    )
    return parser


def load_session(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "characters" not in payload or "session" not in payload:
        raise ValueError("Session file must contain 'characters' and 'session' keys.")
    return payload


def load_schema(schema_path: Path) -> dict[str, Any] | None:
    """Load column schema from JSON file if it exists."""
    if not schema_path.exists():
        return None
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        return schema
    except Exception:
        return None


def configure_gemini(model_name: str) -> genai.GenerativeModel:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in the environment or .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def split_speaker_label(value: str) -> List[str]:
    parts: List[str] = []
    for part in value.split(","):
        label = part.strip()
        if label:
            parts.append(label)
    return parts


def build_characters_index(characters: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for entry in characters:
        speaker = entry.get("speaker")
        if not isinstance(speaker, str):
            continue
        index[speaker] = entry
    return index


def format_persona_summary(characters: List[Dict[str, Any]], unique_key: str = "KEY") -> str:
    lines: List[str] = []
    for entry in characters:
        speaker = entry.get("speaker", "UNKNOWN")
        english_name = entry.get("english_name", "UNKNOWN")
        key_value = entry.get(unique_key, "UNKNOWN")
        persona = entry.get("persona", {})
        gender = persona.get("gender", "unknown")
        age = persona.get("age_range", "unknown")
        occupation = persona.get("occupation", "unknown")
        speech_style = persona.get("speech_style", "N/A")
        personality = persona.get("personality", "N/A")
        notes = persona.get("translation_notes", "N/A")
        lines.append(
            f"- {speaker} → {english_name} ({unique_key}: {key_value})\n"
            f"  gender: {gender}; age: {age}; occupation: {occupation}\n"
            f"  speech style: {speech_style}\n"
            f"  personality: {personality}\n"
            f"  translation notes: {notes}"
        )
    return "\n".join(lines)


def format_dialogue(session: Dict[str, Any], character_index: Dict[str, Dict[str, Any]], unique_key: str = "KEY") -> str:
    blocks: List[str] = []
    turns = session.get("turns", [])
    for idx, turn in enumerate(turns, start=1):
        key_value = turn.get(unique_key, "UNKNOWN")
        speaker_raw = turn.get("speaker", "UNKNOWN")
        utterance = turn.get("utterance", "")

        english_names: List[str] = []
        for label in split_speaker_label(speaker_raw):
            persona_entry = character_index.get(label)
            english_name = persona_entry.get("english_name") if persona_entry else None
            english_names.append(english_name or label)
        speaker_display = ", ".join(english_names)

        blocks.append(
            f"{idx}. {unique_key}: {key_value}\n"
            f"   speaker: {speaker_raw} -> {speaker_display}\n"
            f"   line: {utterance}"
        )
    return "\n".join(blocks)


def build_prompt(session_data: Dict[str, Any], unique_key: str = "KEY") -> str:
    characters = session_data["characters"]
    session = session_data["session"]

    persona_section = format_persona_summary(characters, unique_key)
    character_index = build_characters_index(characters)
    dialogue_section = format_dialogue(session, character_index, unique_key)

    expected_json = f"""{{
  "session_id": "<same as input>",
  "turns": [
    {{
      "{unique_key}": "<unchanged>",
      "speaker": "<english_name from persona>",
      "utterance": "<translated line>"
    }}
  ]
}}"""

    instructions = f"""
You are a professional localisation translator. Translate the dialogue session into natural,
fluent English while preserving persona voice and all structural markers.

Personas:
{persona_section}

Instructions:
- Output must be JSON matching exactly this structure:
{expected_json}
- Keep 'session_id' and every '{unique_key}' value identical to the input.
- For each turn, replace the 'speaker' value with the english_name from the matching persona
  (if multiple speakers share a line, join their english_names with ', ' in the same order).
- Translate only the 'utterance' values. Preserve placeholders, tags, and formatting exactly:
  * Keep tokens like {{{{NICK}}}}, {{{{PLAYER}}}}, etc. unchanged.
  * Keep inline tags such as [2561e7]...[-] exactly as they appear.
  * Preserve the number and position of newline characters.
- Maintain each character's speech style and tone as described in the persona notes.

Dialogue to translate:
{dialogue_section}
"""
    return instructions.strip()


def extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Unable to locate JSON object in Gemini response.")
    return json.loads(match.group(0))


def enforce_structure(
    translated_session: Dict[str, Any],
    original_session: Dict[str, Any],
    character_index: Dict[str, Dict[str, Any]],
    unique_key: str = "KEY"
) -> Dict[str, Any]:
    translated_session["session_id"] = original_session.get("session_id")

    original_turns = original_session.get("turns", [])
    translated_turns = translated_session.get("turns", [])

    if len(original_turns) != len(translated_turns):
        raise ValueError("Turn count mismatch between original and translated sessions.")

    for original, translated in zip(original_turns, translated_turns):
        original_key = original.get(unique_key)
        translated[unique_key] = original_key

        speakers = []
        for label in split_speaker_label(original.get("speaker", "")):
            persona_entry = character_index.get(label)
            if not persona_entry:
                speakers.append(label)
            else:
                speakers.append(persona_entry.get("english_name") or label)
        translated["speaker"] = ", ".join(speakers)
    return translated_session


def merge_sessions(
    characters: List[Dict[str, Any]],
    original_session: Dict[str, Any],
    translated_session: Dict[str, Any],
    character_index: Dict[str, Dict[str, Any]],
    unique_key: str = "KEY"
) -> Dict[str, Any]:
    merged_turns: List[Dict[str, Any]] = []
    original_turns = original_session.get("turns", [])
    translated_turns = translated_session.get("turns", [])

    for original, translated in zip(original_turns, translated_turns):
        merged_turn = dict(original)  # preserves unique_key, speaker, utterance, and any extra fields

        english_speakers: List[str] = []
        for label in split_speaker_label(original.get("speaker", "")):
            persona_entry = character_index.get(label)
            if persona_entry:
                english_speakers.append(persona_entry.get("english_name") or label)
            else:
                english_speakers.append(label)

        merged_turn["english_speaker"] = ", ".join(english_speakers)
        merged_turn["english_utterance"] = translated.get("utterance", "")

        merged_turns.append(merged_turn)

    merged_session = {
        "session_id": original_session.get("session_id"),
        "turns": merged_turns,
    }

    return {
        "characters": characters,
        "session": merged_session,
    }


def determine_output_path(
    input_path: Path,
    output_arg: Path | None,
    output_dir: Path | None,
) -> Path:
    if output_arg:
        return output_arg
    base_dir = output_dir or Path("tmp/translated")
    session_id = input_path.stem
    return base_dir / f"{session_id}.json"


def translate_file(
    input_path: Path,
    output_path: Path,
    model: genai.GenerativeModel,
    temperature: float,
    unique_key: str = "KEY"
) -> None:
    session_data = load_session(input_path)
    characters = session_data["characters"]
    session = session_data["session"]
    character_index = build_characters_index(characters)

    prompt = build_prompt(session_data, unique_key)

    print(f"Translating session '{session.get('session_id', input_path.stem)}'...")

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )

    if not response or not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    raw_text = response.text.strip()
    try:
        translated_session = extract_json(raw_text)
    except json.JSONDecodeError as exc:
        dump_path = Path("tmp") / f"gemini_translation_raw_{input_path.stem}.json"
        dump_path.write_text(raw_text, encoding="utf-8")
        raise ValueError(
            f"Gemini response is not valid JSON. Raw output saved to {dump_path}"
        ) from exc

    translated_session = enforce_structure(translated_session, session, character_index, unique_key)
    merged_payload = merge_sessions(characters, session, translated_session, character_index, unique_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"  -> Wrote {output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load schema if exists and override args
    schema = load_schema(args.schema)
    if schema:
        args.unique_key = schema.get("unique_id", args.unique_key)

    model = configure_gemini(args.model)
    input_path = args.input

    if input_path.is_dir():
        session_files = sorted(p for p in input_path.glob("*.json") if p.is_file())
        if not session_files:
            raise ValueError(f"No session JSON files found in directory: {input_path}")
        print(f"Found {len(session_files)} session files in {input_path}")
        for path in session_files:
            output_path = determine_output_path(path, None, args.output_dir)
            translate_file(path, output_path, model, args.temperature, args.unique_key)
        print("All sessions translated.")
    else:
        output_path = determine_output_path(input_path, args.output, args.output_dir)
        translate_file(input_path, output_path, model, args.temperature, args.unique_key)
        print(f"Translation complete for {input_path}.")


if __name__ == "__main__":
    main()
