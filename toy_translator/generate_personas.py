"""Generate translation personas for each speaker using Gemini."""

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
        description=(
            "Infer translation personas for each speaker by analysing aggregated dialogue and "
            "metadata with Gemini."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tmp/speakers.json"),
        help="Path to the aggregated speakers JSON (default: tmp/speakers.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/personas.json"),
        help="Where to write the persona list (default: tmp/personas.json).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model identifier (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Generation temperature for Gemini (default: 0.3).",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=0,
        help="Limit the number of utterances passed to Gemini (0 means all).",
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


def load_speakers(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Speakers JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Speakers payload must be a JSON array.")
    return data


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
        raise EnvironmentError("GEMINI_API_KEY is not set; update your .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


PLAYER_NAMES = {"유저", "플레이어", "player"}


def is_player_speaker(name: str) -> bool:
    if not name:
        return False
    return name.strip().lower() in PLAYER_NAMES


def build_player_persona(speaker: str, metadata: Dict[str, Any], unique_key: str = "KEY") -> Dict[str, Any]:
    key_value = metadata.get(unique_key) or "UNKNOWN"
    return {
        "speaker": speaker,
        "english_name": "Player",
        unique_key: key_value,
        "persona": {
            "gender": "neutral",
            "age_range": "flexible (reader/player self-insert)",
            "occupation": "player-controlled protagonist",
            "speech_style": (
                "Neutral and adaptable. Keep the tone friendly and accessible, "
                "avoiding strong regional or gendered quirks. Mirror the tone of other "
                "characters while remaining clear and natural."
            ),
            "personality": (
                "Acts as the audience surrogate. Curious, supportive, and reactive to events "
                "around them without overwhelming personal traits."
            ),
            "relationships": (
                "Interacts with party members and NPCs across the story. Maintain flexible "
                "second-person or first-person context as required by translation style."
            ),
            "translation_notes": (
                "Always translate as 'Player'. Keep language neutral so any reader can identify "
                "with the character. Avoid gendered or age-specific cues unless explicitly "
                "provided in context."
            ),
        },
    }


def is_ascii_name(value: str | None) -> bool:
    if not value:
        return False
    return all(ord(ch) < 128 for ch in value)


def collect_english_candidates(speaker: str, metadata: Dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    primary = metadata.get("english_name")
    if isinstance(primary, str) and primary.strip():
        candidates.append(primary.strip())

    if is_ascii_name(speaker):
        candidates.append(speaker)

    char_name = metadata.get("character_name")
    if isinstance(char_name, str) and is_ascii_name(char_name):
        candidates.append(char_name)

    aliases = metadata.get("aliases") or []
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and is_ascii_name(alias):
                candidates.append(alias.strip())

    return list(dict.fromkeys(candidates))


def build_prompt(
    speaker: str,
    metadata: Dict[str, Any],
    utterances: List[str],
    unique_key: str = "KEY"
) -> str:
    key_value = metadata.get(unique_key) or "UNKNOWN"
    description_parts = []
    for field in ("character_name", "speaker_gender", "description", "aliases"):
        value = metadata.get(field)
        if not value:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value if item)
        description_parts.append(f"{field}: {value}")
    english_hints = collect_english_candidates(speaker, metadata)
    metadata_block = "\n".join(description_parts) if description_parts else "N/A"

    dialogue_preview = "\n".join(f"- {line}" for line in utterances) or "No dialogue samples."

    english_hint_block = ", ".join(english_hints) if english_hints else "None provided"

    instructions = f"""
You are generating a translation persona guide for localisation. Use the metadata and dialogue
to infer details that help translators stay consistent with character voice. The response must be
valid JSON matching this schema exactly (no additional keys, no markdown fences):
{{
  "speaker": "{{speaker name}}",
  "english_name": "{{English/Romanised name}}",
  "{unique_key}": "{{character {unique_key}}}",
  "persona": {{
    "gender": "string (inferred or 'unknown')",
    "age_range": "string",
    "occupation": "string",
    "speech_style": "string describing tone, register, and quirks",
    "personality": "string outlining traits",
    "relationships": "string summarising important connections",
    "translation_notes": "string noting constraints or guidance"
  }}
}}

Rules:
- Preserve the provided speaker name and {unique_key} values exactly.
- english_name must always be present. If the name is already in English (ASCII) use it; otherwise
  infer a practical Romanised form. Prefer available candidates: {english_hint_block}.
- When information is missing, infer from dialogue; if truly unknown, write "unknown".
- Highlight speech quirks, catchphrases, or mood cues useful for translators.

Speaker: {speaker}
{unique_key}: {key_value}
Metadata:
{metadata_block}

Dialogue Samples:
{dialogue_preview}
"""
    return instructions.strip()


def extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)

    if not match:
        raise ValueError("Unable to locate JSON object in Gemini response.")

    return json.loads(match.group(0))


def safe_filename(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "unknown"


def save_personas(path: Path, personas: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(personas, ensure_ascii=False, indent=2), encoding="utf-8")


def fallback_english_name(speaker: str, metadata: Dict[str, Any]) -> str:
    candidates = collect_english_candidates(speaker, metadata)
    if candidates:
        return candidates[0]
    if is_ascii_name(speaker):
        return speaker
    return speaker


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load schema if exists and override args
    schema = load_schema(args.schema)
    if schema:
        args.unique_key = schema.get("unique_id", args.unique_key)

    speakers = load_speakers(args.input)
    model = configure_gemini(args.model)
    personas: List[Dict[str, Any]] = []

    total = len(speakers)
    for index, entry in enumerate(speakers, start=1):
        speaker_name = entry.get("speaker") or "UNKNOWN"
        metadata = entry.get("metadata") or {}
        utterances = entry.get("utterances") or []

        print(f"[{index}/{total}] Generating persona for '{speaker_name}'...")

        if is_player_speaker(speaker_name):
            persona_payload = build_player_persona(speaker_name, metadata, args.unique_key)
            personas.append(persona_payload)
            print("  -> Applied predefined Player persona.")
            continue

        max_utts = args.max_utterances
        if max_utts > 0:
            utterance_subset = utterances[:max_utts]
        else:
            utterance_subset = utterances

        prompt = build_prompt(speaker_name, metadata, utterance_subset, args.unique_key)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=args.temperature,
                response_mime_type="application/json",
            ),
        )

        if not response or not response.text:
            print(f"  ! Empty response for speaker '{speaker_name}'. Skipping.")
            continue

        raw_text = response.text.strip()

        try:
            persona_payload = extract_json(raw_text)
        except json.JSONDecodeError as exc:
            dump_path = Path("tmp") / f"gemini_persona_{safe_filename(speaker_name)}.json"
            dump_path.write_text(raw_text, encoding="utf-8")
            raise ValueError(
                f"Gemini response for '{speaker_name}' is not valid JSON. "
                f"Raw output saved to {dump_path}"
            ) from exc

        english_name = persona_payload.get("english_name")
        if not isinstance(english_name, str) or not english_name.strip():
            persona_payload["english_name"] = fallback_english_name(speaker_name, metadata)

        personas.append(persona_payload)

    save_personas(args.output, personas)
    print(f"Persona generation complete. Output written to {args.output}")


if __name__ == "__main__":
    main()
