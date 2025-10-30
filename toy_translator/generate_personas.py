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
    return parser


def load_speakers(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Speakers JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Speakers payload must be a JSON array.")
    return data


def configure_gemini(model_name: str) -> genai.GenerativeModel:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set; update your .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def build_prompt(
    speaker: str,
    metadata: Dict[str, Any],
    utterances: List[str],
) -> str:
    key_value = metadata.get("KEY") or "UNKNOWN"
    description_parts = []
    for field in ("character_name", "speaker_gender", "description", "aliases"):
        value = metadata.get(field)
        if not value:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value if item)
        description_parts.append(f"{field}: {value}")
    metadata_block = "\n".join(description_parts) if description_parts else "N/A"

    dialogue_preview = "\n".join(f"- {line}" for line in utterances) or "No dialogue samples."

    instructions = f"""
You are generating a translation persona guide for localisation. Use the metadata and dialogue
to infer details that help translators stay consistent with character voice. The response must be
valid JSON matching this schema exactly (no additional keys, no markdown fences):
{{
  "speaker": "{{speaker name}}",
  "KEY": "{{character KEY}}",
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
- Preserve the provided speaker name and KEY values exactly.
- When information is missing, infer from dialogue; if truly unknown, write "unknown".
- Highlight speech quirks, catchphrases, or mood cues useful for translators.

Speaker: {speaker}
KEY: {key_value}
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    speakers = load_speakers(args.input)
    model = configure_gemini(args.model)
    personas: List[Dict[str, Any]] = []

    total = len(speakers)
    for index, entry in enumerate(speakers, start=1):
        speaker_name = entry.get("speaker") or "UNKNOWN"
        metadata = entry.get("metadata") or {}
        utterances = entry.get("utterances") or []

        print(f"[{index}/{total}] Generating persona for '{speaker_name}'...")

        max_utts = args.max_utterances
        if max_utts > 0:
            utterance_subset = utterances[:max_utts]
        else:
            utterance_subset = utterances

        prompt = build_prompt(speaker_name, metadata, utterance_subset)
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

        personas.append(persona_payload)

    save_personas(args.output, personas)
    print(f"Persona generation complete. Output written to {args.output}")


if __name__ == "__main__":
    main()

