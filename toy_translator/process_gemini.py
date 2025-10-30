"""Use Gemini to derive character metadata and dialogue sessions from the dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
import re

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send the translated dataset to Gemini and receive structured metadata."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/source.json"),
        help="Path to the JSON dataset produced from the XLSX file (default: data/source.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gemini_output.json"),
        help="Where to write the combined Gemini response (default: data/gemini_output.json).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model name to use when calling Gemini (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature when generating responses (default: 0.2).",
    )
    return parser


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of objects.")
    return data


def configure_gemini(model_name: str) -> genai.GenerativeModel:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in the environment or .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def build_prompt(dataset: list[dict[str, Any]]) -> str:
    sample = json.dumps(dataset, ensure_ascii=False, indent=2)
    instructions = """
You are an assistant that processes translation dataset entries. Each entry contains at least the fields "KEY" and "Korean". Some rows also include speaker or context metadata.

From the provided JSON array, extract two artefacts:
1. "characters": a list of character metadata entries. Include only rows that represent characters (e.g., NPC descriptions). Each item must contain:
   - "KEY": original KEY (string)
   - "character_name": the primary name (string)
   - Optional supporting fields that appear in the source (e.g., "speaker_gender", "description", "aliases").
   Omit empty or unknown values.
2. "sessions": a list of dialogue sessions. A session groups all conversational turns that belong together. Each session must look like:
   {
     "session_id": a human-readable identifier (string, may be derived from KEY prefixes),
     "turns": [
        {"KEY": original KEY (string), "speaker": speaker name or role (string), "utterance": Korean dialogue (string)}
     ]
   }
   Keep every KEY from the original dataset that represents an utterance. Preserve the order of turns as they appear in the source.

Return a JSON object *only* in the following format:
{
  "characters": [...],
  "sessions": [...]
}

Do not include markdown fences, commentary, or additional text—just valid JSON.
"""
    return f"{instructions.strip()}\n\nDataset:\n{sample}"


def call_gemini(model: genai.GenerativeModel, prompt: str, temperature: float) -> str:
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    if not response or not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text.strip()


def parse_gemini_json(text: str) -> dict[str, Any]:
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        raise ValueError("Could not locate JSON object in Gemini response.")
    cleaned = json_match.group(0)
    return json.loads(cleaned)


def save_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOGGER.info("Loading dataset from %s", args.input)
    dataset = load_dataset(args.input)

    model = configure_gemini(args.model)
    prompt = build_prompt(dataset)
    LOGGER.info("Requesting structured data from Gemini (%s)", args.model)
    raw_response = call_gemini(model, prompt, args.temperature)

    try:
        parsed = parse_gemini_json(raw_response)
    except json.JSONDecodeError as exc:
        dump_path = Path("tmp/gemini_raw_response.json")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(raw_response, encoding="utf-8")
        raise ValueError(
            f"Gemini response was not valid JSON. Raw output saved to {dump_path}"
        ) from exc

    if "characters" not in parsed or "sessions" not in parsed:
        raise ValueError("Gemini response must include 'characters' and 'sessions' keys.")

    save_output(args.output, parsed)
    LOGGER.info("Wrote Gemini output to %s", args.output)


if __name__ == "__main__":
    main()
