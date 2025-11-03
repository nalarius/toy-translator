"""Generate acting guides for dialogue sessions using Gemini."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate acting guides for dialogue sessions using Gemini."
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
        help="Where to write the session with acting guide (file mode only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/sessions_with_guide"),
        help="Directory for sessions with acting guides (default: tmp/sessions_with_guide).",
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
        help="Sampling temperature (default: 0.3).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent processing threads (default: 5).",
    )
    return parser


def load_session(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "characters" not in payload or "session" not in payload:
        raise ValueError("Session file must contain 'characters' and 'session' keys.")
    return payload


def configure_gemini(model_name: str) -> genai.GenerativeModel:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in the environment or .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def format_character_summary(characters: List[Dict[str, Any]]) -> str:
    """Format character information for the prompt."""
    lines: List[str] = []
    for entry in characters:
        speaker = entry.get("speaker", "UNKNOWN")
        english_name = entry.get("english_name", "UNKNOWN")
        persona = entry.get("persona", {})

        gender = persona.get("gender", "unknown")
        age = persona.get("age_range", "unknown")
        occupation = persona.get("occupation", "unknown")
        speech_style = persona.get("speech_style", "N/A")
        personality = persona.get("personality", "N/A")

        lines.append(
            f"- {speaker} ({english_name})\n"
            f"  Gender: {gender}, Age: {age}, Occupation: {occupation}\n"
            f"  Speech style: {speech_style}\n"
            f"  Personality: {personality}"
        )
    return "\n".join(lines)


def format_dialogue_for_analysis(session: Dict[str, Any]) -> str:
    """Format dialogue turns for analysis."""
    blocks: List[str] = []
    turns = session.get("turns", [])
    for idx, turn in enumerate(turns, start=1):
        speaker = turn.get("speaker", "UNKNOWN")
        utterance = turn.get("utterance", "")
        blocks.append(f"{idx}. {speaker}: {utterance}")
    return "\n".join(blocks)


def build_acting_guide_prompt(session_data: Dict[str, Any]) -> str:
    """Build prompt for generating acting guide."""
    characters = session_data["characters"]
    session = session_data["session"]
    session_id = session.get("session_id", "UNKNOWN")

    character_section = format_character_summary(characters)
    dialogue_section = format_dialogue_for_analysis(session)

    expected_json = """{
  "scene_overview": "Brief description of what happens in this scene",
  "emotional_arc": "How emotions flow and change (e.g., 'suspicion → confrontation → shocking revelation → silence')",
  "pacing": "Speed and rhythm of the dialogue (e.g., 'rapid exchanges building to a slow, heavy ending')",
  "key_moments": [
    {
      "line_range": [start_line, end_line],
      "description": "What's happening in this segment",
      "emotion": "Dominant emotion or tone",
      "acting_note": "How this should be performed/translated"
    }
  ],
  "character_dynamics": {
    "character_name": "Their role and attitude in this scene"
  },
  "subtext": "What's not being said directly; underlying meanings, tensions, or ironies",
  "tone": "Overall atmosphere (e.g., 'tense', 'melancholic', 'playful', 'ominous')",
  "translation_notes": "Specific guidance for translators about maintaining the scene's intent"
}"""

    instructions = f"""
You are a professional director analyzing a dialogue scene for translation purposes.
Analyze this dialogue session and create a comprehensive acting guide that will help translators
understand the scene's emotional flow, character dynamics, and subtext.

Session ID: {session_id}

Characters:
{character_section}

Dialogue:
{dialogue_section}

Create an acting guide in JSON format that follows this structure:
{expected_json}

Instructions:
- Analyze the scene as a director would, identifying emotional beats and character intentions
- Identify key moments where the tone or dynamic shifts
- Describe the subtext - what characters mean vs. what they say
- Note any cultural, situational, or relationship context that affects how lines should be interpreted
- Provide translation guidance that preserves the scene's dramatic intent
- Use line numbers from the dialogue above when specifying line_range in key_moments
- Be specific and actionable in your acting notes

Output only valid JSON matching the structure above.
"""
    return instructions.strip()


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from Gemini response."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Unable to locate JSON object in Gemini response.")
    return json.loads(match.group(0))


def generate_acting_guide_for_session(
    session_data: Dict[str, Any],
    model: genai.GenerativeModel,
    temperature: float,
    session_id: str,
    safe_print
) -> Dict[str, Any]:
    """
    Generate acting guide for a single session.

    Returns:
        The acting guide as a dictionary
    """
    prompt = build_acting_guide_prompt(session_data)

    safe_print(f"[{session_id}] Generating acting guide...")

    # Retry logic with exponential backoff
    max_retries = 3
    response = None

    for attempt in range(1, max_retries + 1):
        try:
            safe_print(f"[{session_id}] Calling Gemini API (attempt {attempt}/{max_retries})...")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            safe_print(f"[{session_id}] ✓ API call succeeded")
            break  # Success

        except Exception as exc:
            safe_print(f"[{session_id}] ✗ API call failed: {type(exc).__name__}: {exc}")

            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                safe_print(f"[{session_id}] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                safe_print(f"[{session_id}] ✗ All {max_retries} attempts failed.")
                raise

    if not response or not response.text:
        raise RuntimeError(f"[{session_id}] Gemini returned an empty response.")

    raw_text = response.text.strip()

    try:
        safe_print(f"[{session_id}] Parsing JSON response...")
        acting_guide = extract_json(raw_text)
        safe_print(f"[{session_id}] ✓ Acting guide generated successfully")
        return acting_guide
    except json.JSONDecodeError as exc:
        dump_path = Path("tmp") / f"gemini_acting_guide_raw_{session_id}.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(raw_text, encoding="utf-8")
        safe_print(f"[{session_id}] ✗ JSON parsing failed: {exc}")
        raise ValueError(
            f"Gemini response is not valid JSON. Raw output saved to {dump_path}"
        ) from exc


def determine_output_path(
    input_path: Path,
    output_arg: Path | None,
    output_dir: Path | None,
) -> Path:
    """Determine output path for a session file."""
    if output_arg:
        return output_arg
    base_dir = output_dir or Path("tmp/sessions_with_guide")
    session_id = input_path.stem
    return base_dir / f"{session_id}.json"


def process_file(
    input_path: Path,
    output_path: Path,
    model: genai.GenerativeModel,
    temperature: float,
    print_lock: Lock = None
) -> None:
    """
    Process a single session file and add acting guide.

    Args:
        input_path: Path to input session file
        output_path: Path to output file with acting guide
        model: Gemini model
        temperature: Temperature for generation
        print_lock: Thread lock for synchronized printing
    """
    def safe_print(msg: str):
        """Thread-safe print."""
        if print_lock:
            with print_lock:
                print(msg, flush=True)
        else:
            print(msg, flush=True)

    session_data = load_session(input_path)
    session = session_data["session"]
    session_id = session.get('session_id', input_path.stem)

    # Generate acting guide
    acting_guide = generate_acting_guide_for_session(
        session_data,
        model,
        temperature,
        session_id,
        safe_print
    )

    # Add acting guide to session data
    enriched_session = {
        "characters": session_data["characters"],
        "acting_guide": acting_guide,
        "session": session_data["session"],
    }

    # Save enriched session
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(enriched_session, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    safe_print(f"[{session_id}] ✓ Acting guide added → {output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model = configure_gemini(args.model)
    input_path = args.input

    if input_path.is_dir():
        session_files = sorted(p for p in input_path.glob("*.json") if p.is_file())
        if not session_files:
            raise ValueError(f"No session JSON files found in directory: {input_path}")

        total = len(session_files)
        print(f"\n{'='*60}")
        print(f"Found {total} session files in {input_path}")
        print(f"Using {args.max_workers} concurrent workers")
        print(f"{'='*60}\n")

        # Thread-safe print lock
        print_lock = Lock()

        # Track progress
        completed = 0
        failed = []

        # Multi-threaded processing
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_path = {}
            for path in session_files:
                output_path = determine_output_path(path, None, args.output_dir)
                future = executor.submit(
                    process_file,
                    path,
                    output_path,
                    model,
                    args.temperature,
                    print_lock
                )
                future_to_path[future] = path

            # Process completed tasks
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    future.result()  # Raises exception if task failed
                    completed += 1
                    with print_lock:
                        print(f"\n{'='*60}")
                        print(f"Progress: {completed}/{total} sessions completed")
                        print(f"{'='*60}\n", flush=True)
                except Exception as exc:
                    failed.append((path.stem, exc))
                    with print_lock:
                        print(f"\n✗ Failed to process {path.stem}: {exc}\n", flush=True)

        # Final summary
        print(f"\n{'='*60}")
        print(f"Acting Guide Generation Summary:")
        print(f"  Total: {total}")
        print(f"  Completed: {completed}")
        print(f"  Failed: {len(failed)}")
        if failed:
            print(f"\nFailed sessions:")
            for session_id, exc in failed:
                print(f"  - {session_id}: {exc}")
        print(f"{'='*60}\n")

    else:
        # Single file mode
        output_path = determine_output_path(input_path, args.output, args.output_dir)
        process_file(input_path, output_path, model, args.temperature)
        print(f"\nActing guide generation complete for {input_path}.")


if __name__ == "__main__":
    main()
