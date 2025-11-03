"""Translate a dialogue session JSON using Gemini personas."""

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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent translation threads (default: 5).",
    )
    return parser


def load_session(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "characters" not in payload or "session" not in payload:
        raise ValueError("Session file must contain 'characters' and 'session' keys.")
    # acting_guide is optional - will be included if present
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


def format_acting_guide(acting_guide: Dict[str, Any]) -> str:
    """Format acting guide for the translation prompt."""
    lines: List[str] = []

    # Scene overview
    if "scene_overview" in acting_guide:
        lines.append(f"Scene Overview: {acting_guide['scene_overview']}")

    # Emotional arc
    if "emotional_arc" in acting_guide:
        lines.append(f"Emotional Arc: {acting_guide['emotional_arc']}")

    # Pacing
    if "pacing" in acting_guide:
        lines.append(f"Pacing: {acting_guide['pacing']}")

    # Tone
    if "tone" in acting_guide:
        lines.append(f"Tone: {acting_guide['tone']}")

    # Subtext
    if "subtext" in acting_guide:
        lines.append(f"Subtext: {acting_guide['subtext']}")

    # Key moments
    if "key_moments" in acting_guide and acting_guide["key_moments"]:
        lines.append("\nKey Moments:")
        for moment in acting_guide["key_moments"]:
            line_range = moment.get("line_range", [])
            description = moment.get("description", "")
            emotion = moment.get("emotion", "")
            acting_note = moment.get("acting_note", "")

            range_str = f"Lines {line_range[0]}-{line_range[1]}" if len(line_range) == 2 else "Unknown range"
            lines.append(f"  • {range_str}: {description}")
            if emotion:
                lines.append(f"    Emotion: {emotion}")
            if acting_note:
                lines.append(f"    Acting note: {acting_note}")

    # Character dynamics
    if "character_dynamics" in acting_guide and acting_guide["character_dynamics"]:
        lines.append("\nCharacter Dynamics:")
        for character, dynamic in acting_guide["character_dynamics"].items():
            lines.append(f"  • {character}: {dynamic}")

    # Translation notes
    if "translation_notes" in acting_guide:
        lines.append(f"\nTranslation Notes: {acting_guide['translation_notes']}")

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
    acting_guide = session_data.get("acting_guide")  # Optional

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

    # Build acting guide section if available
    acting_guide_section = ""
    if acting_guide:
        acting_guide_text = format_acting_guide(acting_guide)
        acting_guide_section = f"""
Acting Guide:
{acting_guide_text}

IMPORTANT: Use this acting guide to inform your translation choices. Translate each line
as if you were directing actors to perform this scene. Consider the emotional flow,
character dynamics, and subtext when choosing words and phrasing. The translation should
feel like a performance that captures the scene's dramatic intent, not just literal word
conversion.
"""

    instructions = f"""
You are a professional localisation translator. Translate the dialogue session into natural,
fluent English while preserving persona voice and all structural markers.

Personas:
{persona_section}
{acting_guide_section}
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


def validate_format_tags(original: str, translated: str) -> List[str]:
    """
    Validate that format tags match between original and translated text.

    Returns list of validation errors.
    """
    errors = []

    # Check italic tags
    orig_i_open = original.count('[i]')
    orig_i_close = original.count('[/i]')
    trans_i_open = translated.count('[i]')
    trans_i_close = translated.count('[/i]')

    if orig_i_open != trans_i_open or orig_i_close != trans_i_close:
        errors.append(f"Italic tag mismatch: original [{orig_i_open},{orig_i_close}], translated [{trans_i_open},{trans_i_close}]")

    # Check color tags (format: [HEXCODE]...[- ])
    import re
    orig_color_open = len(re.findall(r'\[[0-9a-fA-F]{6}\]', original))
    trans_color_open = len(re.findall(r'\[[0-9a-fA-F]{6}\]', translated))
    orig_color_close = original.count('[-]')
    trans_color_close = translated.count('[-]')

    if orig_color_open != trans_color_open or orig_color_close != trans_color_close:
        errors.append(f"Color tag mismatch: original [{orig_color_open},{orig_color_close}], translated [{trans_color_open},{trans_color_close}]")

    return errors


def validate_placeholders(original: str, translated: str) -> List[str]:
    """
    Validate that placeholders like {NICK}, {PLAYER} are preserved.

    Returns list of validation errors.
    """
    errors = []

    import re
    # Find all placeholders like {WORD}
    orig_placeholders = set(re.findall(r'\{[A-Z_]+\}', original))
    trans_placeholders = set(re.findall(r'\{[A-Z_]+\}', translated))

    missing = orig_placeholders - trans_placeholders
    extra = trans_placeholders - orig_placeholders

    if missing:
        errors.append(f"Missing placeholders: {missing}")
    if extra:
        errors.append(f"Extra placeholders: {extra}")

    return errors


def validate_newlines(original: str, translated: str) -> List[str]:
    """
    Validate that newline count matches between original and translated.

    Returns list of validation errors.
    """
    errors = []

    orig_count = original.count('\n')
    trans_count = translated.count('\n')

    if orig_count != trans_count:
        errors.append(f"Newline count mismatch: original {orig_count}, translated {trans_count}")

    return errors


def fix_newlines(original: str, translated: str) -> str:
    """
    Adjust newlines in translated text to match original count.

    Inserts newlines at reasonable positions (sentence boundaries, after punctuation)
    when original has more newlines, or removes excess newlines when original has fewer.
    """
    orig_count = original.count('\n')
    trans_count = translated.count('\n')

    if orig_count == trans_count:
        return translated  # Already matches

    if orig_count > trans_count:
        # Need to add newlines
        needed = orig_count - trans_count

        # Find good split points: sentence endings, commas, conjunctions
        import re

        # Remove existing newlines temporarily to find split points
        text = translated.replace('\n', ' ')

        # Find potential split points (after: . ! ? , ; and conjunctions)
        # Prefer sentence endings over commas
        sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', text)]
        comma_points = [m.end() for m in re.finditer(r'[,;]\s+', text)]

        # Combine and sort split points, preferring sentence ends
        split_points = sentence_ends + comma_points
        split_points.sort()

        if len(split_points) >= needed:
            # Choose evenly distributed points
            step = len(split_points) / (needed + 1)
            selected_points = [split_points[int((i + 1) * step)] for i in range(needed)]

            # Insert newlines at selected points
            result = []
            last_pos = 0
            for point in sorted(selected_points):
                result.append(text[last_pos:point].rstrip())
                last_pos = point
            result.append(text[last_pos:])

            return '\n'.join(result)
        else:
            # Not enough good split points, just split evenly
            words = text.split()
            chunk_size = max(1, len(words) // (needed + 1))
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunks.append(' '.join(words[i:i + chunk_size]))
            return '\n'.join(chunks[:needed + 1])

    else:
        # Need to remove newlines
        # Just join excess newlines with space
        lines = translated.split('\n')
        to_remove = trans_count - orig_count

        # Keep first and last lines, merge middle ones
        if len(lines) <= orig_count + 1:
            return translated

        # Merge consecutive lines to reduce count
        result_lines = []
        merge_indices = set(range(1, min(to_remove + 1, len(lines))))

        for i, line in enumerate(lines):
            if i in merge_indices and result_lines:
                result_lines[-1] = result_lines[-1] + ' ' + line
            else:
                result_lines.append(line)

        return '\n'.join(result_lines)

    return translated


def validate_and_fix_translation(
    original_session: Dict[str, Any],
    merged_payload: Dict[str, Any],
    session_id: str
) -> tuple[Dict[str, Any], List[str]]:
    """
    Validate and fix translation quality and consistency.

    Auto-fixes newline mismatches. Reports other validation errors.

    Returns:
        (fixed_payload, list of validation errors)
    """
    all_errors = []
    fixed_count = 0

    original_turns = original_session.get("turns", [])
    merged_turns = merged_payload.get("session", {}).get("turns", [])

    if len(original_turns) != len(merged_turns):
        all_errors.append(f"Turn count mismatch: {len(original_turns)} vs {len(merged_turns)}")
        return merged_payload, all_errors  # Can't validate individual turns if counts don't match

    for idx, (orig, merged) in enumerate(zip(original_turns, merged_turns), start=1):
        orig_utterance = orig.get("utterance", "")
        translated_utterance = merged.get("english_utterance", "")

        turn_errors = []

        # Validate format tags
        turn_errors.extend(validate_format_tags(orig_utterance, translated_utterance))

        # Validate placeholders
        turn_errors.extend(validate_placeholders(orig_utterance, translated_utterance))

        # Check and fix newlines
        newline_errors = validate_newlines(orig_utterance, translated_utterance)
        if newline_errors:
            # Auto-fix newline mismatches
            fixed_utterance = fix_newlines(orig_utterance, translated_utterance)
            merged["english_utterance"] = fixed_utterance
            fixed_count += 1
            turn_errors.append(f"Newline mismatch auto-fixed ({orig_utterance.count(chr(10))} expected)")

        if turn_errors:
            all_errors.append(f"Turn {idx}: {'; '.join(turn_errors)}")

    if fixed_count > 0:
        all_errors.insert(0, f"Auto-fixed {fixed_count} newline mismatches")

    return merged_payload, all_errors


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
    unique_key: str = "KEY",
    print_lock: Lock = None
) -> None:
    """
    Translate a single session file with retry logic.

    Args:
        input_path: Path to input session file
        output_path: Path to output translated file
        model: Gemini model
        temperature: Temperature for generation
        unique_key: Unique identifier field name
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
    characters = session_data["characters"]
    session = session_data["session"]
    session_id = session.get('session_id', input_path.stem)
    character_index = build_characters_index(characters)

    # Check if acting guide is present
    has_acting_guide = "acting_guide" in session_data
    guide_status = "with acting guide" if has_acting_guide else "without acting guide"

    prompt = build_prompt(session_data, unique_key)

    safe_print(f"[{session_id}] Starting translation ({len(session.get('turns', []))} turns, {guide_status})...")

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
        translated_session = extract_json(raw_text)
        safe_print(f"[{session_id}] ✓ JSON parsed successfully")
    except json.JSONDecodeError as exc:
        dump_path = Path("tmp") / f"gemini_translation_raw_{input_path.stem}.json"
        dump_path.write_text(raw_text, encoding="utf-8")
        safe_print(f"[{session_id}] ✗ JSON parsing failed: {exc}")
        raise ValueError(
            f"Gemini response is not valid JSON. Raw output saved to {dump_path}"
        ) from exc

    translated_session = enforce_structure(translated_session, session, character_index, unique_key)
    merged_payload = merge_sessions(characters, session, translated_session, character_index, unique_key)

    # Validate and fix translation before saving
    safe_print(f"[{session_id}] Validating translation...")
    merged_payload, validation_errors = validate_and_fix_translation(session, merged_payload, session_id)

    if validation_errors:
        safe_print(f"[{session_id}] ℹ Validation report:")
        for error in validation_errors:
            safe_print(f"[{session_id}]   - {error}")
    else:
        safe_print(f"[{session_id}] ✓ Validation passed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    safe_print(f"[{session_id}] ✓ Translation complete → {output_path}")


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

        # Multi-threaded translation
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_path = {}
            for path in session_files:
                output_path = determine_output_path(path, None, args.output_dir)
                future = executor.submit(
                    translate_file,
                    path,
                    output_path,
                    model,
                    args.temperature,
                    args.unique_key,
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
                        print(f"\n✗ Failed to translate {path.stem}: {exc}\n", flush=True)

        # Final summary
        print(f"\n{'='*60}")
        print(f"Translation Summary:")
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
        translate_file(input_path, output_path, model, args.temperature, args.unique_key)
        print(f"\nTranslation complete for {input_path}.")


if __name__ == "__main__":
    main()
