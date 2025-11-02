"""Use planner + skills architecture to classify data and extract dialogue sessions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from toy_translator.skills.utility_skills import analyze_fields
from toy_translator.classification_planner import create_classification_plan
from toy_translator.grouping_planner import create_grouping_plan
from toy_translator.execution_engine import (
    execute_classification_plan,
    execute_grouping_plan,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify dataset rows and extract dialogue sessions using planner + skills."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tmp/source.json"),
        help="Path to the JSON dataset (default: tmp/source.json).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("tmp/column_schema.json"),
        help="Path to column schema JSON (default: tmp/column_schema.json).",
    )
    parser.add_argument(
        "--classified-output",
        type=Path,
        default=Path("tmp/classified_data.json"),
        help="Where to write classification results (default: tmp/classified_data.json).",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path("tmp/sessions"),
        help="Directory to write session JSON files (default: tmp/sessions).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model for planners (default: gemini-2.5-flash).",
    )
    return parser


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load dataset from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of objects.")
    LOGGER.info(f"Loaded {len(data)} rows from {path}")
    return data


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load column schema from JSON file."""
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    LOGGER.info(f"Loaded schema from {schema_path}")
    return schema


def save_json(path: Path, data: Any) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(f"Wrote {path}")


def save_sessions(
    sessions_dir: Path,
    sessions: dict[str, list[dict[str, Any]]],
    schema: dict[str, Any]
) -> None:
    """
    Save sessions as individual JSON files in standardized format.

    Converts raw session data to standard format with 'speaker' and 'utterance' fields.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing session files
    for existing_file in sessions_dir.glob("*.json"):
        existing_file.unlink()

    # Get field mappings from schema
    unique_key = schema.get('unique_id', 'KEY')
    primary_text = schema.get('text_columns', {}).get('primary', 'Korean')

    # Find speaker field (try common field names)
    speaker_candidates = ['캐릭터이름', 'speaker', 'Speaker', 'character_name', 'CharacterName']

    # Save each session
    for session_id, turns in sessions.items():
        # Sanitize session_id for filename
        safe_id = session_id.replace('/', '_').replace('\\', '_')
        session_file = sessions_dir / f"{safe_id}.json"

        # Convert turns to standard format
        standardized_turns = []
        for turn in turns:
            # Find speaker field
            speaker = None
            for candidate in speaker_candidates:
                if candidate in turn:
                    speaker = turn.get(candidate)
                    break

            standardized_turn = {
                unique_key: turn.get(unique_key),
                "speaker": speaker or "UNKNOWN",
                "utterance": turn.get(primary_text, ""),
            }
            standardized_turns.append(standardized_turn)

        session_data = {
            "session_id": session_id,
            "turns": standardized_turns,
        }

        save_json(session_file, session_data)

    LOGGER.info(f"Saved {len(sessions)} sessions to {sessions_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 70)
    print(" TOY TRANSLATOR - PLANNER + SKILLS ARCHITECTURE")
    print("=" * 70)

    # Load inputs
    dataset = load_dataset(args.input)
    schema = load_schema(args.schema)

    # Step 0: Analyze field statistics
    print("\n" + "=" * 70)
    print("STEP 0: FIELD ANALYSIS")
    print("=" * 70)
    field_stats = analyze_fields(rows=dataset, schema=schema)

    # Step 1: Classification Planning
    print("\n" + "=" * 70)
    print("STEP 1: CLASSIFICATION PLANNING")
    print("=" * 70)
    classification_plan_result = create_classification_plan(
        rows=dataset,
        schema=schema,
        field_stats=field_stats,
        model_name=args.model,
    )
    classification_plan = classification_plan_result['plan']

    # Step 2: Execute Classification
    print("\n" + "=" * 70)
    print("STEP 2: EXECUTE CLASSIFICATION")
    print("=" * 70)
    classification_result = execute_classification_plan(
        plan=classification_plan,
        rows=dataset,
        schema=schema,
    )

    # Save classification results
    classification = classification_result['classification']
    classified_data = {
        'total_rows': len(dataset),
        'coverage': classification_result['coverage'],
        'classification': classification,
        'rows_by_type': {},
    }

    # Organize rows by type
    for type_name, indices in classification.items():
        classified_data['rows_by_type'][type_name] = [
            dataset[idx] for idx in indices
        ]

    save_json(args.classified_output, classified_data)

    # Step 3: Extract dialogue rows
    print("\n" + "=" * 70)
    print("STEP 3: EXTRACT DIALOGUE ROWS")
    print("=" * 70)
    dialogue_indices = classification.get('dialogue', [])
    dialogue_rows = [dataset[idx] for idx in dialogue_indices]

    if not dialogue_rows:
        LOGGER.warning("No dialogue rows found. Skipping session grouping.")
        print("\n" + "=" * 70)
        print("COMPLETED (No dialogues to group)")
        print("=" * 70)
        return

    print(f"Extracted {len(dialogue_rows)} dialogue rows for grouping")

    # Step 4: Grouping Planning
    print("\n" + "=" * 70)
    print("STEP 4: GROUPING PLANNING")
    print("=" * 70)
    grouping_plan_result = create_grouping_plan(
        dialogue_rows=dialogue_rows,
        schema=schema,
        model_name=args.model,
    )
    grouping_plan = grouping_plan_result['plan']

    # Step 5: Execute Grouping
    print("\n" + "=" * 70)
    print("STEP 5: EXECUTE GROUPING")
    print("=" * 70)
    grouping_result = execute_grouping_plan(
        plan=grouping_plan,
        dialogue_rows=dialogue_rows,
        schema=schema,
    )

    # Save sessions
    sessions = grouping_result['sessions']
    save_sessions(args.sessions_dir, sessions, schema)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n✅ Classification: {len(dataset)} rows → {len(classification)} types")
    for type_name, indices in sorted(classification.items()):
        print(f"   - {type_name}: {len(indices)} rows")
    print(f"\n✅ Grouping: {len(dialogue_rows)} dialogue rows → {len(sessions)} sessions")
    print(f"\n📁 Outputs:")
    print(f"   - Classification: {args.classified_output}")
    print(f"   - Sessions: {args.sessions_dir}")


if __name__ == "__main__":
    main()
