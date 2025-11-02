"""Grouping skills for creating dialogue sessions."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv


def id_prefix_grouper(
    dialogue_rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    unique_key: str = 'KEY',
    separator: str = '_',
    **kwargs
) -> Dict[str, Any]:
    """
    Group dialogue rows by ID prefix (removes last segment).

    This is the simplest and fastest grouping method for hierarchical IDs.
    It assumes the last segment is a turn/utterance number within a session.

    How it works:
    - Split ID by separator (default: '_')
    - Remove last segment
    - Use remainder as session ID
    - All rows with same prefix → same session

    Best when:
    - IDs have hierarchical structure (CHAPTER_SCENE_TURN)
    - Last segment represents turn numbers
    - Prefix represents session identity

    Example:
        STORY_ROMA_34_001 → session: STORY_ROMA_34
        STORY_ROMA_34_002 → session: STORY_ROMA_34 (same session)
        STORY_ROMA_34_003 → session: STORY_ROMA_34 (same session)
        STORY_INTRO_01_001 → session: STORY_INTRO_01 (different session)

    Args:
        dialogue_rows: List of dialogue rows
        schema: Column schema
        unique_key: Name of the unique ID field (default: 'KEY')
        separator: Separator character (default: '_')

    Returns:
        {
            'sessions': {
                'STORY_ROMA_34': [rows],
                'STORY_INTRO_01': [rows],
                ...
            },
            'method': 'id_prefix_grouper'
        }

    Pros: Very fast, simple, works well for hierarchical IDs
    Cons: Assumes last segment is always turn number
    """
    print(f"\n[id_prefix_grouper] Grouping by ID prefix (separator: '{separator}')")

    sessions: Dict[str, List[Dict[str, Any]]] = {}

    for row in dialogue_rows:
        id_value = row.get(unique_key)

        if not id_value or not isinstance(id_value, str):
            # Skip rows without ID
            continue

        # Extract prefix (everything before last separator)
        parts = id_value.split(separator)
        if len(parts) > 1:
            # Remove last part (suffix)
            prefix = separator.join(parts[:-1])
        else:
            prefix = id_value

        sessions.setdefault(prefix, []).append(row)

    # Report results
    print(f"  Created {len(sessions)} sessions")
    session_sizes = [len(turns) for turns in sessions.values()]
    if session_sizes:
        avg_size = sum(session_sizes) / len(session_sizes)
        print(f"  Avg session size: {avg_size:.1f} turns")
        print(f"  Size range: {min(session_sizes)}-{max(session_sizes)} turns")

    # Show sample
    sample_sessions = list(sessions.keys())[:5]
    print(f"  Sample session IDs: {sample_sessions}")

    return {
        'sessions': sessions,
        'method': 'id_prefix_grouper'
    }


def id_suffix_grouper(
    dialogue_rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    unique_key: str = 'KEY',
    separator: str = '_',
    **kwargs
) -> Dict[str, Any]:
    """
    Group dialogue rows by numeric suffix continuity.

    Args:
        dialogue_rows: List of dialogue rows
        schema: Column schema
        unique_key: Name of the unique ID field
        separator: Separator character(s) in ID

    Returns:
        Sessions grouped by suffix continuity
    """
    print(f"\n[id_suffix_grouper] Grouping by suffix continuity")

    sessions: Dict[str, List[Dict[str, Any]]] = {}
    current_session_id = None
    current_prefix = None
    last_suffix = None

    for row in dialogue_rows:
        id_value = row.get(unique_key)

        if not id_value or not isinstance(id_value, str):
            continue

        # Extract prefix and suffix
        parts = id_value.split(separator)
        if len(parts) > 1:
            prefix = separator.join(parts[:-1])
            suffix_str = parts[-1]

            # Try to parse suffix as number
            try:
                suffix_num = int(suffix_str)
            except ValueError:
                suffix_num = None
        else:
            prefix = id_value
            suffix_num = None

        # Check if we need to start a new session
        if current_prefix != prefix:
            # Different prefix, new session
            current_session_id = f"{prefix}_session"
            current_prefix = prefix
            last_suffix = suffix_num
        elif suffix_num is not None and last_suffix is not None:
            # Check continuity
            if suffix_num != last_suffix + 1:
                # Gap in sequence, new session
                current_session_id = f"{prefix}_session_{suffix_num}"
            last_suffix = suffix_num
        else:
            # No numeric suffix, continue current session
            pass

        if current_session_id:
            sessions.setdefault(current_session_id, []).append(row)

    # Report results
    print(f"  Created {len(sessions)} sessions (based on continuity)")
    session_sizes = [len(turns) for turns in sessions.values()]
    if session_sizes:
        avg_size = sum(session_sizes) / len(session_sizes)
        print(f"  Avg session size: {avg_size:.1f} turns")

    return {
        'sessions': sessions,
        'method': 'id_suffix_grouper'
    }


def metadata_boundary_grouper(
    dialogue_rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    boundary_fields: List[str],
    target_session_filter: str = 'all',
    **kwargs
) -> Dict[str, Any]:
    """
    Group (or re-group) dialogues using metadata field changes as boundaries.

    Args:
        dialogue_rows: List of dialogue rows
        schema: Column schema
        boundary_fields: List of field names to watch for changes
        target_session_filter: 'all' or 'large_only' (only split sessions >30 turns)

    Returns:
        Sessions grouped by metadata boundaries
    """
    print(f"\n[metadata_boundary_grouper] Using boundary fields: {boundary_fields}")

    sessions: Dict[str, List[Dict[str, Any]]] = {}
    current_session_id = "session_0"
    session_counter = 0
    prev_values = {}

    for row in dialogue_rows:
        # Check for boundary crossing
        boundary_crossed = False

        for field in boundary_fields:
            current_value = row.get(field)
            prev_value = prev_values.get(field)

            if prev_value is not None and current_value != prev_value:
                # Boundary detected
                boundary_crossed = True
                print(f"  Boundary at row: {field} changed from '{prev_value}' to '{current_value}'")
                break

        if boundary_crossed:
            # Start new session
            session_counter += 1
            current_session_id = f"session_{session_counter}"

        sessions.setdefault(current_session_id, []).append(row)

        # Update previous values
        for field in boundary_fields:
            prev_values[field] = row.get(field)

    # Report results
    print(f"  Created {len(sessions)} sessions (boundary-based)")
    session_sizes = [len(turns) for turns in sessions.values()]
    if session_sizes:
        avg_size = sum(session_sizes) / len(session_sizes)
        print(f"  Avg session size: {avg_size:.1f} turns")

    return {
        'sessions': sessions,
        'method': 'metadata_boundary_grouper'
    }


def gemini_flow_grouper(
    dialogue_rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    text_column: str = 'Korean',
    unique_key: str = 'KEY',
    sample_size: int = 50,
    model_name: str = 'gemini-2.5-flash',
    **kwargs
) -> Dict[str, Any]:
    """
    Use Gemini to analyze dialogue flow and identify session boundaries.

    This is the most intelligent grouping method, using semantic understanding
    to detect conversation boundaries. It analyzes dialogue content to identify
    topic changes, scene shifts, and time jumps.

    How it works:
    1. Extract sample dialogue sequence (first N rows)
    2. Send to Gemini with text snippets
    3. Gemini identifies where new sessions should start based on:
       - Topic changes
       - Scene shifts
       - Time jumps
       - Context breaks
    4. Returns boundary indices (e.g., [0, 15, 32])
    5. Split full data at those boundaries

    Best when:
    - IDs and metadata are unreliable
    - Need semantic/contextual understanding
    - Structural methods fail

    IMPORTANT: Ignores speaker changes (speaker changes are NORMAL in dialogue)

    Args:
        dialogue_rows: List of dialogue rows
        schema: Column schema
        text_column: Name of the text field (default: 'Korean')
        unique_key: Name of the unique ID field (default: 'KEY')
        sample_size: Number of consecutive rows to analyze (default: 50)
        model_name: Gemini model to use (default: gemini-2.5-flash)

    Returns:
        {
            'sessions': {session_id: [rows], ...},
            'boundaries': [0, 15, 32, ...],
            'method': 'gemini_flow_grouper'
        }

    Pros: Most intelligent, understands context and meaning
    Cons: Sample-based (doesn't analyze all data), uses 1 API call
    Limitation: Only analyzes first sample_size rows, then applies to all
    """
    print(f"\n[gemini_flow_grouper] Analyzing dialogue flow with Gemini...")

    # Configure Gemini
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Analyze a sample to identify boundaries
    sample = dialogue_rows[:sample_size]
    sample_data = []
    for idx, row in enumerate(sample):
        sample_data.append({
            'index': idx,
            'id': row.get(unique_key),
            'text': str(row.get(text_column, ''))[:100]
        })

    prompt = f"""
Analyze this dialogue sequence and identify where conversation sessions should be split.

**Dialogue Sequence:**
{json.dumps(sample_data, ensure_ascii=False, indent=2)}

**Task:**
Identify indices where a new conversation session should start.
Consider: topic changes, scene shifts, time jumps, but NOT speaker changes.

**Return JSON:**
{{
  "session_boundaries": [0, 15, 32],  // indices where new sessions start
  "reasoning": "explanation"
}}
"""

    print(f"  Sending {len(sample)} rows to Gemini...")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    if not response or not response.text:
        print(f"  ⚠ Empty response, falling back to simple grouping")
        return id_prefix_grouper(dialogue_rows, schema, unique_key)

    # Parse response
    result = json.loads(response.text.strip())
    boundaries = result.get('session_boundaries', [0])
    print(f"  Gemini identified {len(boundaries)} boundaries")
    print(f"  Reasoning: {result.get('reasoning', 'N/A')}")

    # Create sessions based on boundaries
    sessions: Dict[str, List[Dict[str, Any]]] = {}

    for i, boundary_idx in enumerate(boundaries):
        next_boundary = boundaries[i + 1] if i + 1 < len(boundaries) else len(dialogue_rows)
        session_id = f"session_{i}"
        sessions[session_id] = dialogue_rows[boundary_idx:next_boundary]

    # Report results
    print(f"  Created {len(sessions)} sessions")
    session_sizes = [len(turns) for turns in sessions.values()]
    if session_sizes:
        avg_size = sum(session_sizes) / len(session_sizes)
        print(f"  Avg session size: {avg_size:.1f} turns")

    return {
        'sessions': sessions,
        'boundaries': boundaries,
        'method': 'gemini_flow_grouper'
    }


def fixed_window_grouper(
    dialogue_rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    window_size: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Group dialogue rows into fixed-size windows (fallback method).

    Args:
        dialogue_rows: List of dialogue rows
        schema: Column schema
        window_size: Number of turns per session

    Returns:
        Fixed-size sessions
    """
    print(f"\n[fixed_window_grouper] Grouping by fixed window size: {window_size}")

    sessions: Dict[str, List[Dict[str, Any]]] = {}

    for idx in range(0, len(dialogue_rows), window_size):
        session_id = f"window_{idx // window_size}"
        sessions[session_id] = dialogue_rows[idx:idx + window_size]

    # Report results
    print(f"  Created {len(sessions)} sessions")
    session_sizes = [len(turns) for turns in sessions.values()]
    if session_sizes:
        avg_size = sum(session_sizes) / len(session_sizes)
        print(f"  Avg session size: {avg_size:.1f} turns")

    return {
        'sessions': sessions,
        'method': 'fixed_window_grouper'
    }


def hybrid_grouper(
    dialogue_rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    primary_method: str,
    secondary_method: str,
    primary_params: Dict[str, Any] = None,
    secondary_params: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Combine two grouping methods: primary for initial grouping, secondary for refinement.

    Args:
        dialogue_rows: List of dialogue rows
        schema: Column schema
        primary_method: Name of primary grouping skill
        secondary_method: Name of secondary grouping skill
        primary_params: Parameters for primary method
        secondary_params: Parameters for secondary method

    Returns:
        Combined grouping result
    """
    print(f"\n[hybrid_grouper] Combining {primary_method} + {secondary_method}")

    # Import dynamically
    from . import grouping_skills as gs

    primary_func = getattr(gs, primary_method)
    secondary_func = getattr(gs, secondary_method)

    # Apply primary method
    print(f"  Step 1: {primary_method}")
    primary_result = primary_func(dialogue_rows, schema, **(primary_params or {}))
    primary_sessions = primary_result['sessions']

    # Apply secondary method to each primary session
    print(f"  Step 2: Refining with {secondary_method}")
    refined_sessions: Dict[str, List[Dict[str, Any]]] = {}

    for session_id, session_rows in primary_sessions.items():
        # Apply secondary grouping
        secondary_result = secondary_func(session_rows, schema, **(secondary_params or {}))
        sub_sessions = secondary_result['sessions']

        # Merge with prefixed IDs
        for sub_id, sub_rows in sub_sessions.items():
            combined_id = f"{session_id}_{sub_id}"
            refined_sessions[combined_id] = sub_rows

    # Report results
    print(f"  Final: {len(refined_sessions)} sessions (refined from {len(primary_sessions)})")

    return {
        'sessions': refined_sessions,
        'method': 'hybrid_grouper'
    }
