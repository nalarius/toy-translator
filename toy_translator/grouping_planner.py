"""Grouping Planner - decides which grouping skills to use."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv


def analyze_dialogue_patterns(
    dialogue_rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze dialogue rows to understand grouping requirements.

    Args:
        dialogue_rows: Rows classified as dialogue
        schema: Column schema

    Returns:
        Dialogue pattern characteristics
    """
    print("\n[Grouping Planner] Analyzing dialogue patterns...")

    if not dialogue_rows:
        print("  ⚠ No dialogue rows to analyze")
        return {'total_dialogues': 0}

    unique_key = schema.get('unique_id', 'KEY')
    text_columns = schema.get('text_columns', {})
    primary_text = text_columns.get('primary', 'Korean')
    metadata = schema.get('metadata', {})

    # Analyze ID structure
    id_samples = []
    for row in dialogue_rows[:50]:
        id_value = row.get(unique_key)
        if id_value:
            id_samples.append(str(id_value))

    id_patterns = {}
    if id_samples:
        # Check for common separators
        has_underscores = sum('_' in id_val for id_val in id_samples) / len(id_samples)
        has_dashes = sum('-' in id_val for id_val in id_samples) / len(id_samples)

        # Check for numeric suffixes
        has_numeric_suffix = 0
        for id_val in id_samples:
            parts = id_val.replace('-', '_').split('_')
            if parts and parts[-1].isdigit():
                has_numeric_suffix += 1
        has_numeric_suffix = has_numeric_suffix / len(id_samples)

        # Check for consistent prefixes
        prefixes = set()
        for id_val in id_samples:
            parts = id_val.replace('-', '_').split('_')
            if len(parts) > 1:
                prefixes.add(parts[0])

        id_patterns = {
            'has_underscores': has_underscores > 0.5,
            'has_dashes': has_dashes > 0.5,
            'has_numeric_suffix': has_numeric_suffix > 0.7,
            'prefix_diversity': len(prefixes) / max(len(id_samples), 1),
            'sample_ids': id_samples[:10],
        }

    # Check for metadata boundary fields
    boundary_fields = []
    for category in ['context', 'other']:
        fields = metadata.get(category, [])
        for field in fields:
            # Look for common boundary indicators
            if any(keyword in field.lower() for keyword in ['scene', 'chapter', 'act', 'episode', 'quest']):
                # Check if field has reasonable values
                values = [row.get(field) for row in dialogue_rows[:100]]
                non_null = [v for v in values if v is not None and v != '']
                if len(non_null) > 50:  # At least 50% populated
                    unique_values = len(set(non_null))
                    if 2 < unique_values < len(non_null) * 0.5:  # Some grouping potential
                        boundary_fields.append({
                            'field': field,
                            'unique_values': unique_values,
                            'sample': list(set(non_null))[:5]
                        })

    # Analyze text flow (sample)
    text_samples = []
    for row in dialogue_rows[:20]:
        text = row.get(primary_text, '')
        text_samples.append(str(text)[:100])  # First 100 chars

    characteristics = {
        'total_dialogues': len(dialogue_rows),
        'id_patterns': id_patterns,
        'boundary_fields': boundary_fields,
        'has_boundary_metadata': len(boundary_fields) > 0,
        'text_samples': text_samples[:10],
        'unique_key': unique_key,
        'primary_text': primary_text,
    }

    print(f"  Total dialogue rows: {characteristics['total_dialogues']}")
    print(f"  ID structured: {id_patterns.get('has_numeric_suffix', False)}")
    print(f"  Boundary fields found: {len(boundary_fields)}")
    if boundary_fields:
        for bf in boundary_fields:
            print(f"    - {bf['field']} ({bf['unique_values']} unique values)")

    return characteristics


def build_planner_prompt(
    characteristics: Dict[str, Any],
) -> str:
    """Build prompt for the grouping planner."""

    return f"""You are a session grouping strategy planner for game dialogue data.

**Dialogue Characteristics:**
{json.dumps(characteristics, ensure_ascii=False, indent=2)}

**Available Grouping Skills:**

1. id_prefix_grouper
   - Description: Groups dialogues by removing the last segment of ID
   - Best when: IDs have hierarchical structure (CHAPTER_SCENE_TURN)
   - How it works: Splits ID by separator, removes last part, uses remainder as session_id
   - Params:
     * separator (str, default='_'): Delimiter character
   - Pros: Very fast, simple, works well for hierarchical IDs
   - Cons: Assumes last segment is turn number
   - Example:
     * STORY_ROMA_34_001 → session: STORY_ROMA_34
     * STORY_ROMA_34_002 → session: STORY_ROMA_34 (same session)
     * STORY_INTRO_01_001 → session: STORY_INTRO_01 (different session)

2. id_suffix_grouper
   - Description: Groups by numeric suffix continuity, breaks on gaps
   - Best when: IDs have sequential numbers, gaps indicate session boundaries
   - How it works: Tracks numeric suffixes, starts new session when sequence breaks
   - Params:
     * separator (str, default='_'): Delimiter character
   - Pros: Detects sequence interruptions
   - Cons: Requires numeric suffixes, prefix must stay same
   - Example:
     * STORY_001, STORY_002, STORY_003 → session 1 (continuous)
     * STORY_010 → session 2 (gap detected, new session)
     * STORY_011, STORY_012 → session 2 (continuous)

3. metadata_boundary_grouper
   - Description: Starts new session when specified metadata fields change
   - Best when: Scene/Chapter/Quest/Episode fields exist and indicate boundaries
   - How it works: Monitors specified fields, creates boundary when value changes
   - Params:
     * boundary_fields (list[str]): Fields to monitor (e.g., ['Scene', 'Chapter'])
   - Pros: Respects semantic boundaries, uses domain knowledge
   - Cons: Requires appropriate metadata fields
   - Example:
     * Rows 1-5: Scene='Forest' → session 1
     * Rows 6-10: Scene='Castle' → session 2 (Scene changed)
     * Rows 11-15: Scene='Castle', Chapter='2' → session 3 (Chapter changed)

4. gemini_flow_grouper
   - Description: Uses Gemini to analyze dialogue flow and detect session boundaries
   - Best when: IDs and metadata unreliable, need semantic/contextual analysis
   - How it works:
     1. Sends sample dialogue sequence to Gemini
     2. Gemini identifies topic changes, scene shifts, time jumps
     3. Returns boundary indices
     4. Applies boundaries to split data
   - Params:
     * sample_size (int, default=50): Number of consecutive rows to analyze
   - Pros: Most intelligent, understands context and flow
   - Cons: Sample-based (doesn't see all data), uses API call
   - Use case: When structural methods fail and need content understanding
   - Note: Ignores speaker changes (speaker changes are NORMAL in dialogue)

5. fixed_window_grouper
   - Description: Mechanically splits into fixed-size chunks (fallback)
   - Best when: All other methods fail, need guaranteed grouping
   - How it works: Every N rows becomes a session
   - Params:
     * window_size (int, default=20): Turns per session
   - Pros: Always works, predictable session sizes
   - Cons: Ignores all semantic meaning, may split mid-conversation
   - Example: Rows 0-19 → session 0, Rows 20-39 → session 1
   - Use case: Absolute fallback when nothing else works

6. hybrid_grouper
   - Description: Combines two grouping methods hierarchically
   - Best when: Need multi-level grouping (e.g., chapter → scene, or prefix → flow)
   - How it works:
     1. Apply primary method to create initial groups
     2. Apply secondary method to each group to subdivide
   - Params:
     * primary_method (str): First grouping skill name
     * secondary_method (str): Second grouping skill name
     * primary_params (dict): Parameters for primary
     * secondary_params (dict): Parameters for secondary
   - Pros: Handles complex structures, combines strengths of multiple methods
   - Cons: More complex, slower
   - Example:
     * Primary: id_prefix_grouper (group by CHAPTER)
     * Secondary: metadata_boundary_grouper (subdivide by Scene)
     * Result: CHAPTER1_scene1, CHAPTER1_scene2, CHAPTER2_scene1

**Available Validation Skills:**
- session_quality_checker - Check for too-short/long sessions
- gemini_validator - Sample-based quality review

**Important Guidelines:**
- Speaker changes are NORMAL in dialogue - do NOT use as session boundary
- Sessions represent conversation flows, not individual speakers
- Sessions can be loose/approximate groupings
- Prefer efficiency over perfect accuracy

**Your Task:**
Create a grouping execution plan. Choose 1 primary grouping skill (or hybrid).
Consider dialogue patterns and select the most appropriate approach.

**Return JSON Plan:**
{{
  "strategy": "brief strategy explanation",
  "steps": [
    {{
      "skill": "id_prefix_grouper",
      "params": {{"separator": "_"}},
      "reason": "why this skill"
    }},
    {{
      "skill": "session_quality_checker",
      "params": {{}},
      "reason": "check session quality"
    }}
  ]
}}
"""


def create_grouping_plan(
    dialogue_rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    model_name: str = 'gemini-2.5-flash',
) -> Dict[str, Any]:
    """
    Use Gemini to create a grouping execution plan.

    Args:
        dialogue_rows: Rows classified as dialogue
        schema: Column schema
        model_name: Gemini model to use

    Returns:
        Grouping execution plan
    """
    print("\n[Grouping Planner] Creating execution plan...")

    # Analyze dialogue patterns
    characteristics = analyze_dialogue_patterns(dialogue_rows, schema)

    if characteristics['total_dialogues'] == 0:
        print("  ⚠ No dialogues to group, skipping")
        return {
            'plan': {'strategy': 'No dialogues', 'steps': []},
            'characteristics': characteristics,
        }

    # Configure Gemini
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Build prompt
    prompt = build_planner_prompt(characteristics)

    print(f"  Sending to Gemini ({model_name})...")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )

    if not response or not response.text:
        raise RuntimeError("Gemini returned empty response")

    # Parse plan
    plan = json.loads(response.text.strip())

    print(f"\n  Strategy: {plan.get('strategy', 'N/A')}")
    print(f"  Steps planned: {len(plan.get('steps', []))}")
    for idx, step in enumerate(plan.get('steps', []), 1):
        condition = step.get('condition', '')
        cond_str = f" (condition: {condition})" if condition else ""
        print(f"    {idx}. {step['skill']}{cond_str}")
        print(f"       Reason: {step.get('reason', 'N/A')}")

    return {
        'plan': plan,
        'characteristics': characteristics,
    }
