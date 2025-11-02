"""Classification Planner - decides which classification skills to use."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv


def analyze_data_characteristics(
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    field_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze data to understand classification requirements.

    Args:
        rows: All data rows
        schema: Column schema
        field_stats: Field statistics from analyze_fields

    Returns:
        Data characteristics summary
    """
    print("\n[Classification Planner] Analyzing data characteristics...")

    unique_key = schema.get('unique_id', 'KEY')

    # Check for type/category fields in metadata
    metadata = schema.get('metadata', {})
    potential_type_fields = []

    for category, fields in metadata.items():
        if category in ['tags', 'other']:
            for field in fields:
                if 'type' in field.lower() or 'category' in field.lower():
                    stats = field_stats.get(field, {})
                    null_ratio = stats.get('null_ratio', 1.0)
                    if null_ratio < 0.5:  # At least 50% populated
                        potential_type_fields.append({
                            'field': field,
                            'null_ratio': null_ratio,
                            'unique_count': stats.get('unique_count', 0)
                        })

    # Analyze ID patterns
    id_samples = []
    for idx, row in enumerate(rows[:50]):
        id_value = row.get(unique_key)
        if id_value:
            id_samples.append(str(id_value))

    has_structured_ids = False
    if id_samples:
        # Check if IDs have common patterns
        has_underscores = sum('_' in id_val for id_val in id_samples) / len(id_samples) > 0.8
        has_prefixes = len(set(id_val.split('_')[0] for id_val in id_samples if '_' in id_val)) < len(id_samples) * 0.3
        has_structured_ids = has_underscores and has_prefixes

    # Text column info
    text_columns = schema.get('text_columns', {})
    primary_text = text_columns.get('primary')

    characteristics = {
        'total_rows': len(rows),
        'has_type_field': len(potential_type_fields) > 0,
        'type_fields': potential_type_fields,
        'has_structured_ids': has_structured_ids,
        'id_samples': id_samples[:10],
        'has_text_content': bool(primary_text),
        'unique_key': unique_key,
        'primary_text_column': primary_text,
    }

    print(f"  Total rows: {characteristics['total_rows']}")
    print(f"  Type fields found: {len(potential_type_fields)}")
    if potential_type_fields:
        for tf in potential_type_fields:
            print(f"    - {tf['field']} (null: {tf['null_ratio']:.1%}, unique: {tf['unique_count']})")
    print(f"  Structured IDs: {has_structured_ids}")
    print(f"  Has text content: {characteristics['has_text_content']}")

    return characteristics


def build_planner_prompt(
    characteristics: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
) -> str:
    """Build prompt for the classification planner."""

    return f"""You are a classification strategy planner for game localization data.

**Data Characteristics:**
{json.dumps(characteristics, ensure_ascii=False, indent=2)}

**Sample Rows (first 5):**
{json.dumps(sample_rows[:5], ensure_ascii=False, indent=2)}

**Available Classification Skills:**

1. metadata_type_classifier
   - Description: Uses an existing type/category field from the data's metadata
   - Best when: Type field exists with good coverage (>70%)
   - How it works: Directly reads the type field value and uses it as classification
   - Params:
     * type_field (str): Name of the field to use (e.g., 'Type', 'Category')
   - Pros: Fastest, most accurate (uses existing metadata)
   - Cons: Requires pre-existing type field
   - Example: If row has Type='dialogue', classify as 'dialogue'

2. id_pattern_classifier
   - Description: Matches ID patterns using regex rules
   - Best when: IDs have clear, consistent prefixes/patterns
   - How it works: Applies regex patterns to ID field, matches to types
   - Params:
     * patterns (dict): Pattern->type mappings, e.g., {{'CHARACTER_*': 'character', 'STORY_*': 'dialogue'}}
   - Pros: Fast, deterministic, no API calls
   - Cons: Requires knowing patterns in advance
   - Example: ID 'CHARACTER_ROMA_01' matches 'CHARACTER_*' → 'character'

3. id_gemini_classifier
   - Description: Uses Gemini to analyze ID samples and infer classification patterns
   - Best when: IDs are structured but patterns are not obvious
   - How it works:
     1. Sends sample IDs to Gemini
     2. Gemini infers patterns (e.g., "IDs starting with NPC_ are characters")
     3. Applies inferred patterns to all rows using id_pattern_classifier
   - Params:
     * sample_size (int, default=50): Number of sample IDs to analyze
   - Pros: Automatic pattern discovery, no manual pattern definition needed
   - Cons: Single API call, relies on ID structure being meaningful
   - Use case: When you have structured IDs but don't want to manually define patterns

4. text_gemini_classifier
   - Description: Uses Gemini to classify rows by analyzing text content directly
   - Best when: No metadata or ID patterns available, need content-based classification
   - How it works: Processes data in batches, sends text to Gemini for classification
   - Params:
     * batch_size (int, default=50): Rows per batch (controls API usage)
   - Pros: Most flexible, works without any metadata/structure, most intelligent
   - Cons: Slowest, most expensive (multiple API calls), scales with data size
   - Use case: Last resort when metadata and ID-based methods won't work
   - Warning: Can be expensive for large datasets (uses multiple API calls)

**Available Validation Skills:**
- coverage_checker - Verify 100% classification
- gemini_validator - Sample-based quality review

**Your Task:**
Create a classification execution plan. Choose 1-2 classification skills (primary + optional fallback).
Consider data characteristics and select the most efficient approach.

**Target Categories:**
- dialogue: conversational exchanges
- character: character descriptions/profiles
- narrative: story narration, scene descriptions
- system: UI text, tutorials, menus
- other: anything else

**Return JSON Plan:**
{{
  "strategy": "brief strategy explanation",
  "steps": [
    {{
      "skill": "metadata_type_classifier",
      "params": {{"type_field": "Type"}},
      "reason": "why this skill"
    }},
    {{
      "skill": "coverage_checker",
      "params": {{}},
      "reason": "verify coverage"
    }},
    {{
      "skill": "text_gemini_classifier",
      "params": {{"batch_size": 50}},
      "condition": "if coverage < 0.9",
      "reason": "fallback for unclassified rows"
    }}
  ]
}}
"""


def create_classification_plan(
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    field_stats: Dict[str, Any],
    model_name: str = 'gemini-2.5-flash',
) -> Dict[str, Any]:
    """
    Use Gemini to create a classification execution plan.

    Args:
        rows: All data rows
        schema: Column schema
        field_stats: Field statistics
        model_name: Gemini model to use

    Returns:
        Classification execution plan
    """
    print("\n[Classification Planner] Creating execution plan...")

    # Analyze data
    characteristics = analyze_data_characteristics(rows, schema, field_stats)

    # Configure Gemini
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Build prompt
    prompt = build_planner_prompt(characteristics, rows)

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
