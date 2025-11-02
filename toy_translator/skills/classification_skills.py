"""Classification skills for categorizing data rows."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv


def metadata_type_classifier(
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    type_field: str = 'type',
    **kwargs
) -> Dict[str, Any]:
    """
    Classify rows using an existing type/category field from metadata.

    This is the fastest and most accurate classification method when a type
    field already exists in the data with good coverage.

    How it works:
    - Directly reads the specified type field from each row
    - Uses the field value as the classification type
    - Normalizes values (lowercase, trim)
    - Tracks unclassified rows (missing or empty type field)

    Best when:
    - Type field exists with >70% coverage
    - Type values are already meaningful (dialogue, character, etc.)

    Args:
        rows: List of data rows
        schema: Column schema
        type_field: Name of the type field to use (e.g., 'Type', 'Category')

    Returns:
        {
            'classification': {
                'dialogue': [row_indices],
                'character': [row_indices],
                ...
            },
            'unclassified': [row_indices],
            'coverage': 0.95,
            'method': 'metadata_type_classifier'
        }

    Example:
        If row has Type='Dialogue', classify as 'dialogue'
        If row has Type='Character', classify as 'character'
    """
    print(f"\n[metadata_type_classifier] Using field: '{type_field}'")

    classification: Dict[str, List[int]] = {}
    unclassified: List[int] = []

    for idx, row in enumerate(rows):
        type_value = row.get(type_field)

        if type_value and isinstance(type_value, str):
            type_normalized = type_value.strip().lower()
            classification.setdefault(type_normalized, []).append(idx)
        else:
            unclassified.append(idx)

    # Report results
    total = len(rows)
    classified_count = total - len(unclassified)
    coverage = classified_count / total if total > 0 else 0

    print(f"  Classified: {classified_count}/{total} rows ({coverage:.1%})")
    print(f"  Types found:")
    for type_name, indices in sorted(classification.items()):
        print(f"    {type_name}: {len(indices)} rows")

    if unclassified:
        print(f"  Unclassified: {len(unclassified)} rows")

    return {
        'classification': classification,
        'unclassified': unclassified,
        'coverage': coverage,
        'method': 'metadata_type_classifier'
    }


def id_pattern_classifier(
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    patterns: Dict[str, str],
    unique_key: str = 'KEY',
    **kwargs
) -> Dict[str, Any]:
    """
    Classify rows using ID pattern matching.

    Args:
        rows: List of data rows
        schema: Column schema
        patterns: Dict of pattern -> type mapping
                  e.g., {'CHARACTER_*': 'character', 'STORY_*': 'dialogue'}
        unique_key: Name of the unique ID field

    Returns:
        Classification result
    """
    print(f"\n[id_pattern_classifier] Patterns: {patterns}")

    classification: Dict[str, List[int]] = {}
    unmatched: List[int] = []

    # Convert glob patterns to regex
    pattern_regexes = []
    for pattern, type_name in patterns.items():
        regex_pattern = pattern.replace('*', '.*')
        pattern_regexes.append((re.compile(f'^{regex_pattern}$'), type_name))

    for idx, row in enumerate(rows):
        id_value = row.get(unique_key)

        if not id_value or not isinstance(id_value, str):
            unmatched.append(idx)
            continue

        matched = False
        for regex, type_name in pattern_regexes:
            if regex.match(id_value):
                classification.setdefault(type_name, []).append(idx)
                matched = True
                break

        if not matched:
            unmatched.append(idx)

    # Report results
    total = len(rows)
    matched_count = total - len(unmatched)
    coverage = matched_count / total if total > 0 else 0

    print(f"  Matched: {matched_count}/{total} rows ({coverage:.1%})")
    print(f"  Types found:")
    for type_name, indices in sorted(classification.items()):
        print(f"    {type_name}: {len(indices)} rows")

    if unmatched:
        print(f"  Unmatched: {len(unmatched)} rows")

    return {
        'classification': classification,
        'unmatched': unmatched,
        'coverage': coverage,
        'method': 'id_pattern_classifier'
    }


def id_gemini_classifier(
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    unique_key: str = 'KEY',
    sample_size: int = 50,
    model_name: str = 'gemini-2.5-flash',
    **kwargs
) -> Dict[str, Any]:
    """
    Use Gemini to analyze ID samples and infer classification patterns.

    This method automatically discovers patterns in structured IDs without
    requiring manual pattern definition. Uses Gemini once to analyze samples,
    then applies discovered patterns to all rows.

    How it works:
    1. Extract sample IDs from the data
    2. Send samples to Gemini for analysis
    3. Gemini identifies patterns (e.g., "IDs starting with NPC_ are characters")
    4. Returns pattern rules (e.g., {'NPC_*': 'character', 'QUEST_*': 'dialogue'})
    5. Applies these patterns to all rows using id_pattern_classifier

    Best when:
    - IDs have structure but patterns are unclear
    - You want automatic pattern discovery
    - IDs are more reliable than text content

    Args:
        rows: List of data rows
        schema: Column schema
        unique_key: Name of the unique ID field
        sample_size: Number of samples to send to Gemini (default: 50)
        model_name: Gemini model to use (default: gemini-2.5-flash)

    Returns:
        {
            'classification': {...},
            'unmatched': [row_indices],
            'coverage': 0.95,
            'inferred_patterns': [pattern objects from Gemini],
            'method': 'id_gemini_classifier'
        }

    Cost: 1 API call (analyzes samples only, not all data)
    """
    print(f"\n[id_gemini_classifier] Analyzing {min(sample_size, len(rows))} ID samples...")

    # Extract ID samples
    id_samples = []
    for idx, row in enumerate(rows[:sample_size]):
        id_value = row.get(unique_key)
        if id_value:
            id_samples.append({'index': idx, 'id': str(id_value)})

    # Configure Gemini
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Build prompt
    prompt = f"""
Analyze these ID samples from a game localization dataset and infer classification rules.

**ID Samples:**
{json.dumps(id_samples, ensure_ascii=False, indent=2)}

**Task:**
Identify patterns in the IDs and classify them into types:
- dialogue: conversational content
- character: character descriptions/metadata
- narrative: story narration
- system: UI/tutorial/system messages
- other: anything else

**Return JSON:**
{{
  "patterns": [
    {{"pattern": "CHARACTER_*", "type": "character", "confidence": 0.95}},
    {{"pattern": "STORY_*", "type": "dialogue", "confidence": 0.90}},
    ...
  ],
  "reasoning": "explanation of patterns found"
}}
"""

    print(f"  Sending to Gemini...")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    if not response or not response.text:
        raise RuntimeError("Gemini returned empty response")

    # Parse response
    result = json.loads(response.text.strip())
    patterns = {p['pattern']: p['type'] for p in result.get('patterns', [])}

    print(f"  Gemini identified {len(patterns)} patterns:")
    for pattern, type_name in patterns.items():
        print(f"    {pattern} → {type_name}")
    print(f"  Reasoning: {result.get('reasoning', 'N/A')}")

    # Apply inferred patterns to all rows
    classification_result = id_pattern_classifier(
        rows, schema, patterns, unique_key
    )

    classification_result['inferred_patterns'] = result.get('patterns', [])
    classification_result['method'] = 'id_gemini_classifier'

    return classification_result


def text_gemini_classifier(
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    text_column: str = 'Korean',
    sample_size: int = 30,
    batch_size: int = 50,
    model_name: str = 'gemini-2.5-flash',
    **kwargs
) -> Dict[str, Any]:
    """
    Use Gemini to classify rows by analyzing text content directly.

    This is the most flexible but also most expensive classification method.
    It processes all data in batches, sending text content to Gemini for
    intelligent classification. Use as last resort when metadata and ID-based
    methods are insufficient.

    How it works:
    1. Divide data into batches (e.g., 50 rows per batch)
    2. For each batch, send text samples to Gemini
    3. Gemini classifies each row based on content
    4. Accumulate results across all batches

    Best when:
    - No metadata or ID patterns available
    - Text content is the only reliable classification signal
    - Need intelligent, content-based classification

    WARNING: Expensive for large datasets!
    - Number of API calls = ceil(total_rows / batch_size)
    - For 1000 rows with batch_size=50: 20 API calls

    Args:
        rows: List of data rows
        schema: Column schema
        text_column: Name of the text field to analyze
        sample_size: Unused (kept for compatibility)
        batch_size: Rows per batch (default: 50) - controls API usage
        model_name: Gemini model to use (default: gemini-2.5-flash)

    Returns:
        {
            'classification': {...},
            'coverage': 0.95,
            'method': 'text_gemini_classifier'
        }

    Cost: Multiple API calls (scales with data size)
    """
    print(f"\n[text_gemini_classifier] Analyzing text content...")
    print(f"  Sample size: {min(sample_size, len(rows))}")

    # Configure Gemini
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    classification: Dict[str, List[int]] = {}

    # Process in batches
    total_batches = (len(rows) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(rows))
        batch = rows[start_idx:end_idx]

        print(f"  Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} rows)...")

        # Prepare batch for Gemini
        batch_data = []
        for idx, row in enumerate(batch, start=start_idx):
            text = row.get(text_column, '')
            batch_data.append({'index': idx, 'text': str(text)[:200]})  # Limit text length

        # Build prompt
        prompt = f"""
Classify these text entries from a game localization dataset.

**Entries:**
{json.dumps(batch_data, ensure_ascii=False, indent=2)}

**Categories:**
- dialogue: conversational exchanges
- character: character descriptions
- narrative: story narration, scene descriptions
- system: UI text, tutorials, help messages
- other: anything else

**Return JSON:**
{{
  "classifications": [
    {{"index": 0, "type": "dialogue"}},
    {{"index": 1, "type": "character"}},
    ...
  ]
}}
"""

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        if not response or not response.text:
            print(f"    ⚠ Empty response for batch {batch_idx + 1}, skipping")
            continue

        # Parse and accumulate results
        result = json.loads(response.text.strip())
        for item in result.get('classifications', []):
            idx = item['index']
            type_name = item['type']
            classification.setdefault(type_name, []).append(idx)

    # Report results
    total = len(rows)
    classified_count = sum(len(indices) for indices in classification.values())
    coverage = classified_count / total if total > 0 else 0

    print(f"  Classified: {classified_count}/{total} rows ({coverage:.1%})")
    print(f"  Types found:")
    for type_name, indices in sorted(classification.items()):
        print(f"    {type_name}: {len(indices)} rows")

    return {
        'classification': classification,
        'coverage': coverage,
        'method': 'text_gemini_classifier'
    }
