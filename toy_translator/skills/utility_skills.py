"""Utility skills for validation, merging, and analysis."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv


def coverage_checker(
    original_rows: List[Dict[str, Any]],
    classification: Dict[str, List[int]],
    unclassified: List[int] = None,
    unmatched: List[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Verify that all rows have been classified.

    Args:
        original_rows: Original list of rows
        classification: Classification dict (type -> indices)
        unclassified: List of unclassified indices (optional)
        unmatched: List of unmatched indices (optional)

    Returns:
        Coverage report
    """
    print(f"\n[coverage_checker] Verifying classification coverage...")

    total_rows = len(original_rows)

    # Collect all classified indices
    classified_indices = set()
    for type_name, indices in classification.items():
        classified_indices.update(indices)

    # Check for unclassified
    unclassified = unclassified or []
    unmatched = unmatched or []
    missing = list(set(unclassified) | set(unmatched))

    coverage = len(classified_indices) / total_rows if total_rows > 0 else 0

    print(f"  Total rows: {total_rows}")
    print(f"  Classified: {len(classified_indices)} ({coverage:.1%})")

    if missing:
        print(f"  ⚠ Missing: {len(missing)} rows")
        print(f"    Indices: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        print(f"  ✓ 100% coverage")

    # Check for duplicates
    all_indices = []
    for indices in classification.values():
        all_indices.extend(indices)

    duplicates = [idx for idx, count in Counter(all_indices).items() if count > 1]

    if duplicates:
        print(f"  ⚠ Duplicate classifications: {len(duplicates)} rows")
        print(f"    Indices: {duplicates[:10]}{'...' if len(duplicates) > 10 else ''}")
    else:
        print(f"  ✓ No duplicates")

    return {
        'total': total_rows,
        'classified': len(classified_indices),
        'missing': missing,
        'duplicates': duplicates,
        'coverage': coverage,
        'passed': len(missing) == 0 and len(duplicates) == 0
    }


def session_quality_checker(
    sessions: Dict[str, List[Dict[str, Any]]],
    text_column: str = 'Korean',
    **kwargs
) -> Dict[str, Any]:
    """
    Check the quality of grouped sessions.

    Args:
        sessions: Dictionary of session_id -> rows
        text_column: Name of text field

    Returns:
        Quality report
    """
    print(f"\n[session_quality_checker] Checking session quality...")

    issues = []
    session_sizes = []

    for session_id, turns in sessions.items():
        size = len(turns)
        session_sizes.append(size)

        # Check for very short sessions
        if size < 3:
            issues.append({
                'session_id': session_id,
                'issue': f'Very short session ({size} turns)',
                'severity': 'low'
            })

        # Check for very long sessions
        elif size > 50:
            issues.append({
                'session_id': session_id,
                'issue': f'Very long session ({size} turns)',
                'severity': 'medium'
            })

        # Check speaker distribution (if speaker field exists)
        speakers = [turn.get('speaker') for turn in turns if turn.get('speaker')]
        if speakers and len(set(speakers)) == 1:
            issues.append({
                'session_id': session_id,
                'issue': f'Only one speaker in session',
                'severity': 'low'
            })

    # Statistics
    if session_sizes:
        avg_size = sum(session_sizes) / len(session_sizes)
        min_size = min(session_sizes)
        max_size = max(session_sizes)
    else:
        avg_size = min_size = max_size = 0

    print(f"  Total sessions: {len(sessions)}")
    print(f"  Session size: avg={avg_size:.1f}, min={min_size}, max={max_size}")

    if issues:
        print(f"  ⚠ Issues found: {len(issues)}")
        # Group by severity
        high = [i for i in issues if i['severity'] == 'high']
        medium = [i for i in issues if i['severity'] == 'medium']
        low = [i for i in issues if i['severity'] == 'low']

        if high:
            print(f"    High severity: {len(high)}")
        if medium:
            print(f"    Medium severity: {len(medium)}")
        if low:
            print(f"    Low severity: {len(low)}")

        # Show samples
        for issue in issues[:3]:
            print(f"    - {issue['session_id']}: {issue['issue']}")
    else:
        print(f"  ✓ All sessions look healthy")

    return {
        'total_sessions': len(sessions),
        'avg_size': avg_size,
        'min_size': min_size,
        'max_size': max_size,
        'issues': issues,
        'passed': len([i for i in issues if i['severity'] == 'high']) == 0
    }


def vote_merger(
    classification_results: List[Dict[str, Any]],
    original_rows: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Merge multiple classification results by voting.

    Args:
        classification_results: List of classification results
        original_rows: Original rows

    Returns:
        Merged classification
    """
    print(f"\n[vote_merger] Merging {len(classification_results)} classification results...")

    # Vote for each row
    merged_classification: Dict[str, List[int]] = {}

    for row_idx in range(len(original_rows)):
        votes: List[str] = []

        # Collect votes from each result
        for result in classification_results:
            classification = result.get('classification', {})
            for type_name, indices in classification.items():
                if row_idx in indices:
                    votes.append(type_name)
                    break

        # Determine winner
        if votes:
            vote_counts = Counter(votes)
            winner, count = vote_counts.most_common(1)[0]

            # Require at least 2 votes for consensus (if we have 3+ results)
            if len(classification_results) >= 3 and count < 2:
                print(f"  ⚠ Row {row_idx}: No consensus (votes: {dict(vote_counts)})")
                merged_classification.setdefault('uncertain', []).append(row_idx)
            else:
                merged_classification.setdefault(winner, []).append(row_idx)
        else:
            merged_classification.setdefault('unclassified', []).append(row_idx)

    # Report
    print(f"  Merged results:")
    for type_name, indices in sorted(merged_classification.items()):
        print(f"    {type_name}: {len(indices)} rows")

    return {
        'classification': merged_classification,
        'method': 'vote_merger'
    }


def confidence_merger(
    classification_results_with_confidence: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Merge classification results by selecting highest confidence.

    Args:
        classification_results_with_confidence: List of results with confidence scores

    Returns:
        Merged classification
    """
    print(f"\n[confidence_merger] Selecting highest confidence results...")

    # Sort by confidence
    sorted_results = sorted(
        classification_results_with_confidence,
        key=lambda x: x.get('confidence', 0),
        reverse=True
    )

    # Take highest confidence
    best_result = sorted_results[0]

    print(f"  Selected: {best_result.get('method', 'unknown')}")
    print(f"  Confidence: {best_result.get('confidence', 0):.0%}")

    classification = best_result.get('classification', {})
    for type_name, indices in sorted(classification.items()):
        print(f"    {type_name}: {len(indices)} rows")

    return best_result


def gemini_validator(
    classification_or_sessions: Dict[str, Any],
    original_rows: List[Dict[str, Any]],
    sample_count: int = 5,
    model_name: str = 'gemini-2.5-flash',
    **kwargs
) -> Dict[str, Any]:
    """
    Use Gemini to validate classification or session quality.

    Args:
        classification_or_sessions: Classification result or sessions
        original_rows: Original rows
        sample_count: Number of samples to validate
        model_name: Gemini model to use

    Returns:
        Validation report
    """
    print(f"\n[gemini_validator] Validating with Gemini (sample size: {sample_count})...")

    # Configure Gemini
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Determine what we're validating
    if 'classification' in classification_or_sessions:
        # Validating classification
        validation_type = 'classification'
        data = classification_or_sessions['classification']
    elif 'sessions' in classification_or_sessions:
        # Validating sessions
        validation_type = 'sessions'
        data = classification_or_sessions['sessions']
    else:
        print(f"  ⚠ Unknown data type for validation")
        return {'passed': False, 'issues': 'Unknown data type'}

    # Sample and prepare for Gemini
    import random
    sampled_items = random.sample(list(data.items()), min(sample_count, len(data)))

    sample_data = []
    for key, value in sampled_items:
        if validation_type == 'classification':
            # value is list of indices
            sample_rows = [original_rows[idx] for idx in value[:5]]  # Max 5 rows per type
            sample_data.append({'type': key, 'sample_rows': sample_rows})
        else:
            # value is list of rows
            sample_data.append({'session_id': key, 'turns': value[:10]})  # Max 10 turns

    prompt = f"""
Review this {validation_type} for quality.

**Sample Data:**
{json.dumps(sample_data, ensure_ascii=False, indent=2)}

**Validation Criteria:**
- Are items correctly categorized/grouped?
- Any obvious errors?
- Overall quality assessment

**Return JSON:**
{{
  "overall_quality": "excellent/good/fair/poor",
  "issues": ["issue 1", "issue 2", ...],
  "suggestions": ["suggestion 1", ...],
  "passed": true/false
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
        print(f"  ⚠ Empty response")
        return {'passed': False}

    result = json.loads(response.text.strip())

    print(f"  Quality: {result.get('overall_quality', 'unknown')}")
    print(f"  Passed: {result.get('passed', False)}")

    if result.get('issues'):
        print(f"  Issues: {len(result['issues'])}")
        for issue in result['issues'][:3]:
            print(f"    - {issue}")

    return result


def analyze_fields(
    rows: List[Dict[str, Any]],
    schema: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze field statistics.

    Args:
        rows: List of rows
        schema: Column schema

    Returns:
        Field statistics
    """
    print(f"\n[analyze_fields] Analyzing field statistics...")

    stats = {}
    total_rows = len(rows)

    # Get all field names
    all_fields = set()
    for row in rows:
        all_fields.update(row.keys())

    for field in all_fields:
        values = [row.get(field) for row in rows]
        non_null_values = [v for v in values if v is not None and v != '']

        unique_values = list(set(non_null_values))
        null_count = total_rows - len(non_null_values)
        null_ratio = null_count / total_rows if total_rows > 0 else 0

        stats[field] = {
            'unique_count': len(unique_values),
            'null_count': null_count,
            'null_ratio': null_ratio,
            'samples': unique_values[:5]
        }

    # Print summary
    for field, field_stats in sorted(stats.items()):
        print(f"  {field}:")
        print(f"    Unique: {field_stats['unique_count']}, Null: {field_stats['null_ratio']:.1%}")
        print(f"    Samples: {field_stats['samples']}")

    return stats
