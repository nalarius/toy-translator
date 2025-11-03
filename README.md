# toy-translator

## Overview
Toy Translator is a lightweight playground for taking localisation spreadsheets, transforming them into structured JSON, and enriching them with AI-assisted metadata. The repository is optimised for quick experiments: data lives under `data/`, scripts stay inside the flat `toy_translator/` package, and `uv` manages Python dependencies.

## Quick Start
1. **Install tooling**  
   ```bash
   uv sync
   ```
2. **Provide Gemini credentials**  
   Create a `.env` file in the project root with your Google AI Studio key:
   ```
   GEMINI_API_KEY=your-secret-key
   ```
   Generated artefacts default to the `tmp/` directory so the repository stays clean; override the paths with `--input`/`--output` if you prefer a different location.
3. **Convert the raw spreadsheet**  
   ```bash
   uv run python -m toy_translator.convert_dataset \
     --input data/source.xlsx \
     --output tmp/source.json
   ```

## Generate Gemini Output
Use the converted JSON to request character and dialogue structure from Gemini 2.5 Flash. The script saves a combined payload containing both character metadata and dialogue sessions, and generates standardized session files with `speaker` and `utterance` fields.

```bash
uv run python -m toy_translator.process_gemini \
  --input tmp/source.json \
  --output tmp/gemini_output.json \
  --sessions-dir tmp/sessions \
  --model gemini-2.5-flash
```

If Gemini returns malformed JSON, the raw response is stashed at `tmp/gemini_raw_response.json` for inspection.

## Aggregate Speakers
Build a speaker-centric view that merges metadata and dialogue lines. **This step extracts ALL speakers that appear in dialogue sessions**, even if they weren't explicitly defined in the character list. Multiple speakers within a single turn are split automatically, and aliases from the characters list are applied where possible.

**Recommended**: Use `--gemini-output` to read directly from Gemini's output, which contains both character metadata and all dialogue sessions:

```bash
uv run python -m toy_translator.aggregate_speakers \
  --gemini-output tmp/gemini_output.json \
  --output tmp/speakers.json
```

**Alternative**: Use the new classification-based architecture (characters may be incomplete):

```bash
uv run python -m toy_translator.aggregate_speakers \
  --classified-data tmp/classified_data.json \
  --sessions-dir tmp/sessions_raw \
  --output tmp/speakers.json
```

**Key behavior**:
- Speakers found in dialogue but not in the character list will be included with empty metadata
- A warning will be displayed for any speakers without character metadata
- This ensures all speakers get personas in the next step

## Generate Translation Personas
Create per-speaker persona briefs that summarise voice, background, and translation guidance. These artefacts are derived from the aggregated speaker data. Supports concurrent processing for faster generation.

```bash
uv run python -m toy_translator.generate_personas \
  --input tmp/speakers.json \
  --output tmp/personas.json \
  --model gemini-2.5-flash \
  --max-workers 5
```

**Features:**
- Concurrent API calls (default: 5 workers)
- Automatic retry with exponential backoff (2, 4, 8 seconds)
- Detailed progress output showing per-speaker status

## Export Session Packages
Produce per-session JSON bundles that retain the original dialogue while attaching the relevant
character personas. Each file in `tmp/sessions/` is intended for downstream translation work.

```bash
uv run python -m toy_translator.attach_personas \
  --gemini-output tmp/gemini_output.json \
  --personas tmp/personas.json \
  --output-dir tmp/sessions
```

## Generate Acting Guides
Create directorial acting guides for each dialogue session. These guides analyze the scene's emotional flow, character dynamics, pacing, and subtext to help translators understand the dramatic intent behind each line. This step produces richer, performance-oriented translations rather than literal line-by-line conversions.

```bash
# Generate acting guide for a single session
uv run python -m toy_translator.generate_acting_guide \
  --input tmp/sessions/STORY_ROMA_34.json \
  --output tmp/sessions_with_guide/STORY_ROMA_34.json \
  --model gemini-2.5-flash

# Generate acting guides for entire directory with concurrent processing
uv run python -m toy_translator.generate_acting_guide \
  --input tmp/sessions \
  --output-dir tmp/sessions_with_guide \
  --model gemini-2.5-flash \
  --max-workers 5
```

**Features:**
- Concurrent API calls (default: 5 workers)
- Automatic retry with exponential backoff (2, 4, 8 seconds)
- Analyzes emotional arc, pacing, character dynamics, and subtext
- Provides acting notes for key moments in the scene
- Outputs enriched session files with `acting_guide` field

**Acting Guide Contents:**
- **Scene Overview**: Brief description of what happens
- **Emotional Arc**: How emotions flow and change throughout the scene
- **Pacing**: Speed and rhythm of the dialogue
- **Tone**: Overall atmosphere (tense, playful, melancholic, etc.)
- **Key Moments**: Critical turning points with line ranges and acting notes
- **Character Dynamics**: Each character's role and attitude in the scene
- **Subtext**: Underlying meanings, tensions, or ironies not explicitly stated
- **Translation Notes**: Specific guidance for maintaining dramatic intent

## Translate Sessions
Feed an individual session package to Gemini for localisation. The translator preserves KEY values, uses persona English names for speakers, and keeps placeholders (e.g., {NICK}, [2561e7]). Translations are validated and auto-fixed before saving.

**Context-Aware Translation**: If the session file includes an `acting_guide` field (from the previous step), the translator automatically uses it to produce performance-oriented translations that capture emotional nuance and dramatic intent. Session files without acting guides are translated using persona information only.

```bash
# Translate a single session (with or without acting guide)
uv run python -m toy_translator.translate_session \
  --input tmp/sessions_with_guide/STORY_ROMA_34.json \
  --output tmp/translated/STORY_ROMA_34.json \
  --model gemini-2.5-flash

# Or translate from sessions without acting guides
uv run python -m toy_translator.translate_session \
  --input tmp/sessions/STORY_ROMA_34.json \
  --output tmp/translated/STORY_ROMA_34.json \
  --model gemini-2.5-flash

# Translate entire directory with concurrent processing
uv run python -m toy_translator.translate_session \
  --input tmp/sessions_with_guide \
  --output-dir tmp/translated \
  --model gemini-2.5-flash \
  --max-workers 5
```

**Features:**
- Concurrent API calls (default: 5 workers)
- Automatic retry with exponential backoff (2, 4, 8 seconds)
- Pre-save validation and auto-fixing:
  - Format tags (`[i]`, `[HEXCODE]`) preservation check
  - Placeholder (`{NICK}`, `{PLAYER}`) preservation check
  - Newline count auto-correction at sentence boundaries
- Detailed progress and validation reports

## Merge Into Dataset
Combine the translated lines and persona-derived names back into the source JSON/XLSX.

```bash
uv run python -m toy_translator.merge_translations \
  --dataset tmp/source.json \
  --personas tmp/personas.json \
  --translations-dir tmp/translated \
  --output-json tmp/source_translated.json \
  --output-xlsx tmp/source_translated.xlsx

## Full Pipeline Runner
Run every step automatically (with optional flags for dry runs, skipping stages, or resuming):

```bash
python toy-translator.py

# Example with options
python toy-translator.py --show-commands --max-utterances 40
```
```

## Environment Variables
- `GEMINI_API_KEY` *(required)*: API key for the Gemini model. Keep it in `.env` (which is git-ignored) and never commit personal keys.

For alternate datasets or outputs, adjust the `--input`/`--output` flags on each script. All commands can be chained inside shell scripts or Make targets as the project evolves.***
