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
Use the converted JSON to request character and dialogue structure from Gemini 2.5 Flash. The script saves a combined payload containing both character metadata and dialogue sessions.

```bash
uv run python -m toy_translator.process_gemini \
  --input tmp/source.json \
  --output tmp/gemini_output.json \
  --model gemini-2.5-flash
```

If Gemini returns malformed JSON, the raw response is stashed at `tmp/gemini_raw_response.json` for inspection.

## Aggregate Speakers
Build a speaker-centric view that merges metadata and dialogue lines. Multiple speakers within a single turn are split automatically, and aliases from the characters list are applied where possible.

```bash
uv run python -m toy_translator.aggregate_speakers \
  --input tmp/gemini_output.json \
  --output tmp/speakers.json
```

## Generate Translation Personas
Create per-speaker persona briefs that summarise voice, background, and translation guidance. These artefacts are derived from the aggregated speaker data.

```bash
uv run python -m toy_translator.generate_personas \
  --input tmp/speakers.json \
  --output tmp/personas.json \
  --model gemini-2.5-flash
```

## Export Session Packages
Produce per-session JSON bundles that retain the original dialogue while attaching the relevant
character personas. Each file in `tmp/sessions/` is intended for downstream translation work.

```bash
uv run python -m toy_translator.attach_personas \
  --gemini-output tmp/gemini_output.json \
  --personas tmp/personas.json \
  --output-dir tmp/sessions
```

## Environment Variables
- `GEMINI_API_KEY` *(required)*: API key for the Gemini model. Keep it in `.env` (which is git-ignored) and never commit personal keys.

For alternate datasets or outputs, adjust the `--input`/`--output` flags on each script. All commands can be chained inside shell scripts or Make targets as the project evolves.***
