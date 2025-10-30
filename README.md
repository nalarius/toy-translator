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
3. **Convert the raw spreadsheet**  
   ```bash
   uv run python -m toy_translator.convert_dataset \
     --input data/source.xlsx \
     --output data/source.json
   ```

## Generate Gemini Output
Use the converted JSON to request character and dialogue structure from Gemini 2.5 Flash. The script saves a combined payload containing both character metadata and dialogue sessions.

```bash
uv run python -m toy_translator.process_gemini \
  --input data/source.json \
  --output data/gemini_output.json \
  --model gemini-2.5-flash
```

If Gemini returns malformed JSON, the raw response is stashed at `tmp/gemini_raw_response.json` for inspection.

## Aggregate Speakers
Build a speaker-centric view that merges metadata and dialogue lines. Multiple speakers within a single turn are split automatically, and aliases from the characters list are applied where possible.

```bash
uv run python -m toy_translator.aggregate_speakers \
  --input data/gemini_output.json \
  --output data/speakers.json
```

## Environment Variables
- `GEMINI_API_KEY` *(required)*: API key for the Gemini model. Keep it in `.env` (which is git-ignored) and never commit personal keys.

For alternate datasets or outputs, adjust the `--input`/`--output` flags on each script. All commands can be chained inside shell scripts or Make targets as the project evolves.***
