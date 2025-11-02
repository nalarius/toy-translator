#!/usr/bin/env python3
"""End-to-end orchestration for the Toy Translator pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence


class PipelineError(RuntimeError):
    """Raised when a pipeline step fails."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full Toy Translator workflow from source XLSX to merged outputs."
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=0,
        help=(
            "Limit the number of utterances passed to persona generation "
            "(0 keeps all lines; default: 0)."
        ),
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip the session translation stage.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the final dataset merge stage.",
    )
    parser.add_argument(
        "--resume-from",
        choices=[
            "convert",
            "process",
            "aggregate",
            "personas",
            "attach",
            "translate",
            "merge",
        ],
        help="Resume the pipeline from the specified stage (inclusive).",
    )
    parser.add_argument(
        "--show-commands",
        action="store_true",
        help="Print the command for each step before execution.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the steps that would run without executing them.",
    )
    return parser


def step(label: str, *command: str) -> tuple[str, List[str]]:
    return label, list(command)


def build_steps(args: argparse.Namespace) -> List[tuple[str, List[str]]]:
    max_utts = str(args.max_utterances)
    schema_path = "tmp/column_schema.json"

    commands: List[tuple[str, List[str]]] = [
        step("Convert XLSX to JSON and classify columns",
             sys.executable, "-m", "toy_translator.convert_dataset"),
        step(
            "Extract sessions via Gemini",
            sys.executable,
            "-m",
            "toy_translator.process_gemini",
            "--schema",
            schema_path,
        ),
        step(
            "Aggregate speakers",
            sys.executable,
            "-m",
            "toy_translator.aggregate_speakers",
            "--schema",
            schema_path,
        ),
        step(
            "Generate personas",
            sys.executable,
            "-m",
            "toy_translator.generate_personas",
            "--max-utterances",
            max_utts,
            "--schema",
            schema_path,
        ),
        step(
            "Attach personas to sessions",
            sys.executable,
            "-m",
            "toy_translator.attach_personas",
        ),
    ]

    if not args.skip_translate:
        commands.append(
            step(
                "Translate sessions",
                sys.executable,
                "-m",
                "toy_translator.translate_session",
                "--input",
                "tmp/sessions",
                "--schema",
                schema_path,
            )
        )

    if not args.skip_merge:
        commands.append(
            step(
                "Merge translations into dataset",
                sys.executable,
                "-m",
                "toy_translator.merge_translations",
                "--schema",
                schema_path,
            )
        )

    if args.resume_from:
        resume_order = [
            "convert",
            "process",
            "aggregate",
            "personas",
            "attach",
            "translate",
            "merge",
        ]
        resume_idx = resume_order.index(args.resume_from)
        commands = commands[resume_idx:]

    return commands


def format_command(command: Sequence[str]) -> str:
    return " ".join(f'"{arg}"' if " " in arg else arg for arg in command)


def run_step(
    index: int,
    total: int,
    label: str,
    command: List[str],
    *,
    show_command: bool,
) -> None:
    start = time.time()
    print(f"[{index}/{total}] {label}")
    if show_command:
        print(f"    Command: {format_command(command)}")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"Step '{label}' failed with exit code {exc.returncode}.") from exc

    elapsed = time.time() - start
    print(f"    ✅ Completed in {elapsed:.1f}s")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    steps = build_steps(args)
    if not steps:
        print("No steps scheduled. Did you skip everything?")
        return

    if args.dry_run:
        print("Planned steps:")
        for idx, (label, command) in enumerate(steps, start=1):
            print(f"  {idx}. {label}")
            if args.show_commands:
                print(f"     {format_command(command)}")
        return

    overall_start = time.time()
    total = len(steps)

    for idx, (label, command) in enumerate(steps, start=1):
        run_step(idx, total, label, command, show_command=args.show_commands)

    overall_elapsed = time.time() - overall_start
    print(f"\n🎉 Pipeline completed in {overall_elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except PipelineError as error:
        print(f"❌ {error}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)

