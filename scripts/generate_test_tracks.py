#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]

from audiocraft.data.audio import audio_write  # type: ignore[import-not-found]
from audiocraft.models import MusicGen  # type: ignore[import-not-found]
from audiocraft.utils import export  # type: ignore[import-not-found]


REQUIRED_FIELDS: tuple[str, ...] = (
    "description",
    "general_mood",
    "genre_tags",
    "lead_instrument",
    "accompaniment",
    "tempo_and_rhythm",
    "vocal_presence",
    "production_quality",
)


def prompt_to_text(prompt: dict[str, Any]) -> str:
    """Flatten structured prompt to training-like text format."""
    fields = [
        ("description", prompt["description"]),
        ("general_mood", prompt["general_mood"]),
        ("genre_tags", ", ".join(prompt["genre_tags"])),
        ("lead_instrument", prompt["lead_instrument"]),
        ("accompaniment", prompt["accompaniment"]),
        ("tempo_and_rhythm", prompt["tempo_and_rhythm"]),
        ("vocal_presence", prompt["vocal_presence"]),
        ("production_quality", prompt["production_quality"]),
    ]
    return ". ".join([f"{k}: {v}" for k, v in fields])


def load_prompts(prompts_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    prompt_files = sorted(prompts_dir.glob("*.json"))
    if not prompt_files:
        raise FileNotFoundError(f"No .json prompt files found in: {prompts_dir}")

    prompts: list[tuple[str, dict[str, Any]]] = []
    for path in prompt_files:
        with path.open("r", encoding="utf-8") as f:
            prompt = json.load(f)
        if not isinstance(prompt, dict):
            raise ValueError(f"Prompt file must contain JSON object: {path}")

        missing = [field for field in REQUIRED_FIELDS if field not in prompt]
        if missing:
            raise ValueError(f"Prompt file {path} is missing required fields: {', '.join(missing)}")

        if not isinstance(prompt.get("genre_tags"), list):
            raise ValueError(f"'genre_tags' must be a list in prompt file: {path}")

        prompts.append((path.stem, prompt))
    return prompts


def resolve_model_path(model_path: str, export_dir: Path) -> str:
    """
    Accept either:
    - exported directory with state_dict.bin and compression_state_dict.bin
    - training checkpoint *.th (auto-export)
    - pretrained id (facebook/musicgen-small, etc.)
    """
    p = Path(model_path)
    if not p.exists():
        return model_path

    if p.is_dir():
        lm = p / "state_dict.bin"
        comp = p / "compression_state_dict.bin"
        if lm.exists() and comp.exists():
            return str(p)
        raise ValueError(
            f"Directory exists but missing exported files: {p}. "
            "Expected state_dict.bin and compression_state_dict.bin."
        )

    if p.suffix == ".th":
        export_dir.mkdir(parents=True, exist_ok=True)
        lm_path = export_dir / "state_dict.bin"
        comp_path = export_dir / "compression_state_dict.bin"
        if not lm_path.exists():
            export.export_lm(p, lm_path)
        if not comp_path.exists():
            export.export_pretrained_compression_model("facebook/encodec_32khz", comp_path)
        return str(export_dir)

    return model_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to exported AudioCraft model directory or pretrained model name.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("generated_test_tracks"))
    parser.add_argument("--duration", type=int, default=12, help="Track duration in seconds (10-15 recommended).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--auto-export-dir",
        type=Path,
        default=Path("auto_export_model"),
        help="Where to auto-export if --model-path points to *.th checkpoint.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path("prompts"),
        help="Directory with JSON prompt files.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    prompts = load_prompts(args.prompts_dir)

    resolved_model_path = resolve_model_path(args.model_path, args.auto_export_dir)
    model = MusicGen.get_pretrained(resolved_model_path, device=args.device)
    model.set_generation_params(duration=args.duration, top_k=args.top_k, top_p=0.0, temperature=args.temperature)

    for stem_name, prompt in prompts:
        prompt_text = prompt_to_text(prompt)
        wav = model.generate([prompt_text], progress=True)[0].cpu()

        stem = args.output_dir / stem_name
        audio_write(stem, wav, model.sample_rate, strategy="loudness")

        with (args.output_dir / f"{stem_name}.json").open("w", encoding="utf-8") as f:
            json.dump(prompt, f, ensure_ascii=False, indent=2)

        with (args.output_dir / f"{stem_name}.txt").open("w", encoding="utf-8") as f:
            f.write(prompt_text + "\n")

        print(f"[OK] {stem_name} generated")

    print(f"Done. Files saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
