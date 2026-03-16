import argparse
import gzip
import json
import random
import wave
from pathlib import Path


def wav_meta(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        sample_rate = int(wf.getframerate())
        frames = int(wf.getnframes())
    duration = frames / float(sample_rate) if sample_rate > 0 else 0.0
    return {"path": str(path.resolve()), "duration": duration, "sample_rate": sample_rate}


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=Path, default=Path("data/wav"))
    parser.add_argument("--manifests-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wav_files = sorted(args.wav_dir.glob("*.wav"))

    rng = random.Random(args.seed)
    rng.shuffle(wav_files)

    valid_count = max(1, int(len(wav_files) * args.valid_ratio))
    valid_files = wav_files[:valid_count]
    train_files = wav_files[valid_count:]

    train_meta = [wav_meta(p) for p in train_files]
    valid_meta = [wav_meta(p) for p in valid_files]

    train_manifest = args.manifests_dir / "train.jsonl.gz"
    valid_manifest = args.manifests_dir / "valid.jsonl.gz"
    write_jsonl_gz(train_manifest, train_meta)
    write_jsonl_gz(valid_manifest, valid_meta)

    print(f"Train manifest: {train_manifest} ({len(train_meta)} files)")
    print(f"Valid manifest: {valid_manifest} ({len(valid_meta)} files)")


if __name__ == "__main__":
    main()
