import gdown
from pathlib import Path
import sys


def main() -> None:
    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = gdown.download_folder(
        url="https://drive.google.com/drive/folders/10wjpFoKhsLegbCSnr2rlUEza2R_p6nuZ?usp=sharing",
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )

    print(f"Downloaded {len(files)} files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
