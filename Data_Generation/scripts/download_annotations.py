"""
Descarga las anotaciones COCO val2017 necesarias para el experimento.

Solo extrae los dos archivos requeridos del zip (~241 MB):
  - instances_val2017.json
  - person_keypoints_val2017.json

Uso:
    python download_annotations.py [--output-dir DIR]
"""

import argparse
import sys
import urllib.request
import zipfile
import io
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
NEEDED = {"annotations/instances_val2017.json", "annotations/person_keypoints_val2017.json"}


def _progress_hook(label: str):
    last = [0]
    def hook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb = downloaded / 1_048_576
            total_mb = total_size / 1_048_576
            bar = "#" * int(pct / 2)
            sys.stdout.write(f"\r{label}: [{bar:<50}] {pct:5.1f}%  {mb:.0f}/{total_mb:.0f} MB")
            sys.stdout.flush()
            if downloaded >= total_size:
                print()
    return hook


def download_annotations(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    already = {f"annotations/{p.name}" for p in output_dir.iterdir() if p.suffix == ".json"}
    missing = NEEDED - already
    if not missing:
        print("Anotaciones ya presentes:")
        for f in sorted(NEEDED):
            print(f"  {output_dir / Path(f).name}")
        return

    print(f"Descargando anotaciones COCO val2017 ({ANNOTATIONS_URL})")
    print("Tamaño aproximado: 241 MB\n")

    tmp_path, _ = urllib.request.urlretrieve(
        ANNOTATIONS_URL,
        reporthook=_progress_hook("Descargando"),
    )

    print("Extrayendo archivos necesarios...")
    with zipfile.ZipFile(tmp_path) as zf:
        for member in NEEDED:
            dest = output_dir / Path(member).name
            if dest.exists():
                print(f"  ya existe, omitiendo: {dest.name}")
                continue
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  extraído: {dest.name}  ({size_mb:.1f} MB)")

    Path(tmp_path).unlink(missing_ok=True)

    print(f"\nListo. Archivos en {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="Archive/annotations",
        help="Directorio destino (default: Archive/annotations)",
    )
    args = parser.parse_args()
    download_annotations(Path(args.output_dir))


if __name__ == "__main__":
    main()
