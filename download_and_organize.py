"""
Descarga las imágenes COCO val2017 necesarias para el experimento
y las organiza en carpetas indoor/outdoor.

Uso:
    python download_and_organize.py [--output-dir DIR] [--workers N]

Solo descarga las imágenes listadas en images_indoor.json e images_outdoor.json.
"""

import json
import os
import sys
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COCO_BASE_URL = "http://images.cocodataset.org/val2017/"


def load_image_list(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def download_image(file_name: str, dest_path: Path) -> tuple[str, bool, str]:
    if dest_path.exists():
        return file_name, True, "cached"
    url = COCO_BASE_URL + file_name
    try:
        urllib.request.urlretrieve(url, dest_path)
        return file_name, True, "downloaded"
    except urllib.error.URLError as e:
        return file_name, False, str(e)


def download_and_organize(
    indoor_json: str,
    outdoor_json: str,
    output_dir: Path,
    workers: int,
):
    indoor_images = load_image_list(indoor_json)
    outdoor_images = load_image_list(outdoor_json)

    indoor_dir = output_dir / "indoor"
    outdoor_dir = output_dir / "outdoor"
    indoor_dir.mkdir(parents=True, exist_ok=True)
    outdoor_dir.mkdir(parents=True, exist_ok=True)

    # Build (file_name, dest_path) pairs — deduplicate across sets
    tasks: list[tuple[str, Path, str]] = []
    seen: set[str] = set()

    for img in indoor_images:
        fn = img["file_name"]
        if fn not in seen:
            tasks.append((fn, indoor_dir / fn, "indoor"))
            seen.add(fn)

    for img in outdoor_images:
        fn = img["file_name"]
        if fn not in seen:
            tasks.append((fn, outdoor_dir / fn, "outdoor"))
            seen.add(fn)
        else:
            # Image appears in both sets → symlink or copy to outdoor too
            src = indoor_dir / fn
            dst = outdoor_dir / fn
            if not dst.exists() and src.exists():
                import shutil
                shutil.copy2(src, dst)

    total = len(tasks)
    print(f"Imágenes a descargar: {total}  ({len(indoor_images)} indoor, {len(outdoor_images)} outdoor)")
    print(f"Destino: {output_dir}")
    print(f"Workers: {workers}\n")

    success = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_image, fn, dest): (fn, dest, category)
            for fn, dest, category in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            fn, ok, msg = future.result()
            if ok:
                success += 1
                status = "OK" if msg == "downloaded" else "cached"
            else:
                fail += 1
                status = f"ERROR: {msg}"
                print(f"  [FAIL] {fn}: {status}")

            if i % 100 == 0 or i == total:
                print(f"  Progreso: {i}/{total}  ok={success}  fail={fail}")

    print(f"\nListo. Descargadas: {success}  Errores: {fail}")
    print(f"  indoor/  → {indoor_dir}")
    print(f"  outdoor/ → {outdoor_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indoor-json", default="images_indoor.json")
    parser.add_argument("--outdoor-json", default="images_outdoor.json")
    parser.add_argument("--output-dir", default="Archive/images",
                        help="Directorio raíz donde se crean indoor/ y outdoor/")
    parser.add_argument("--workers", type=int, default=16,
                        help="Hilos de descarga paralela")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_and_organize(
        args.indoor_json,
        args.outdoor_json,
        output_dir,
        args.workers,
    )


if __name__ == "__main__":
    main()
