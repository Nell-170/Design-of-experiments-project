import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", default="Archive/annotations/instances_val2017.json")
    parser.add_argument("--keypoints", default="Archive/annotations/person_keypoints_val2017.json")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--include-non-keypoints", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    keypoints = json.loads(Path(args.keypoints).read_text(encoding="utf-8"))

    name_to_id = {c["name"]: c["id"] for c in instances["categories"]}

    outdoor_names = ["traffic light", "stop sign", "fire hydrant", "parking meter", "bench"]
    indoor_names = ["microwave", "oven", "refrigerator", "sink", "toaster", "tv", "laptop", "keyboard", "mouse", "remote"]

    outdoor_ids = {name_to_id[n] for n in outdoor_names if n in name_to_id}
    indoor_ids = {name_to_id[n] for n in indoor_names if n in name_to_id}

    image_file_by_id = {img["id"]: img["file_name"] for img in instances["images"]}

    image_categories = defaultdict(set)
    for ann in instances["annotations"]:
        image_categories[ann["image_id"]].add(ann["category_id"])

    outdoor_images = {img_id for img_id, cats in image_categories.items() if cats & outdoor_ids}
    indoor_images = {img_id for img_id, cats in image_categories.items() if cats & indoor_ids}

    ambiguous_images = outdoor_images & indoor_images
    only_outdoor = outdoor_images - indoor_images
    only_indoor = indoor_images - outdoor_images

    if args.include_non_keypoints:
        base_ids = set(image_file_by_id)
    else:
        base_ids = {img["id"] for img in keypoints["images"]}

    only_outdoor = sorted(only_outdoor & base_ids)
    only_indoor = sorted(only_indoor & base_ids)
    ambiguous_images = sorted(ambiguous_images & base_ids)

    indoor_payload = [{"image_id": i, "file_name": image_file_by_id.get(i)} for i in only_indoor]
    outdoor_payload = [{"image_id": i, "file_name": image_file_by_id.get(i)} for i in only_outdoor]
    ambiguous_payload = [{"image_id": i, "file_name": image_file_by_id.get(i)} for i in ambiguous_images]

    (output_dir / "images_indoor.json").write_text(json.dumps(indoor_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "images_outdoor.json").write_text(json.dumps(outdoor_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "images_ambiguous.json").write_text(json.dumps(ambiguous_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "instances_path": args.instances,
        "keypoints_path": args.keypoints,
        "include_non_keypoints": args.include_non_keypoints,
        "outdoor_reference_categories": [n for n in outdoor_names if n in name_to_id],
        "indoor_reference_categories": [n for n in indoor_names if n in name_to_id],
        "counts": {
            "indoor": len(indoor_payload),
            "outdoor": len(outdoor_payload),
            "ambiguous": len(ambiguous_payload),
        },
        "outputs": {
            "indoor": str(output_dir / "images_indoor.json"),
            "outdoor": str(output_dir / "images_outdoor.json"),
            "ambiguous": str(output_dir / "images_ambiguous.json"),
        },
    }
    (output_dir / "blocks_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
