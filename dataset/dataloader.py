import random
import json
import logging
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_wikiart(cache_dir: str = "cache"):
    """
    Load the WikiArt dataset using Hugging Face datasets library.
    """
    return load_dataset("huggan/wikiart", cache_dir=cache_dir)


def select_balanced_subset(dataset, target_total: int = 5000):
    """
    Select a balanced subset of the WikiArt dataset across styles.
    Returns dataset indices.
    """
    style_to_indices = defaultdict(list)
    for i, sample in enumerate(dataset["train"]):
        style = sample["style"] if sample["style"] else "unknown"
        style_to_indices[style].append(i)

    num_styles = len(style_to_indices)
    per_style = target_total // num_styles
    logging.info("Found %d styles, selecting ~%d images per style", num_styles, per_style)

    selected_indices = []
    for style, indices in style_to_indices.items():
        if len(indices) <= per_style:
            chosen = indices
        else:
            chosen = random.sample(indices, per_style)
        selected_indices.extend(chosen)

    selected_indices = selected_indices[:target_total]
    logging.info("Selected %d images in total", len(selected_indices))
    return selected_indices


def save_subset(dataset, selected_indices, out_dir: str = "wikiart_5k", out_json: str = "wikiart_5k.jsonl"):
    """
    Save selected subset of WikiArt images to disk and write JSONL annotations.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        for new_id, idx in enumerate(selected_indices):
            sample = dataset["train"][idx]
            img = sample["image"]

            img_path = out_path / f"wikiart_{new_id:05d}.jpg"
            img.save(img_path, format="JPEG", quality=95)

            entry = {
                "image_path": str(img_path),
                "instruction": (
                    "Analyze this image and return JSON with fields: "
                    "watermarks, text, main object, style."
                ),
                "output": {
                    "watermarks": 0,
                    "text": "",
                    "main object": "painting",
                    "style": sample["style"] if sample["style"] else "unknown"
                },
                "meta": {
                    "artist": sample.get("artist", None),
                    "genre": sample.get("genre", None),
                    "style_raw": sample.get("style", None)
                }
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if new_id % 500 == 0 and new_id > 0:
                logging.info("Saved %d / %d images", new_id, len(selected_indices))

    logging.info("Finished: %d images saved to %s, metadata in %s",
                 len(selected_indices), out_path, out_json)


def main():
    dataset = load_wikiart()
    selected_indices = select_balanced_subset(dataset, target_total=5000)
    save_subset(dataset, selected_indices)


if __name__ == "__main__":
    main()