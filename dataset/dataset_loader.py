import os
import json
import logging
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_wikiart_dataset(cache_dir: str = "cache"):
    """
    Load the WikiArt dataset using the Hugging Face datasets library.
    """
    return load_dataset("huggan/wikiart", cache_dir=cache_dir)

def prepare_output_directory(path: str = "wikiart_images") -> Path:
    """
    Create the root directory for storing images if it does not exist.
    """
    root_dir = Path(path)
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir

def save_images_and_annotations(dataset, root_dir: Path, out_json: str = "wikiart.jsonl", chunk_size: int = 10_000):
    """
    Save images from the dataset into structured folders and generate JSONL annotations.
    Each folder contains up to `chunk_size` images.
    """
    with open(out_json, "w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset["train"]):
            img = sample["image"]

            folder_idx = i // chunk_size
            folder_path = root_dir / f"part_{folder_idx:02d}"
            folder_path.mkdir(parents=True, exist_ok=True)

            out_path = folder_path / f"wikiart_{i:05d}.jpg"
            img.save(out_path, format="JPEG", quality=95)

            entry = {
                "image_path": str(out_path),
                "instruction": "Analyze this image and provide the following information in JSON format: watermarks count, text in the image, main object, and visual style.",
                "output": {
                    "watermarks": 0,
                    "text": "",
                    "main object": "",
                    "style": ""
                },
                "meta": {
                    "artist": sample.get("artist", None),
                    "genre": sample.get("genre", None),
                    "style_raw": sample.get("style", None)
                }
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if i % 1000 == 0:
                logging.info("Saved %d images...", i)

def main():
    """
    The workflow: load dataset, prepare directories, save images and annotations.
    """
    dataset = load_wikiart_dataset()
    root_dir = prepare_output_directory()
    save_images_and_annotations(dataset, root_dir)

if __name__ == "__main__":
    main()