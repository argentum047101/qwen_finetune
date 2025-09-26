import os
import json
from pathlib import Path
from datasets import load_dataset

# Загружаем WikiArt
dataset = load_dataset("huggan/wikiart", cache_dir="cache")

# Папка куда сохраняем
root_dir = Path("wikiart_images")
root_dir.mkdir(parents=True, exist_ok=True)

# JSONL для аннотаций
out_json = "wikiart.jsonl"

# Сколько картинок в одной подпапке
chunk_size = 10_000

with open(out_json, "w", encoding="utf-8") as f:
    for i, sample in enumerate(dataset["train"]):
        img = sample["image"]  # PIL.Image

        # Определяем подпапку
        folder_idx = i // chunk_size
        folder_path = root_dir / f"part_{folder_idx:02d}"
        folder_path.mkdir(parents=True, exist_ok=True)

        # Путь для картинки
        out_path = folder_path / f"wikiart_{i:05d}.jpg"
        img.save(out_path, format="JPEG", quality=95)

        # Формируем JSON-запись
        entry = {
            "image_path": str(out_path),
            "instruction": "Analyze this image and provide the following information in JSON format: watermarks count, text in the image, main object, and visual style.",
            "output": {
                "watermarks": 0,
                "text": "",
                "main object": "", 
                "style": ""
            },
            # Дополнительно можно сохранить оригинальную мету WikiArt
            "meta": {
                "artist": sample.get("artist", None),
                "genre": sample.get("genre", None),
                "style_raw": sample.get("style", None)
            }
        }

        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if i % 1000 == 0:
            print(f"Saved {i} images...")