import json
import random
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_and_convert(input_file: str) -> list:
    """
    Load a JSONL dataset and convert each entry into Qwen-compatible conversation format.
    """
    converted_data = []
    with open(input_file, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            output_json = {
                "watermarks": data["watermarks"],
                "text": data["text"],
                "main object": data["main object"],
                "style": data["style"]
            }

            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": data["image_path"]
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this image and provide the following information in JSON format: "
                                    "watermarks count, text in the image, main object, and visual style."
                                )
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(output_json, ensure_ascii=False, indent=0).replace('\n', '\\n')
                            }
                        ]
                    }
                ]
            }

            converted_data.append(conversation)
    logging.info("Converted %d records into Qwen format", len(converted_data))
    return converted_data

def split_data(data: list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Shuffle and split the dataset into train/val/test according to given ratios.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    random.shuffle(data)

    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    logging.info("Dataset split complete: total=%d, train=%d, val=%d, test=%d",
                 total_samples, len(train_data), len(val_data), len(test_data))
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

def save_splits(splits: dict, output_dir: str):
    """
    Save split datasets into JSON files within the output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        output_file = output_path / f"{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        logging.info("Saved %d samples to %s", len(split_data), output_file)

def convert_and_split_dataset(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    End-to-end process: load + convert dataset, split into subsets, and save them.
    """
    data = load_and_convert(input_file)
    splits = split_data(data, train_ratio, val_ratio, test_ratio)
    save_splits(splits, output_dir)

def main():
    """
    Main entry point for dataset conversion and splitting.
    """
    convert_and_split_dataset(
        input_file='vlm_finetune_data_new.jsonl',
        output_dir='qwen_dataset',
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

if __name__ == "__main__":
    main()