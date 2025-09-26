import json
import random
from pathlib import Path

def convert_and_split_dataset(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Convert JSONL to Qwen format and split into train/val/test sets
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save the split datasets
        train_ratio: Proportion of data for training (default: 0.8)
        val_ratio: Proportion of data for validation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.1)
    """
    
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read and convert all data
    converted_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Create the expected output JSON
            output_json = {
                "watermarks": data["watermarks"],
                "text": data["text"],
                "main object": data["main object"],
                "style": data["style"]
            }
            
            # Create the conversation
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
                                "text": "Analyze this image and provide the following information in JSON format: watermarks count, text in the image, main object, and visual style."
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
    
    # Shuffle the data
    random.shuffle(converted_data)
    
    # Calculate split indices
    total_samples = len(converted_data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    # Split the data
    train_data = converted_data[:train_size]
    val_data = converted_data[train_size:train_size + val_size]
    test_data = converted_data[train_size + val_size:]
    
    # Save the splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = output_path / f'{split_name}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(split_data)} samples to {output_file}")
    
    # Print statistics
    print(f"\nDataset split complete:")
    print(f"Total samples: {total_samples}")
    print(f"Train: {len(train_data)} ({len(train_data)/total_samples*100:.1f}%)")
    print(f"Val: {len(val_data)} ({len(val_data)/total_samples*100:.1f}%)")
    print(f"Test: {len(test_data)} ({len(test_data)/total_samples*100:.1f}%)")


if __name__ == "__main__":
    # Usage
    convert_and_split_dataset(
        input_file='vlm_finetune_data_new.jsonl',
        output_dir='qwen_dataset',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )