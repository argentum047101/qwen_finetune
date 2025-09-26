import os
import json
import glob
import random
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class SimpleTextWatermark:
    """
    Utility for adding simple random text watermarks to images.
    Produces both watermarked PNG images and corresponding JSON metadata.
    """
    def __init__(self):
        self.texts = [
            "CONFIDENTIAL", "DRAFT", "COPY", "SAMPLE", "VOID",
            "APPROVED", "PENDING", "ORIGINAL", "DUPLICATE", "EXPIRED"
        ]

    def add_watermark(self, image_path: str, output_folder: str) -> bool:
        """
        Add a semi-transparent text watermark at a random position inside the image.
        Save both modified image (PNG) and metadata JSON.

        Args:
            image_path: Path to input image
            output_folder: Folder to save output PNG and JSON
        """
        try:
            img = Image.open(image_path).convert('RGBA')
            width, height = img.size

            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            text = random.choice(self.texts)
            number = random.randint(1, 5)
            watermark_text = f"{text} {number}"

            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 50
                )
            except Exception:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x = random.randint(50, max(51, width - text_width - 50))
            y = random.randint(50, max(51, height - text_height - 50))

            colors = [(255, 255, 255), (200, 200, 200), (150, 150, 255)]
            color = random.choice(colors)

            draw.text((x, y), watermark_text, font=font, fill=(*color, 100))

            watermarked = Image.alpha_composite(img, overlay).convert('RGB')

            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_folder, f"{filename}.png")
            watermarked.save(output_path)

            info = {
                "original": os.path.basename(image_path),
                "watermark_text": watermark_text,
                "position": [x, y],
                "color": color,
                "timestamp": datetime.now().isoformat()
            }

            json_path = os.path.join(output_folder, f"{filename}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)

            return True

        except Exception as e:
            logging.error("Processing failed for %s: %s", image_path, e)
            return False

def collect_images(input_folder: str) -> list:
    """
    Find all images (JPG/PNG) in the input folder.
    """
    image_types = ['*.jpg', '*.jpeg', '*.png']
    all_images = []
    for img_type in image_types:
        all_images.extend(glob.glob(os.path.join(input_folder, img_type)))
    return all_images

def process_images(input_folder: str, output_folder: str):
    """
    Iterate over images, apply watermarks, and save results.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    all_images = collect_images(input_folder)
    if not all_images:
        logging.warning("No images found in %s", input_folder)
        return

    logging.info("Found %d images", len(all_images))

    watermarker = SimpleTextWatermark()
    success = 0

    for i, image_path in enumerate(all_images, 1):
        logging.info("Processing %d/%d: %s", i, len(all_images), os.path.basename(image_path))
        if watermarker.add_watermark(image_path, output_folder):
            success += 1

    logging.info("Completed. Successfully processed %d/%d images. Output saved to: %s",
                 success, len(all_images), output_folder)

def main():
    """
    Main entry point for watermark generation.
    """
    input_folder = "wikiart_5k"
    output_folder = "watermark_new"
    process_images(input_folder, output_folder)

if __name__ == "__main__":
    main()