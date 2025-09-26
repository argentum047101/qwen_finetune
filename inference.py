import json
import re
import argparse
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel, Field
from unsloth import FastVisionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class Config:
    """Central configuration for the image analyzer"""
    MODEL_NAME = "lora_model"
    LOAD_IN_4BIT = True
    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.1

    ART_STYLES = [
        "Abstract_Expressionism", "Action_painting", "Analytical_Cubism",
        "Art_Nouveau", "Baroque", "Color_Field_Painting", "Contemporary_Realism",
        "Cubism", "Early_Renaissance", "Expressionism", "Fauvism",
        "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance",
        "Minimalism", "Naive_Art_Primitivism", "New_Realism",
        "Northern_Renaissance", "Pointillism", "Pop_Art", "Post_Impressionism",
        "Realism", "Rococo", "Romanticism", "Symbolism", "Synthetic_Cubism", "Ukiyo_e"
    ]


class ImageAnalysis(BaseModel):
    """Schema for image analysis results"""
    watermarks: int = Field(ge=0, description="Number of watermarks detected")
    text: str = Field(description="Text detected in the image")
    main_object: str = Field(description="Primary subject of the image")
    style: str = Field(description="Art style of the image")
    image_path: Optional[str] = Field(None, description="Path to the analyzed image")


class ImageAnalyzer:
    """Main class for analyzing images using Vision-Language Model"""

    def __init__(self, model_name: str = Config.MODEL_NAME, load_in_4bit: bool = Config.LOAD_IN_4BIT):
        """
        Initialize the VLM model and tokenizer.
        """
        logging.info("Loading model: %s", model_name)
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=load_in_4bit
        )
        FastVisionModel.for_inference(self.model)
        logging.info("Model loaded successfully")

        self.instruction = (
            "Analyze this image and provide the following information in JSON format: "
            "watermarks count, text in the image, main object, and visual style."
        )

    def _extract_json_from_output(self, raw_output: Union[str, List]) -> Dict:
        """
        Extract valid JSON object from model output text.
        """
        if isinstance(raw_output, list):
            candidates = raw_output
        else:
            if isinstance(raw_output, str):
                raw_output = raw_output.strip().replace('\\"', '"')
            candidates = re.findall(r"\{[\s\S]*?\}", raw_output)

        if not candidates:
            raise ValueError("No JSON object found in output")

        for candidate in reversed(candidates):
            try:
                decoded = candidate.encode().decode("unicode_escape")
                parsed = json.loads(decoded)

                if "main object" in parsed:
                    parsed["main_object"] = parsed.pop("main object")

                validated = ImageAnalysis.model_validate(parsed)
                return validated.model_dump()

            except Exception:
                continue

        raise RuntimeError("No valid JSON found in model output")

    def analyze_image(self, image_path: Union[str, Path]) -> ImageAnalysis:
        """
        Analyze a single image and return an ImageAnalysis object.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.instruction}
            ]
        }]

        input_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                use_cache=True
            )

        raw_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        result = self._extract_json_from_output(raw_output)
        result["image_path"] = str(image_path)

        return ImageAnalysis(**result)

    def analyze_folder(self, folder_path: Union[str, Path], extensions: List[str] = None) -> List[ImageAnalysis]:
        """
        Analyze all images inside a folder.
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

        image_files = []
        for ext in extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))

        if not image_files:
            logging.warning("No images found in %s", folder_path)
            return []

        results = []
        for image_path in tqdm(image_files, desc="Analyzing images"):
            try:
                result = self.analyze_image(image_path)
                results.append(result)
                logging.info("Analyzed %s: main_object=%s style=%s",
                             image_path.name, result.main_object, result.style)
            except Exception as e:
                logging.error("Error processing %s: %s", image_path.name, e)

        return results


def save_results(results: List[ImageAnalysis], output_path: Union[str, Path]):
    """
    Save analysis results into a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [result.model_dump() for result in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logging.info("Results saved to %s", output_path)


def log_results(results: List[ImageAnalysis]):
    """
    Log results summary for human inspection.
    """
    logging.info("===== ANALYSIS RESULTS =====")
    for result in results:
        logging.info(
            "Image=%s | Main=%s | Style=%s | Watermarks=%s | Text=%s",
            Path(result.image_path).name,
            result.main_object,
            result.style,
            result.watermarks,
            result.text or "-"
        )

def main():
    parser = argparse.ArgumentParser(
        description="Analyze images using Vision-Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python image_analyzer.py --folder path/to/images --output results.json"
    )

    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--model", type=str, default=Config.MODEL_NAME, help="Model name to use")

    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Please specify either --image or --folder")

    analyzer = ImageAnalyzer(model_name=args.model)

    if args.image:
        logging.info("Analyzing single image: %s", args.image)
        results = [analyzer.analyze_image(args.image)]
    else:
        logging.info("Analyzing folder: %s", args.folder)
        results = analyzer.analyze_folder(args.folder)

    log_results(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()