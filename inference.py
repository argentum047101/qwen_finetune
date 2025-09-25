import json
from pathlib import Path
from PIL import Image
import torch
from pydantic import BaseModel, ValidationError
from unsloth import FastVisionModel
import re, json


# ---------------- Schema ----------------
class Validate(BaseModel):
    watermarks: int
    text: str
    main_object: str
    style: str


# ---------------- Instruction ----------------
# instruction = """
# You are an image analysis system. Analyze the provided image and return only a valid JSON object that exactly follows **this schema including the exact key names**:
# {
#   "watermarks": integer,
#   "text": string,
#   "main_object": string,
#   "style": string
# }

# Rules:
# - Output must be strictly valid JSON (no comments, no explanations, no text outside braces).
# - "watermarks" = integer (use 0 if none).
# - "text" = any detected text in the image (empty string if none).
# - "main_object" = the primary subject of the image in plain English.
# - "style" = choose exactly one from the following list:
# ["Abstract_Expressionism","Action_painting","Analytical_Cubism","Art_Nouveau","Baroque","Color_Field_Painting","Contemporary_Realism","Cubism","Early_Renaissance","Expressionism","Fauvism","High_Renaissance","Impressionism","Mannerism_Late_Renaissance","Minimalism","Naive_Art_Primitivism","New_Realism","Northern_Renaissance","Pointillism","Pop_Art","Post_Impressionism","Realism","Rococo","Romanticism","Symbolism","Synthetic_Cubism","Ukiyo_e"]

# Examples of correct outputs:
# {
#   "watermarks": 0,
#   "text": "",
#   "main_object": "Woman with a parasol",
#   "style": "Impressionism"
# }
# {
#   "watermarks": 1,
#   "text": "COPYRIGHT",
#   "main_object": "Landscape with mountains",
#   "style": "Post_Impressionism"
# }
# {
#   "watermarks": 2,
#   "text": "VOID 4",
#   "main_object": "City buildings",
#   "style": "Cubism"
# }
# """

instruction = """
Analyze this image and provide the following information in JSON format: watermarks count, text in the image, main object, and visual style.
"""

# # ---------------- Model Init ----------------
# model, tokenizer = FastVisionModel.from_pretrained(
#     "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
#     load_in_4bit=True, 
#     use_gradient_checkpointing="unsloth"
# )

# ---------------- LoRa Model Init ----------------
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model) # Enable for inference!

def extract_last_json(raw_output):
    """
    Handles:
      - raw string with escaped \n
      - list of JSON-like strings
      - wrong key names (repairs 'main object' -> 'main_object')
    Always returns the last valid structured JSON.
    """
    # Case 1: list -> iterate directly
    if isinstance(raw_output, list):
        candidates = raw_output
    else:
        # Case 2: raw string
        # unescape \n etc if it looks like JSON is inside quotes
        if isinstance(raw_output, str):
            raw_output = raw_output.strip()
            raw_output = raw_output.replace('\\"', '"')
        # now regex-out all JSON dicts
        candidates = re.findall(r"\{[\s\S]*?\}", raw_output)

    if not candidates:
        raise ValueError("❌ No JSON object found in output.")

    # Try parsing from last candidate backwards
    for cand in reversed(candidates):
        try:
            decoded = cand.encode().decode('unicode_escape') 
            parsed = json.loads(decoded)
            # Repair wrong key names
            if "main object" in parsed:
                parsed["main_object"] = parsed.pop("main object")
            # Validate
            validated = Validate.model_validate(parsed).model_dump()
            return validated
        except Exception:
            continue

    raise RuntimeError("❌ No valid structured JSON found at the end of output.")

# def extract_last_json(text: str) -> dict:
#     # Regex: grab everything between { ... }
#     matches = re.findall(r"\{[\s\S]*?\}", text)
#     if not matches:
#         raise ValueError("❌ No JSON object found in model output")

#     last_json = matches[-1]  # take the last one only
#     parsed = json.loads(last_json)

#     try:
#         validated = Validate.model_validate(parsed).model_dump()
#     except ValidationError as e:
#         raise RuntimeError(f"Schema validation failed: {e}")
#     return validated


# ---------------- Core Analysis ----------------
def analyze_one(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction}
        ]}
    ]

    # Create inputs
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            use_cache=True,
        )

    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)

    result = extract_last_json(raw_output)

    return result


# ---------------- Batch Runner ----------------
def run_batch(test_json: str, output_json: str):
    # Load test.json (list structure)
    examples = json.loads(Path(test_json).read_text())

    results = []
    for idx, conv in enumerate(examples):
        for msg in conv["messages"]:
            for part in msg["content"]:
                if part["type"] == "image":
                    image_path = part["image"]
                    print(f"🔍 Processing {image_path} ...")
                    prediction = analyze_one(image_path)
                    prediction["image"] = image_path  # keep source info
                    results.append(prediction)

    # Save results
    Path(output_json).write_text(json.dumps(results, indent=2))
    print(f"✅ Finished {len(results)} images → saved to {output_json}")


# ---------------- Run ----------------
if __name__ == "__main__":
    run_batch("qwen_dataset/test.json", "lora_test_output.json")