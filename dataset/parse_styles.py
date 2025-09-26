import os, json

STYLE_MAPPING = {
    0: "Abstract_Expressionism",
    1: "Action_painting",
    2: "Analytical_Cubism",
    3: "Art_Nouveau",
    4: "Baroque",
    5: "Color_Field_Painting",
    6: "Contemporary_Realism",
    7: "Cubism",
    8: "Early_Renaissance",
    9: "Expressionism",
    10: "Fauvism",
    11: "High_Renaissance",
    12: "Impressionism",
    13: "Mannerism_Late_Renaissance",
    14: "Minimalism",
    15: "Naive_Art_Primitivism",
    16: "New_Realism",
    17: "Northern_Renaissance",
    18: "Pointillism",
    19: "Pop_Art",
    20: "Post_Impressionism",
    21: "Realism",
    22: "Rococo",
    23: "Romanticism",
    24: "Symbolism",
    25: "Synthetic_Cubism",
    26: "Ukiyo_e"
}

# Paths
json_folder = "watermark_new"             # folder with watermark+style JSONs
annotations_file = "wikiart_5k_tagged.jsonl"  # JSONL you already have
output_file = "wikiart_5k_tagged_parsed.jsonl"

# --- Step 1: build metadata dictionary from folder ---
metadata_dict = {}
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        fpath = os.path.join(json_folder, filename)
        with open(fpath, "r") as f:
            data = json.load(f)

        style_idx = data.get("style")
        style_name = STYLE_MAPPING.get(style_idx, "Unknown_Style")

        # collect watermark texts
        watermark_texts = [wm["final_text"] for wm in data["watermark"]["watermarks"]]
        num_watermarks = len(watermark_texts)

        watermark_str = "; ".join(watermark_texts)
        img_path = f"watermark_new/{data['source_image']}"
        rec = {
            "image_path": img_path,
            "watermark": num_watermarks,
            "text": watermark_str,
            "style": style_name
        }
        # key by basename of source image
        key = os.path.basename(data["source_image"])
        metadata_dict[key] = rec

# --- Step 2: read annotations JSONL and merge ---
with open(output_file, "w", encoding="utf-8") as out_f:
    with open(annotations_file, "r", encoding="utf-8") as f:
        for line in f:
            ann = json.loads(line)
            image_name = os.path.basename(ann["image_path"])

            if image_name in metadata_dict:
                merged = metadata_dict[image_name].copy()
                merged["main_object"] = ann["output"]["main object"]

                style_idx = ann["output"]["style"]
                style_name = STYLE_MAPPING.get(style_idx, "Unknown_Style")
                merged["style"] = style_name
                # optional extras

                out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")