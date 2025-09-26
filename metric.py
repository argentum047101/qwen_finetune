import json
import logging
import difflib
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import Levenshtein

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)



def normalized_edit_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized edit similarity using difflib.
    """
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute raw Levenshtein edit distance.
    """
    return Levenshtein.distance(s1, s2)


def evaluate_example(ground_truth: dict, prediction: dict) -> dict:
    """
    Compare a single prediction to ground truth and compute evaluation metrics.
    """
    report = {}
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Watermarks
    true_wm = ground_truth["watermarks"]
    pred_wm = prediction["watermarks"]
    mae = abs(true_wm - pred_wm)
    wm_acc = int(true_wm == pred_wm)
    report["watermarks"] = {
        "true": true_wm,
        "pred": pred_wm,
        "mae": mae,
        "accuracy": wm_acc
    }

    # Text
    true_text = ground_truth["text"]
    pred_text = prediction["text"]
    sim = normalized_edit_similarity(true_text, pred_text)
    lev_dist = levenshtein_distance(true_text, pred_text)
    report["text"] = {
        "true": true_text,
        "pred": pred_text,
        "normalized_similarity": sim,
        "levenshtein_distance": lev_dist
    }

    # Main object
    true_obj = ground_truth["main object"]
    pred_obj = prediction["main_object"]
    emb_true = model.encode([true_obj])
    emb_pred = model.encode([pred_obj])
    cos_sim = cosine_similarity(emb_true, emb_pred)[0][0]
    obj_acc = int(true_obj.lower() == pred_obj.lower())
    report["main_object"] = {
        "true": true_obj,
        "pred": pred_obj,
        "cosine_similarity": float(cos_sim),
        "accuracy": obj_acc
    }

    # Style
    true_style = ground_truth["style"]
    pred_style = prediction["style"]
    style_acc = int(true_style.lower() == pred_style.lower())
    report["style"] = {
        "true": true_style,
        "pred": pred_style,
        "accuracy": style_acc
    }

    return report


def evaluate_dataset(ground_truths: list, predictions: list) -> dict:
    """
    Evaluate predictions for an entire dataset and compute aggregate metrics.
    """
    per_example = []
    total_mae, text_lev, obj_sim, style_acc = [], [], [], []

    for gt, pred in zip(ground_truths, predictions):
        report = evaluate_example(gt, pred)
        per_example.append(report)

        total_mae.append(report["watermarks"]["mae"])
        text_lev.append(report["text"]["levenshtein_distance"])
        obj_sim.append(report["main_object"]["cosine_similarity"])
        style_acc.append(report["style"]["accuracy"])

    summary = {
        "watermarks_MAE": float(np.mean(total_mae)) if total_mae else None,
        "text_levenshtein_distance": float(np.mean(text_lev)) if text_lev else None,
        "main_object_cosine_similarity": float(np.mean(obj_sim)) if obj_sim else None,
        "style_accuracy": float(np.mean(style_acc)) if style_acc else None
    }

    return {"per_example": per_example, "summary": summary}


def load_ground_truths(json_path: Path) -> list:
    """
    Load ground truth dataset from JSON and extract assistant annotations.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    ground_truths = []
    for ex in test_data:
        image_path = ex["messages"][0]["content"][0]["image"]
        assistant_text = ex["messages"][1]["content"][0]["text"]
        gt = json.loads(assistant_text)

        required_fields = ["watermarks", "text", "main object", "style"]
        missing_fields = [fld for fld in required_fields if fld not in gt]
        if missing_fields:
            raise ValueError(f"Missing fields {missing_fields} in {image_path}")

        gt["image"] = image_path
        ground_truths.append(gt)
    return ground_truths


def main():
    ground_truths = load_ground_truths(Path("qwen_dataset/test.json"))
    with open("lora_test_output.json", "r", encoding="utf-8") as f:
        predictions = json.load(f)

    gt_dict = {g["image"]: g for g in ground_truths}
    pred_sorted, gt_sorted = [], []
    for pred in predictions:
        if pred["image"] in gt_dict:
            pred_sorted.append(pred)
            gt_sorted.append(gt_dict[pred["image"]])

    results = evaluate_dataset(gt_sorted, pred_sorted)

    with open("lora_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info("Evaluation complete. Metrics saved to lora_test_results.json")


if __name__ == "__main__":
    main()