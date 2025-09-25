import json
import difflib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import codecs

# --- Модель для эмбеддингов текста ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

import difflib
import Levenshtein  # pip install python-Levenshtein

def normalized_edit_similarity(s1, s2):
    # Нормализованная похожесть (0...1)
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def levenshtein_distance(s1, s2):
    # Минимальное количество вставок/удалений/замен
    return Levenshtein.distance(s1, s2)

# --- Метрики ---
def normalized_edit_similarity(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()
def parse_assistant_response(assistant_text):
    """
    Robustly parse assistant's JSON response, handling various formatting issues.
    """
    if not assistant_text:
        raise ValueError("Empty assistant text")
    
    # Store original for error reporting
    original_text = assistant_text
    
    # Try different parsing strategies
    strategies = [
        # Strategy 1: Direct parsing (already valid JSON)
        lambda t: json.loads(t),
        
        # Strategy 2: Decode unicode escapes (\\n -> \n, \\" -> ")
        lambda t: json.loads(codecs.decode(t, 'unicode_escape')),
        
        # Strategy 3: Strip whitespace and decode
        lambda t: json.loads(codecs.decode(t.strip(), 'unicode_escape')),
        
        # Strategy 4: Handle double-encoded escapes
        lambda t: json.loads(t.encode().decode('unicode_escape')),
        
        # Strategy 5: Replace single quotes with double quotes
        lambda t: json.loads(t.replace("'", '"')),
        
        # Strategy 6: Clean up common issues
        lambda t: json.loads(
            t.strip()
            .replace("\\n", "\n")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        ),
    ]
    
    # Try each strategy
    for i, strategy in enumerate(strategies):
        try:
            return strategy(assistant_text)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # If this is the last strategy, raise the error
            if i == len(strategies) - 1:
                raise json.JSONDecodeError(
                    f"Failed to parse JSON after trying all strategies. Original text: {repr(original_text)}",
                    original_text, 0
                )
            # Otherwise, try the next strategy
            continue


def evaluate_example(ground_truth, prediction):
    report = {}
    
    # === watermarks (число) ===
    true_wm = ground_truth["watermarks"]
    pred_wm = prediction["watermarks"]
    mae = abs(true_wm - pred_wm)
    wm_acc = int(true_wm == pred_wm)
    report["watermarks"] = {
        "true": true_wm, "pred": pred_wm,
        "mae": mae, "accuracy": wm_acc
    }
    
    # === text (текст в картинке) ===
    true_text = ground_truth["text"]
    pred_text = prediction["text"]
    
    sim = normalized_edit_similarity(true_text, pred_text)
    lev_dist = levenshtein_distance(true_text, pred_text)
    
    report["text"] = {
        "true": true_text, "pred": pred_text,
        "normalized_similarity": sim,
        "levenshtein_distance": lev_dist
    }
    
    # === main object ===
    true_obj = ground_truth["main object"]
    pred_obj = prediction["main_object"]
    
    emb_true = model.encode([true_obj])
    emb_pred = model.encode([pred_obj])
    cos_sim = cosine_similarity(emb_true, emb_pred)[0][0]
    obj_acc = int(true_obj.lower() == pred_obj.lower())
    
    report["main_object"] = {
        "true": true_obj, "pred": pred_obj,
        "cosine_similarity": float(cos_sim),
        "accuracy": obj_acc
    }
    
    # === style ===
    true_style = ground_truth["style"]
    pred_style = prediction["style"]
    style_acc = int(true_style.lower() == pred_style.lower())
    
    report["style"] = {
        "true": true_style, "pred": pred_style,
        "accuracy": style_acc
    }
    
    return report
def evaluate_dataset(ground_truths, predictions):
    per_example = []
    total_mae, wm_acc, text_sim, text_lev, obj_sim, obj_acc, style_acc = [], [], [], [], [], [], []
    
    for gt, pred in zip(ground_truths, predictions):
        report = evaluate_example(gt, pred)
        per_example.append(report)
        
        total_mae.append(report["watermarks"]["mae"])
        wm_acc.append(report["watermarks"]["accuracy"])
        text_sim.append(report["text"]["normalized_similarity"])
        text_lev.append(report["text"]["levenshtein_distance"])
        obj_sim.append(report["main_object"]["cosine_similarity"])
        obj_acc.append(report["main_object"]["accuracy"])
        style_acc.append(report["style"]["accuracy"])
    
    summary = {
        "watermarks_MAE": np.mean(total_mae),
        "watermarks_accuracy": np.mean(wm_acc),
        "text_similarity": np.mean(text_sim),
        "text_levenshtein_distance": np.mean(text_lev),
        "main_object_cosine_similarity": np.mean(obj_sim),
        "main_object_accuracy": np.mean(obj_acc),
        "style_accuracy": np.mean(style_acc)
    }
    
    return {"per_example": per_example, "summary": summary}

# --- 1. Загружаем ground truth ---
with open("qwen_dataset/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

ground_truths = []
errors = []

for idx, ex in enumerate(test_data):
    try:
        # Extract image path
        image_path = ex["messages"][0]["content"][0]["image"]
        
        # Extract assistant's response
        assistant_text = ex["messages"][1]["content"][0]["text"]
        
        # Parse the JSON response
        gt = parse_assistant_response(assistant_text)
        
        # Validate expected fields
        required_fields = ["watermarks", "text", "main object", "style"]
        missing_fields = [field for field in required_fields if field not in gt]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Add image path
        gt["image"] = image_path
        ground_truths.append(gt)
        
    except (KeyError, IndexError) as e:
        error_msg = f"Error accessing data structure at index {idx}: {e}"
        print(error_msg)
        errors.append({"index": idx, "error": error_msg, "data": ex})
        
    except (json.JSONDecodeError, ValueError) as e:
        error_msg = f"Error parsing JSON at index {idx}: {e}"
        print(error_msg)
        if 'assistant_text' in locals():
            print(f"Problematic text: {repr(assistant_text)}")
        errors.append({"index": idx, "error": error_msg, "data": ex})
        
    except Exception as e:
        error_msg = f"Unexpected error at index {idx}: {type(e).__name__}: {e}"
        print(error_msg)
        errors.append({"index": idx, "error": error_msg, "data": ex})

# --- 2. Загружаем предсказания ---
with open("lora_test_output.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

# --- 3. Сопоставим по image ---
gt_dict = {g["image"]: g for g in ground_truths}
pred_sorted = []
gt_sorted = []
for pred in predictions:
    img = pred["image"]
    if img in gt_dict:
        pred_sorted.append(pred)
        gt_sorted.append(gt_dict[img])

# --- 4. Считаем метрики ---
results = evaluate_dataset(gt_sorted, pred_sorted)

# --- 5. Сохраним в файл ---
with open("lora_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Оценка завершена. Метрики сохранены в results.json")