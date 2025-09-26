### Project Overview

This project fine‑tunes the **Qwen2.5‑VL 7B** model to detect and classify specific artifacts in images.  
The model produces a structured JSON‑like output with the following fields:

```json
{
  "watermarks": "number of watermarks in the image",
  "text": "text present in the image",
  "main object": "primary object in the image",
  "style": "visual style of the image"
}
```

### Dataset Source

**WikiArt Dataset**  
The dataset contains **81,444 pieces of visual art** from [WikiArt.org](https://www.wikiart.org/) with the following class labels:

- **Artists:** 129 classes (including *Unknown Artist*)  
- **Genres:** 11 classes (including *Unknown Genre*)  
- **Styles:** 27 classes  

---

### Data Preparation
  
- Selected ~5,000 images with uniform distribution across all style classes
- Predicted main object with Kimi-VL model 
- Applied various watermark transformations:
  - Different patterns  
  - Different positions  
  - Varying quantities  
  - Multiple colors  
  - Rotated at various angles
  - Different watermark words
- Split dataset into:
  - **70%** training  
  - **20%** validation  
  - **10%** testing

 ### Generated image example

<img width="300" height="800" alt="image" src="https://github.com/user-attachments/assets/2ef73361-e515-41f7-b3fa-5f5b7c262485" />


```json
{
    "watermarks": 3, 
    "text": "PROOF 5; ESTIMATE 2; ANNUAL 2",
    "main object": "Girl",
    "style": "Realism"
}
```
---
### How to run

install requirements.txt

Inference
```
# Single image
python inference.py --image photo.jpg  --output results.json

# Folder of images
python inference.py --folder ./images/  --output results.json
```

---
Fine-tuning Method
LoRA Configuration:

```
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0.3,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

---

### Evaluation Metrics

| **Task**       | **Metric**               | **Description**                                 |
|----------------|--------------------------|-------------------------------------------------|
| Watermarks     | MAE (Mean Absolute Error) | Measures count error of detected watermarks     |
| Text           | Levenshtein Distance      | Difference between predict/ground       
| Main Object    | Cosine Similarity         | Semantic similarity of object descriptions      |
| Style          | Accuracy                  | Percentage of correctly predicted styles         |

---

### Results Comparison

| **Metric**                   | **Base Model** | **LoRA Fine-tuned** | **Improvement** |
|-------------------------------|----------------|----------------------|-----------------|
| Watermarks (MAE ↓)            | 0.1965         | 0.1148               | **41.6%**       |
| Text Levenshtein Distance (↓) | 10.786         | 9.196                | **14.7%**       |
| Main Object Similarity (↑)    | 0.4403         | 0.6377               | **44.8%**       |
| Style Accuracy (↑)            | 0.3687         | 0.4967               | **34.7%**       |

> **Legend:** ↓ Lower is better | ↑ Higher is better

---

### Key Improvements

- **Watermark detection:** 41.6% reduction in error  
- **Object recognition:** 44.8% improvement in similarity  
- **Style classification:** 34.7% increase in accuracy  
- **Text recognition:** 14.7% reduction in edit distance  

*LoRA fine‑tuning demonstrates significant improvements across all metrics, with especially strong performance in watermark detection and main object recognition.*

---

### Inference Example
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/659a4099-ca8b-4d89-b77a-1eeeab8b8166" />

```json
{
  "watermarks": 4,
  "text": "Photograph by Jacob Hoffman-Andrews. Licensed CC-BY. Make a copy!\nPhotograph by Jacob Hoffman-Andrews. Licensed CC-BY. Make a copy!\nPhotograph by Jacob Hoffman-Andrews. Licensed CC-BY. Make a copy!\nPhotograph by Jacob Hoffman-Andrews. Licensed CC-BY. Make a copy!"
  "main_object": "Flowers in a field",
  "style":" Contemporary_Realism"
}
```

---
 TODO:
 
1. Prompt engineering with base models
2. Train the LoRA model on the full image dataset
3. Experiment with varying watermark patterns to increase robustness 
4. Start with a larger model for zero-shot predictions
5. LoRA training with larger model
6. Add metric components for better evaluation


