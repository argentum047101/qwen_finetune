# qwen_finetune
The task is to finetune qwen2.5-VL 7B to detect specific artefacts on the image
structure: 
{
  “watermarks”: количество вотермарок на изображении,
  “text”: текст на изображении,
  “main object”: основной объект на изображении,
  “style”: визуальный стиль изображения
}

dataset: https://huggingface.co/datasets/huggan/wikiart/viewer

Dataset containing 81,444 pieces of visual art from various artists, taken from WikiArt.org, along with class labels for each image :
"artist" : 129 artist classes, including a "Unknown Artist" class
"genre" : 11 genre classes, including a "Unknown Genre" class
"style" : 27 style classes

После парсинга датасета было решено взять ~5к с равномерным распределением все присутствующих стилей
Далее было принято решение к уже существуюшему срезу данных применить различные варианты наложения водяных знаков (паттерны, расположение, количество, цвет)

Полученный датасет был разбит на train/val/test выборки в пропорциях 0.7, 0.2, 0.1
В качестве метода finetune был выбран LoRa


Пример изображения
<PIL.PngImagePlugin.PngImageFile image mode=RGB size=1833x1382><img width="1833" height="1382" alt="image" src="https://github.com/user-attachments/assets/9d607ac8-a0b0-454f-aa74-522126d47623" />


Параметры LoRa:
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0.3,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)
