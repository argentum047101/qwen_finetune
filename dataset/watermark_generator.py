import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import glob

class SimpleTextWatermark:
    def __init__(self):
        # Simple list of watermark texts
        self.texts = [
            "CONFIDENTIAL", "DRAFT", "COPY", "SAMPLE", "VOID", 
            "APPROVED", "PENDING", "ORIGINAL", "DUPLICATE", "EXPIRED"
        ]
    
    def add_watermark(self, image_path, output_folder):
        """Add a simple text watermark to an image"""
        try:
            # Open image
            img = Image.open(image_path).convert('RGBA')
            width, height = img.size
            
            # Create transparent overlay
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Choose random text and add a number
            text = random.choice(self.texts)
            number = random.randint(1, 5)
            watermark_text = f"{text} {number}"
            
            # Simple font (default)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 50)
            except:
                font = ImageFont.load_default()
            
            # Get text size
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Random position (but keep it inside the image)
            x = random.randint(50, max(51, width - text_width - 50))
            y = random.randint(50, max(51, height - text_height - 50))
            
            # Random color (light colors)
            colors = [(255, 255, 255), (200, 200, 200), (150, 150, 255)]
            color = random.choice(colors)
            
            # Draw text with transparency
            draw.text((x, y), watermark_text, font=font, fill=(*color, 100))
            
            # Combine with original image
            watermarked = Image.alpha_composite(img, overlay)
            watermarked = watermarked.convert('RGB')
            
            # Save
            filename = os.path.basename(image_path).split('.')[0]
            output_path = os.path.join(output_folder, f"{filename}.png")
            watermarked.save(output_path)
            
            # Save info as JSON
            info = {
                "original": os.path.basename(image_path),
                "watermark_text": watermark_text,
                "position": [x, y],
                "color": color,
                "timestamp": datetime.now().isoformat()
            }
            
            json_path = os.path.join(output_folder, f"{filename}.json")
            with open(json_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error with {image_path}: {e}")
            return False

def main():
    # Settings
    INPUT_FOLDER = "wikiart_5k"
    OUTPUT_FOLDER = "watermark_new"
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all images
    image_types = ['*.jpg', '*.jpeg', '*.png']
    all_images = []
    
    for img_type in image_types:
        all_images.extend(glob.glob(os.path.join(INPUT_FOLDER, img_type)))
    
    if not all_images:
        print(f"No images found in {INPUT_FOLDER}")
        return
    
    print(f"Found {len(all_images)} images")
    
    # Process images
    watermarker = SimpleTextWatermark()
    success = 0
    
    for i, image_path in enumerate(all_images):
        print(f"Processing {i+1}/{len(all_images)}: {os.path.basename(image_path)}")
        
        if watermarker.add_watermark(image_path, OUTPUT_FOLDER):
            success += 1
    
    print(f"\nDone! Processed {success}/{len(all_images)} images")
    print(f"Output saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()