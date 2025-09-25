import cv2
import numpy as np
import json
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
from datetime import datetime
import glob
from tqdm import tqdm
import math

class WatermarkGenerator:
    def __init__(self):
        # Watermark text options
        self.watermark_texts = [
            "CONFIDENTIAL", "DRAFT", "COPY", "ORIGINAL", "DUPLICATE", "SAMPLE", 
            "SPECIMEN", "VOID", "CANCELLED", "EXPIRED", "INVALID", "APPROVED", 
            "REJECTED", "PENDING", "CLASSIFIED", "RESTRICTED", "PRIVATE", "PUBLIC", 
            "OFFICIAL", "UNOFFICIAL", "CERTIFIED", "UNCERTIFIED", "AUTHENTICATED", 
            "VERIFIED", "PROPRIETARY", "COPYRIGHT", "TRADEMARK", "PATENT PENDING", 
            "TRADE SECRET", "INTERNAL USE ONLY", "DO NOT COPY", "DO NOT DISTRIBUTE", 
            "FOR REVIEW ONLY", "NOT FOR SALE", "PROOF", "FINAL", "PRELIMINARY", 
            "WORKING COPY", "MASTER COPY", "CONTROLLED DOCUMENT", "UNCONTROLLED", 
            "OBSOLETE", "SUPERSEDED", "PAID", "UNPAID", "OVERDUE", "RECEIVED", 
            "PROCESSED", "AUDITED", "RECONCILED", "BUDGET", "ESTIMATE", "INVOICE", 
            "STATEMENT", "QUOTE", "PROPOSAL", "URGENT", "PRIORITY", "RUSH", "HOLD", 
            "FILE COPY", "REFERENCE ONLY", "ARCHIVE", "DESTROY AFTER USE", 
            "RETAIN UNTIL", "EXPIRES ON", "EFFECTIVE DATE", "REVISION", "VERSION", 
            "AMENDMENT", "TRANSCRIPT", "DIPLOMA", "CERTIFICATE", "LICENSE", 
            "PRESCRIPTION", "MEDICAL RECORD", "TEST RESULTS", "LAB REPORT", 
            "STUDENT COPY", "INSTRUCTOR COPY", "EXAMINATION", "ANSWER KEY", 
            "WATERMARKED", "DIGITAL COPY", "ELECTRONIC VERSION", "SCANNED", 
            "PHOTOGRAPHED", "REPRODUCED", "ENHANCED", "EDITED", "UNEDITED", "RAW", 
            "PROCESSED", "COMPRESSED", "HIGH RESOLUTION", "LOW RESOLUTION", "PREVIEW", 
            "THUMBNAIL", "CURRENT", "OUTDATED", "HISTORICAL", "ARCHIVED", "TEMPORARY", 
            "PERMANENT", "LIMITED TIME", "SEASONAL", "ANNUAL", "QUARTERLY", "MONTHLY", 
            "DAILY"
        ]
        
        # Company/brand names for logo-style watermarks
        self.brand_names = [
            "ACME Corp", "TechVision", "DataSoft", "CloudNet", "InfoSys",
            "Digital Solutions", "Smart Systems", "Global Tech", "ProServices"
        ]
        
        # Auto-detect system fonts
        self.font_paths = self.find_system_fonts()
        
        if not self.font_paths:
            print("Warning: No system fonts found. Using default font.")
            self.font_paths = [None]
        else:
            print(f"Found {len(self.font_paths)} fonts for watermarks")
    
    def find_system_fonts(self):
        """Find all available fonts on the system"""
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            f"{os.path.expanduser('~')}/.fonts",
            f"{os.path.expanduser('~')}/.local/share/fonts",
        ]
        
        fonts = []
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                fonts.extend(glob.glob(os.path.join(font_dir, '**', '*.ttf'), recursive=True))
                fonts.extend(glob.glob(os.path.join(font_dir, '**', '*.otf'), recursive=True))
        
        valid_fonts = [f for f in fonts if os.access(f, os.R_OK)]
        return valid_fonts[:20]
    
    def get_font(self, size, bold=False):
        """Get a font with specified size"""
        if self.font_paths and self.font_paths[0]:
            try:
                # Try to find bold variant if requested
                if bold:
                    bold_fonts = [f for f in self.font_paths if 'bold' in f.lower()]
                    if bold_fonts:
                        return ImageFont.truetype(random.choice(bold_fonts), size)
                
                return ImageFont.truetype(random.choice(self.font_paths), size)
            except:
                return ImageFont.load_default()
        return ImageFont.load_default()
    

    def create_text_watermark(self, img_size, text=None):
        """
        Create multiple watermarks (1–5) with random texts, numbers, rotation, colors,
        without overlapping and keeping inside image boundaries.
        """
        width, height = img_size
        
        # Transparent layer for all watermarks
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        
        # How many watermarks to draw
        num_watermarks = random.randint(1, 5)
        
        # Store metadata and used bounding boxes
        watermark_info = []
        occupied_boxes = []
        
        # Color palette (can extend as needed)
        color_palette = [
            (255, 255, 255),  # white
            (200, 200, 200),  # light gray
            (150, 150, 255),  # light blue
            (255, 180, 180),  # pinkish-red
            (180, 255, 180),  # soft green
            (255, 220, 150),  # warm orange-yellow
        ]

        for i in range(num_watermarks):
            # Pick base text (user-defined or random from set)
            base_text = text if text is not None else random.choice(self.watermark_texts)
            rand_num = random.randint(1, 5)
            final_text = f"{base_text} {rand_num}"

            # Font
            font_size = min(width, height) // random.randint(12, 18)  # случайные размеры шрифта для разнообразия
            font = self.get_font(font_size, bold=True)

            # Text bbox (for size)
            tmp = Image.new("RGBA", img_size, (0, 0, 0, 0))
            draw_tmp = ImageDraw.Draw(tmp)
            bbox = draw_tmp.textbbox((0, 0), final_text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Add padding to prevent letter cutting
            pad_x = int(text_w * 0.3)
            pad_y = int(text_h * 0.3)

            canvas_w = text_w + pad_x * 2
            canvas_h = text_h + pad_y * 2
            text_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_img)

            # Random color + opacity
            opacity = random.randint(80, 120)
            base_color = random.choice(color_palette)
            color = (*base_color, opacity)

            # Draw inside padded area
            draw.text((pad_x, pad_y), final_text, font=font, fill=color)

            # Random rotation
            angle = random.randint(-45, 45)
            rotated_text = text_img.rotate(angle, expand=True)

            # Try to place inside boundaries and without overlap
            margin = min(width, height) // 20
            placed = False
            attempts = 0
            max_attempts = 50
            
            while not placed and attempts < max_attempts:
                attempts += 1
                x = random.randint(margin, max(margin, width - rotated_text.width - margin))
                y = random.randint(margin, max(margin, height - rotated_text.height - margin))

                # Bounding box of the rotated watermark
                box = (x, y, x + rotated_text.width, y + rotated_text.height)

                # Check image boundaries
                if box[2] > width or box[3] > height:
                    continue

                # Check overlap with others
                overlap = False
                for ox1, oy1, ox2, oy2 in occupied_boxes:
                    if not (box[2] <= ox1 or box[0] >= ox2 or box[3] <= oy1 or box[1] >= oy2):
                        overlap = True
                        break
                if overlap:
                    continue

                # Place finally!
                watermark.alpha_composite(rotated_text, (x, y))
                occupied_boxes.append(box)

                watermark_info.append({
                    "base_text": base_text,
                    "number": rand_num,
                    "final_text": final_text,
                    "color": base_color,
                    "opacity": opacity,
                    "rotation": angle,
                    "coordinates": (x, y),
                    "box": box
                })
                placed = True
        
        return watermark, {
            "type": "text",
            "count": len(watermark_info),
            "watermarks": watermark_info
        }

    def create_diagonal_watermark(self, img_size, text=None):
        """Create diagonal repeated text watermark"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Select text
        if text is None:
            text = random.choice(self.watermark_texts)
        
        # Repeat the text a random number of times
        repeat_count = random.randint(2, 5)  # Repeat 2-5 times
        final_text = ' '.join([text] * repeat_count)
        
        # Font size
        font_size = min(width, height) // 15
        font = self.get_font(font_size)
        
        # Calculate diagonal angle
        angle = math.degrees(math.atan(height / width))
        
        # Create pattern
        opacity = random.randint(60, 90)
        color = (200, 200, 200, opacity)
        
        # Spacing between text
        spacing_x = width // 4
        spacing_y = height // 4
        
        # Draw diagonal text pattern
        for i in range(-2, 5):
            for j in range(-2, 5):
                x = i * spacing_x
                y = j * spacing_y
                
                # Create rotated text
                txt_img = Image.new('RGBA', (len(final_text) * font_size, font_size * 2), (0, 0, 0, 0))
                txt_draw = ImageDraw.Draw(txt_img)
                txt_draw.text((0, 0), final_text, font=font, fill=color)
                
                # Rotate
                rotated = txt_img.rotate(angle, expand=1)
                
                # Paste onto watermark
                watermark.paste(rotated, (x, y), rotated)
        
        return watermark, {
            "type": "diagonal_pattern", 
            "text": text,
            "repeat_count": repeat_count,
            "final_text": final_text,
            "opacity": opacity, 
            "angle": angle
        }
    
    def create_corner_watermark(self, img_size, text=None, corner="bottom-right"):
        """Create corner watermark"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Select text
        if text is None:
            text = random.choice(self.brand_names + ["© " + str(datetime.now().year)])
        
        # Font size
        font_size = min(width, height) // 20
        font = self.get_font(font_size)
        
        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Padding
        padding = 20
        
        # Position based on corner
        corners = {
            "top-left": (padding, padding),
            "top-right": (width - text_width - padding, padding),
            "bottom-left": (padding, height - text_height - padding),
            "bottom-right": (width - text_width - padding, height - text_height - padding)
        }
        
        if corner == "random":
            corner = random.choice(list(corners.keys()))
        
        x, y = corners.get(corner, corners["bottom-right"])
        
        # Draw with semi-transparent background
        bg_padding = 10
        opacity_bg = random.randint(60, 90)
        opacity_text = random.randint(90, 140)
        
        # Background
        draw.rectangle(
            [x - bg_padding, y - bg_padding, 
             x + text_width + bg_padding, y + text_height + bg_padding],
            fill=(0, 0, 0, opacity_bg)
        )
        
        # Text
        draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity_text))
        
        return watermark, {"type": "corner", "text": text, "corner": corner, "opacity": opacity_text}
    
    def create_logo_style_watermark(self, img_size, text=None):
        """Create logo-style watermark with shape"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Select text
        if text is None:
            text = random.choice(self.brand_names)
        
        # Font size
        font_size = min(width, height) // 12
        font = self.get_font(font_size, bold=True)
        
        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center position
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw shape behind text
        shape_padding = 30
        opacity = random.randint(60, 100)
        
        shape_type = random.choice(["rectangle", "ellipse", "rounded_rectangle"])
        
        if shape_type == "rectangle":
            draw.rectangle(
                [x - shape_padding, y - shape_padding,
                 x + text_width + shape_padding, y + text_height + shape_padding],
                fill=(255, 255, 255, opacity),
                outline=(200, 200, 200, opacity + 20),
                width=3
            )
        elif shape_type == "ellipse":
            draw.ellipse(
                [x - shape_padding, y - shape_padding,
                 x + text_width + shape_padding, y + text_height + shape_padding],
                fill=(255, 255, 255, opacity),
                outline=(200, 200, 200, opacity + 20),
                width=3
            )
        else:  # rounded_rectangle
            # Simple rounded rectangle approximation
            radius = 20
            draw.rounded_rectangle(
                [x - shape_padding, y - shape_padding,
                 x + text_width + shape_padding, y + text_height + shape_padding],
                radius=radius,
                fill=(255, 255, 255, opacity),
                outline=(200, 200, 200, opacity + 20),
                width=3
            )
        
        # Draw text
        draw.text((x, y), text, font=font, fill=(0, 0, 0, opacity + 100))
        
        return watermark, {"type": "logo_style", "text": text, "shape": shape_type, "opacity": opacity}
    
    def create_circular_text_watermark(self, img_size, text=None):
        """Create circular text watermark"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Select text
        if text is None:
            text = random.choice(self.watermark_texts)
        
        # Repeat the text a random number of times
        repeat_count = random.randint(2, 4)  # Repeat 2-4 times
        repeated_text = ' '.join([text] * repeat_count)
        
        # Add bullet separator for circular pattern
        final_text = repeated_text + " • "
        
        # Center and radius
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 3
        
        # Font size
        font_size = radius // 8
        font = self.get_font(font_size)
        
        # Calculate how many times text fits around circle
        text_width = draw.textlength(final_text, font=font)
        circumference = 2 * math.pi * radius
        repetitions = int(circumference / text_width) + 1
        full_text = final_text * repetitions
        
        # Draw text in circle
        opacity = random.randint(70, 100)
        color = (255, 255, 255, opacity)
        
        angle_step = 360 / len(full_text)
        for i, char in enumerate(full_text):
            angle = math.radians(i * angle_step)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Create rotated character
            char_img = Image.new('RGBA', (font_size*2, font_size*2), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((font_size//2, font_size//2), char, font=font, fill=color)
            
            # Rotate to face outward
            rotated = char_img.rotate(-i * angle_step - 90, expand=1)
            
            # Paste
            watermark.paste(rotated, (int(x - font_size), int(y - font_size)), rotated)
        
        return watermark, {
            "type": "circular_text", 
            "text": text,
            "repeat_count": repeat_count,
            "final_text": repeated_text,
            "radius": radius, 
            "opacity": opacity
        }

    def create_wave_pattern_watermark(self, img_size, text=None):
        """Create wavy text pattern watermark"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Select text
        if text is None:
            text = random.choice(self.watermark_texts)
        
        # Repeat the text a random number of times
        repeat_count = random.randint(2, 5)  # Repeat 2-5 times
        final_text = ' '.join([text] * repeat_count)
        
        # Parameters
        font_size = min(width, height) // 20
        font = self.get_font(font_size)
        opacity = random.randint(70, 90)
        color = (200, 200, 200, opacity)
        
        # Wave parameters
        amplitude = height // 10
        frequency = 0.02
        vertical_spacing = height // 6
        
        # Draw wavy text lines
        for line_num in range(0, 7):
            y_base = line_num * vertical_spacing
            
            # Draw text along wave
            x = 0
            while x < width:
                y = y_base + amplitude * math.sin(frequency * x + line_num)
                draw.text((x, int(y)), final_text, font=font, fill=color)
                x += len(final_text) * font_size // 2
        
        return watermark, {
            "type": "wave_pattern", 
            "text": text,
            "repeat_count": repeat_count,
            "final_text": final_text,
            "amplitude": amplitude, 
            "opacity": opacity
        }

    def create_barcode_watermark(self, img_size, text=None):
        """Create barcode-style watermark"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Generate barcode pattern
        barcode_width = min(width, height) // 2
        barcode_height = barcode_width // 4
        
        # Position
        x_start = (width - barcode_width) // 2
        y_start = (height - barcode_height) // 2
        
        # Draw barcode
        opacity = random.randint(60, 140)
        num_bars = random.randint(30, 50)
        
        for i in range(num_bars):
            bar_width = random.randint(2, 8)
            x = x_start + (i * barcode_width // num_bars)
            
            if random.random() > 0.3:  # 70% chance of bar
                draw.rectangle([x, y_start, x + bar_width, y_start + barcode_height],
                            fill=(0, 0, 0, opacity))
        
        # Add text below barcode
        if text is None:
            text = f"{random.randint(100000, 999999)}-{random.randint(100, 999)}"
        
        font_size = barcode_height // 4
        font = self.get_font(font_size)
        
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        
        draw.text((x_start + (barcode_width - text_width) // 2, 
                y_start + barcode_height + 10),
                text, font=font, fill=(0, 0, 0, opacity))
        
        return watermark, {"type": "barcode", "text": text, "opacity": opacity}

    def create_qr_style_watermark(self, img_size, text=None):
        """Create QR code style watermark (simplified pattern)"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # QR code size
        qr_size = min(width, height) // 4
        module_size = qr_size // 20
        
        # Position
        x_start = (width - qr_size) // 2
        y_start = (height - qr_size) // 2
        
        opacity = random.randint(50, 90)
        
        # Draw corner markers (simplified)
        marker_size = module_size * 7
        for corner_x, corner_y in [(x_start, y_start), 
                                (x_start + qr_size - marker_size, y_start),
                                (x_start, y_start + qr_size - marker_size)]:
            # Outer square
            draw.rectangle([corner_x, corner_y, corner_x + marker_size, corner_y + marker_size],
                        fill=(0, 0, 0, opacity))
            # Inner white square
            draw.rectangle([corner_x + module_size, corner_y + module_size,
                        corner_x + marker_size - module_size, corner_y + marker_size - module_size],
                        fill=(255, 255, 255, opacity))
            # Center square
            draw.rectangle([corner_x + module_size*2, corner_y + module_size*2,
                        corner_x + marker_size - module_size*2, corner_y + marker_size - module_size*2],
                        fill=(0, 0, 0, opacity))
        
        # Random pattern in center
        for i in range(8, 20):
            for j in range(8, 20):
                if random.random() > 0.5:
                    x = x_start + i * module_size
                    y = y_start + j * module_size
                    draw.rectangle([x, y, x + module_size, y + module_size],
                                fill=(0, 0, 0, opacity))
        
        return watermark, {"type": "qr_style", "size": qr_size, "opacity": opacity}

    def create_mosaic_watermark(self, img_size, text=None):
        """Create mosaic/tiled watermark with varying opacity"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Select text
        if text is None:
            text = random.choice(["PROTECTED", "SECURE", "PRIVATE"])
        
        # Tile parameters
        tile_size = min(width, height) // 8
        font_size = tile_size // 3
        font = self.get_font(font_size)
        
        # Create mosaic pattern
        for x in range(0, width, tile_size):
            for y in range(0, height, tile_size):
                # Random opacity for each tile
                opacity = random.randint(10, 60)
                color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255), opacity)
                
                # Draw tile border
                draw.rectangle([x, y, x + tile_size, y + tile_size], outline=color, width=1)
                
                # Draw text in tile
                text_bbox = draw.textbbox((0, 0), text[:4], font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = x + (tile_size - text_width) // 2
                text_y = y + (tile_size - text_height) // 2
                
                draw.text((text_x, text_y), text[:4], font=font, fill=color)
        
        return watermark, {"type": "mosaic", "text": text, "tile_size": tile_size}

    def create_fingerprint_watermark(self, img_size, text=None):
        """Create fingerprint-style watermark"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Center and size
        center_x = width // 2
        center_y = height // 2
        size = min(width, height) // 3
        
        opacity = random.randint(30, 70)
        color = (100, 100, 100, opacity)
        
        # Draw concentric ellipses with gaps
        num_rings = 15
        for i in range(num_rings):
            radius = size * (i + 1) / num_rings
            
            # Create gaps in the rings
            start_angle = random.randint(0, 30)
            for angle in range(start_angle, 360, 45):
                end_angle = angle + random.randint(20, 35)
                
                # Draw arc
                bbox = [center_x - radius, center_y - radius * 0.7,
                    center_x + radius, center_y + radius * 0.7]
                
                draw.arc(bbox, start=angle, end=end_angle, fill=color, width=2)
        
        # Add text below if provided
        if text:
            font_size = size // 10
            font = self.get_font(font_size)
            
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
            draw.text((center_x - text_width // 2, center_y + size),
                    text, font=font, fill=color)
        
        return watermark, {"type": "fingerprint", "size": size, "opacity": opacity}

    def create_radial_watermark(self, img_size, text=None):
        """Create radial/sunburst pattern watermark"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Select text
        if text is None:
            text = random.choice(self.watermark_texts)
        
        # Repeat the text a random number of times
        repeat_count = random.randint(2, 4)  # Repeat 2-4 times
        final_text = ' '.join([text] * repeat_count)
        
        # Center point
        center_x = width // 2
        center_y = height // 2
        
        # Parameters
        num_rays = 12
        font_size = min(width, height) // 25
        font = self.get_font(font_size)
        opacity = random.randint(80, 150)
        
        # Draw radial lines with text
        for i in range(num_rays):
            angle = (360 / num_rays) * i
            angle_rad = math.radians(angle)
            
            # Calculate end point
            end_x = center_x + (width // 2) * math.cos(angle_rad)
            end_y = center_y + (height // 2) * math.sin(angle_rad)
            
            # Draw line
            draw.line([(center_x, center_y), (end_x, end_y)], 
                    fill=(200, 200, 200, opacity // 2), width=1)
            
            # Draw text along ray
            steps = 5
            for step in range(1, steps):
                x = center_x + (step * (end_x - center_x) / steps)
                y = center_y + (step * (end_y - center_y) / steps)
                
                # Create rotated text
                txt_img = Image.new('RGBA', (len(final_text) * font_size, font_size * 2), (0, 0, 0, 0))
                txt_draw = ImageDraw.Draw(txt_img)
                txt_draw.text((0, 0), final_text, font=font, fill=(255, 255, 255, opacity))
                
                # Rotate to align with ray
                rotated = txt_img.rotate(-angle + 90, expand=1)
                
                # Paste
                watermark.paste(rotated, (int(x - len(final_text) * font_size // 2), 
                                        int(y - font_size)), rotated)
        
        return watermark, {
            "type": "radial", 
            "text": text,
            "repeat_count": repeat_count,
            "final_text": final_text,
            "num_rays": num_rays, 
            "opacity": opacity
        }

    def create_grid_watermark(self, img_size, text=None):
        """Create grid pattern watermark"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Select text
        if text is None:
            text = random.choice(["•", "×", "+", "◊", "○"])
        
        # Grid parameters
        grid_size = random.randint(50, 100)
        opacity = random.randint(80, 150)
        color = (150, 150, 150, opacity)
        
        # Font for symbols
        font_size = grid_size // 3
        font = self.get_font(font_size)
        
        # Draw grid
        for x in range(0, width, grid_size):
            for y in range(0, height, grid_size):
                draw.text((x, y), text, font=font, fill=color)
        
        return watermark, {"type": "grid", "symbol": text, "grid_size": grid_size, "opacity": opacity}
    
    def create_stamp_watermark(self, img_size, text=None):
        """Create stamp-style watermark with random position"""
        width, height = img_size
        
        # Create transparent image
        watermark = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Select text
        if text is None:
            text = random.choice(["APPROVED", "CERTIFIED", "VERIFIED", "OFFICIAL"])
        
        # Stamp size
        stamp_size = min(width, height) // 4
        
        # Random position with margin to ensure stamp stays within bounds
        margin = stamp_size + 20  # Extra margin to account for rotation
        center_x = random.randint(margin, width - margin)
        center_y = random.randint(margin, height - margin)
        
        # Draw circular stamp
        opacity = random.randint(90, 150)
        
        # Outer circle
        draw.ellipse(
            [center_x - stamp_size, center_y - stamp_size,
            center_x + stamp_size, center_y + stamp_size],
            outline=(255, 0, 0, opacity),
            width=8
        )
        
        # Inner circle
        inner_size = stamp_size - 20
        draw.ellipse(
            [center_x - inner_size, center_y - inner_size,
            center_x + inner_size, center_y + inner_size],
            outline=(255, 0, 0, opacity),
            width=4
        )
        
        # Text in center
        font_size = stamp_size // 4
        font = self.get_font(font_size, bold=True)
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.text(
            (center_x - text_width // 2, center_y - text_height // 2),
            text, font=font, fill=(255, 0, 0, opacity)
        )
        
        # Add date
        date_text = datetime.now().strftime("%Y-%m-%d")
        date_font = self.get_font(font_size // 2)
        
        bbox = draw.textbbox((0, 0), date_text, font=date_font)
        date_width = bbox[2] - bbox[0]
        
        draw.text(
            (center_x - date_width // 2, center_y + stamp_size // 3),
            date_text, font=date_font, fill=(255, 0, 0, opacity)
        )
        
        # Rotate stamp randomly
        angle = random.randint(-30, 30)
        watermark = watermark.rotate(angle, expand=0)
        
        return watermark, {
            "type": "stamp", 
            "text": text, 
            "date": date_text, 
            "angle": angle, 
            "opacity": opacity,
            "position": (center_x, center_y)
        }
    
    def apply_watermark(self, image_path, output_dir="output", watermark_type="random"):
        """Apply watermark to image"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGBA')
            img_width, img_height = img.size
            
            # Skip very small images
            if img_width < 100 or img_height < 100:
                print(f"Skipping {image_path} - image too small")
                return None, None
            
            # Choose watermark type
           # In apply_watermark method, update the watermark_types dictionary:
            watermark_types = {
                "text": self.create_text_watermark,
                "diagonal": self.create_diagonal_watermark,
                "corner": self.create_corner_watermark,
                "logo": self.create_logo_style_watermark,
                "grid": self.create_grid_watermark,
                "stamp": self.create_stamp_watermark,
                "circular": self.create_circular_text_watermark,
                "wave": self.create_wave_pattern_watermark,
                "mosaic": self.create_mosaic_watermark,
                "radial": self.create_radial_watermark,
                "barcode": self.create_barcode_watermark,
                "qr": self.create_qr_style_watermark,
                "fingerprint": self.create_fingerprint_watermark
            }

            # Update the random selection in process_folder method:
            if watermark_type == "random":
                watermark_type = random.choice(["text", "diagonal", "corner", "logo", "grid", "stamp",
                                            "circular", "wave", "mosaic", "radial", "barcode", "qr", "fingerprint"])

            
            # Create watermark
            watermark_func = watermark_types.get(watermark_type, self.create_text_watermark)
            watermark, watermark_info = watermark_func((img_width, img_height))
            
            # Apply watermark
            watermarked = Image.alpha_composite(img, watermark)
            
            # Convert back to RGB
            watermarked = watermarked.convert('RGB')
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_name = base_name #f"{base_name}_watermarked_{timestamp}"
            
            # Save image
            output_image_path = os.path.join(output_dir, f"{output_name}.png")
            watermarked.save(output_image_path)
            
            # Create metadata
            metadata = {
                "source_image": os.path.basename(image_path),
                "timestamp": datetime.now().isoformat(),
                "image_size": {"width": img_width, "height": img_height},
                "watermark": watermark_info
            }
            
            # Save JSON
            output_json_path = os.path.join(output_dir, f"{output_name}.json")
            with open(output_json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_image_path, output_json_path
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None, None
    
    def process_folder(self, input_folder, output_dir="output", watermark_type="random",
                      image_extensions=None, max_images=None):
        """Process all images in a folder"""
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext.upper()}")))
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            print(f"No images found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} images in {input_folder}")
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
            print(f"Processing first {max_images} images")
        
        # Process statistics
        successful = 0
        failed = 0
        watermark_counts = {}
        
        # Process each image
        try:
            from tqdm import tqdm
            iterator = tqdm(image_files, desc="Adding watermarks")
        except ImportError:
            print("Install tqdm for progress bar: pip install tqdm")
            iterator = image_files
        
        for image_path in iterator:
            # For variety, use different watermark types
            if watermark_type == "random":
                #current_type = random.choice(["text", "diagonal", "corner", "logo", "grid", "stamp"])
                current_type = random.choice(["text", "corner", "logo", "stamp"])
            else:
                current_type = watermark_type
            
            output_img, output_json = self.apply_watermark(
                image_path, 
                output_dir, 
                current_type
            )
            
            if output_img and output_json:
                successful += 1
                watermark_counts[current_type] = watermark_counts.get(current_type, 0) + 1
            else:
                failed += 1
        
        # Summary
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful} images")
        print(f"Failed: {failed} images")
        print(f"Output saved to: {output_dir}")
        print("\nWatermark types used:")
        for wtype, count in watermark_counts.items():
            print(f"  {wtype}: {count}")
        
        # Create summary JSON
        summary = {
            "processing_date": datetime.now().isoformat(),
            "input_folder": input_folder,
            "output_folder": output_dir,
            "total_images": len(image_files),
            "successful": successful,
            "failed": failed,
            "watermark_types": watermark_counts
        }
        
        summary_path = os.path.join(output_dir, "watermark_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

class WatermarkSampleGenerator:
    """Generate sample watermarks for preview"""
    
    @staticmethod
    def generate_samples(output_dir="watermark_samples"):
        """Generate sample images showing different watermark types"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create base image
        width, height = 800, 600
        base_color = (240, 240, 240)
        
        generator = WatermarkGenerator()
        
        # In WatermarkSampleGenerator.generate_samples method, update the watermark_types list:
        watermark_types = [
            ("text", "Simple Text Watermark"),
            #("diagonal", "Diagonal Pattern Watermark"),
            ("corner", "Corner Watermark"),
            ("logo", "Logo Style Watermark"),
            #("grid", "Grid Pattern Watermark"),
            ("stamp", "Stamp Style Watermark"),
            ("circular", "Circular Text Watermark"),
            ("wave", "Wave Pattern Watermark"),
            ("mosaic", "Mosaic Tile Watermark"),
            ("radial", "Radial Sunburst Watermark"),
            #("barcode", "Barcode Style Watermark"),
            #("qr", "QR Code Style Watermark"),
            #("fingerprint", "Fingerprint Style Watermark")
        ]

# And add the corresponding cases in the watermark generation section:

        print("Generating watermark samples...")
        
        for wtype, description in watermark_types:
            # Create sample image
            img = Image.new('RGB', (width, height), base_color)
            draw = ImageDraw.Draw(img)
            
            # Add some content to make it look like a document
            draw.rectangle([50, 50, width-50, height-50], outline=(200, 200, 200), width=2)
            draw.text((60, 60), f"Sample Document - {description}", fill=(100, 100, 100))
            
            # Add some fake text lines
            y_pos = 120
            for i in range(10):
                draw.line([(80, y_pos), (width-80, y_pos)], fill=(220, 220, 220), width=1)
                y_pos += 40
            
            # Convert to RGBA for watermarking
            img = img.convert('RGBA')
            
            # Apply watermark
            if wtype == "text":
                watermark, _ = generator.create_text_watermark((width, height), "SAMPLE")
            elif wtype == "diagonal":
                watermark, _ = generator.create_diagonal_watermark((width, height), "CONFIDENTIAL")
            elif wtype == "corner":
                watermark, _ = generator.create_corner_watermark((width, height), "© 2024 Example Corp", "bottom-right")
            elif wtype == "logo":
                watermark, _ = generator.create_logo_style_watermark((width, height), "DEMO VERSION")
            elif wtype == "grid":
                watermark, _ = generator.create_grid_watermark((width, height))
            elif wtype == "stamp":
                watermark, _ = generator.create_stamp_watermark((width, height), "APPROVED")
            elif wtype == "circular":
                watermark, _ = generator.create_circular_text_watermark((width, height), "CIRCULAR TEXT")
            elif wtype == "wave":
                watermark, _ = generator.create_wave_pattern_watermark((width, height), "WAVE")
            elif wtype == "mosaic":
                watermark, _ = generator.create_mosaic_watermark((width, height), "SECURE")
            elif wtype == "radial":
                watermark, _ = generator.create_radial_watermark((width, height), "RADIAL")
            elif wtype == "barcode":
                watermark, _ = generator.create_barcode_watermark((width, height))
            elif wtype == "qr":
                watermark, _ = generator.create_qr_style_watermark((width, height))
            elif wtype == "fingerprint":
                watermark, _ = generator.create_fingerprint_watermark((width, height), "BIOMETRIC")
                    
            
            # Apply watermark
            watermarked = Image.alpha_composite(img, watermark)
            watermarked = watermarked.convert('RGB')
            
            # Save sample
            output_path = os.path.join(output_dir, f"sample_{wtype}_watermark.png")
            watermarked.save(output_path)
            print(f"Created: {output_path}")
        
        print(f"\nSample watermarks saved to: {output_dir}")

def main():
    # Create generator instance
    generator = WatermarkGenerator()
    
    # Configuration
    INPUT_FOLDER = "wikiart_5k"  # Change this to your folder path
    OUTPUT_FOLDER = "watermark_new"
    WATERMARK_TYPE = "text"  # Options: "text", "diagonal", "corner", "logo", "grid", "stamp", "random"
    MAX_IMAGES = None  # None to process all, or specify a number
    
    # Generate samples first (optional)
    generate_samples = input("Generate watermark samples? (y/n): ").lower() == 'y'
    if generate_samples:
        WatermarkSampleGenerator.generate_samples()
        print("\nCheck the 'watermark_samples' folder to see different watermark types.\n")
    
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found!")
        print("Creating sample folder with test images...")
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        
        # Create sample images
        for i in range(3):
            img = Image.new('RGB', (800, 600), color=(random.randint(200, 255), 
                                                     random.randint(200, 255), 
                                                     random.randint(200, 255)))
            draw = ImageDraw.Draw(img)
            
            # Add some content
            draw.rectangle([40, 40, 760, 560], outline=(150, 150, 150), width=2)
            draw.text((50, 50), f"Sample Document {i+1}", fill=(50, 50, 50))
            
            # Add lines to simulate text
            y = 100
            for j in range(12):
                draw.line([(60, y), (740, y)], fill=(200, 200, 200), width=1)
                y += 40
            
            img.save(os.path.join(INPUT_FOLDER, f"sample_document_{i+1}.jpg"))
        
        print(f"Created 3 sample images in {INPUT_FOLDER}")
    
    # Process the folder
    print(f"\nProcessing images with '{WATERMARK_TYPE}' watermark type...")
    generator.process_folder(
        INPUT_FOLDER,
        OUTPUT_FOLDER,
        watermark_type=WATERMARK_TYPE,
        max_images=MAX_IMAGES
    )

if __name__ == "__main__":
    main()