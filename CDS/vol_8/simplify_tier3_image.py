import os
from PIL import Image, ImageDraw, ImageFont

IN_PATH = r"C:\Users\brook\.gemini\antigravity\brain\574d389b-b6ed-48ef-afb4-dd86c25c56c1\media__1774077151985.png"
OUT_PATH = r"f:\ADDS\CDS\Simplified_Tier3_ADDS_Render.png"

def get_font(size, bold=False):
    font_name = "malgunbd.ttf" if bold else "malgun.ttf"
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", font_name)
    try:
        return ImageFont.truetype(font_path, size)
    except:
        return ImageFont.load_default()

def draw_glass_overlay(canvas, bounds, fill_color, text1, text2, text3):
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    
    # We will draw a dark glass effect over the previous text
    d.rounded_rectangle(bounds, radius=10, fill=fill_color)
    d.rounded_rectangle(bounds, radius=10, outline=(100, 200, 255, 180), width=1)
    
    # Text positions
    cx = (bounds[0] + bounds[2]) // 2
    f_title = get_font(13, True)
    f_sub = get_font(11, False)
    
    d.text((cx, bounds[1] + 15), text1, font=f_title, fill=(255, 255, 255, 255), anchor="mm")
    d.text((cx, bounds[1] + 40), text2, font=f_sub, fill=(180, 255, 200, 255), anchor="mm")
    if text3:
        d.text((cx, bounds[1] + 60), text3, font=f_sub, fill=(200, 200, 200, 255), anchor="mm")
        
    canvas.paste(Image.alpha_composite(canvas.convert('RGBA'), overlay), (0,0))
    return canvas

def main():
    if not os.path.exists(IN_PATH):
        print(f"Error: {IN_PATH} does not exist.")
        return
        
    img = Image.open(IN_PATH).convert("RGBA")
    
    # The image is 640x640. Tier 3 boxes are usually around y=410 to 520
    # Let's try to cover the lower half of each bounding box carefully
    y1 = 435
    y2 = 515
    fill_bg = (15, 30, 50, 240)  # Very dark blue glass to obscure existing text
    
    # Box 1
    # Assuming left box is from x=20 to x=210
    img = draw_glass_overlay(img, [40, y1, 220, y2], fill_bg, 
                             "1. Standard FOLFOX", 
                             "Predicted Efficacy: 78%", 
                             "Confidence: 94% (High)")
                             
    # Box 2
    img = draw_glass_overlay(img, [230, y1, 410, y2], fill_bg, 
                             "2. CAPOX + Bevacizumab", 
                             "Predicted Efficacy: 82%", 
                             "Confidence: 87% (Alt)")
                             
    # Box 3
    img = draw_glass_overlay(img, [420, y1, 600, y2], fill_bg, 
                             "3. Immunotherapy Combo", 
                             "Valid for MSI-High Only", 
                             "Predicted Efficacy: 85%")

    img.convert("RGB").save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
