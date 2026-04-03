import os
from PIL import Image

def combine_images():
    input_dir = r"f:\ADDS\CDS\final_pitch_slides"
    output_path = r"f:\ADDS\CDS\SaMD_Certification_Summary.png"
    
    img_files = [
        "Slide_11_Composite.png",
        "Slide_12_Composite.png",
        "Slide_14_Composite.png"
    ]
    
    images = []
    for f in img_files:
        path = os.path.join(input_dir, f)
        if os.path.exists(path):
            images.append(Image.open(path))
        else:
            print(f"Error: {path} not found.")
            return

    # Assume all images are the same size
    widths, heights = zip(*(i.size for i in images))
    
    max_width = max(widths)
    total_height = sum(heights)
    
    # Add a gap between images
    gap = 20
    total_height += gap * (len(images) - 1)
    
    bg_color = (250, 250, 250)
    new_im = Image.new('RGB', (max_width, total_height), bg_color)
    
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1] + gap
        
    new_im.save(output_path, quality=95)
    print(f"Successfully combined images and saved to {output_path}")

if __name__ == "__main__":
    combine_images()
