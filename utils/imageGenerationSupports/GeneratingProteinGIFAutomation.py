import os
from PIL import Image, ImageDraw, ImageFont
import cv2

proteinFolder = "/data2/TestEman2/ProteinPNG"
output_folder = "/data2/TestEman2/ProteinGIF"

all_folder_paths = [os.path.join(proteinFolder, f) for f in os.listdir(proteinFolder)]
for folder_path in all_folder_paths:
    nameProtein = folder_path.split("/")[-1]
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

    frames = []

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path).convert("RGB")
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text = f"{nameProtein}: {filename.split('.')[0]}"
        
        padding = 20
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
        x = padding
        y = img.height - text_height - padding
        
        draw.rectangle(
            [x-10, y-10, x+text_width+10, y+text_height+10],
            fill=(0, 0, 0)
        )
        
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        frames.append(img)

    frames[0].save(
        f"{output_folder}/{nameProtein}.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=100,  # milliseconds per frame
        loop=0
    )

print("GIF transformation Finished!")