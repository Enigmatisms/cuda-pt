import os
import natsort
import numpy as np
from PIL import Image

def convert_png_to_jpg(input_folder, output_folder, jpg_compress=True, quality=97):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort()
    
    for png_file in png_files:
        png_path = os.path.join(input_folder, png_file)
        jpg_path = os.path.join(output_folder, png_file.replace('.png', '.jpg' if jpg_compress else '.png'))
        
        with Image.open(png_path) as img:
            if jpg_compress:
                img = img.convert("RGB")
                img.save(jpg_path, "JPEG", quality=quality)
            else:
                img.save(jpg_path, "PNG")

            print(f"Converted {png_file} to {jpg_path}")

import imageio
from PIL import Image

def create_video_from_images(image_folder, output_video_path, frame_rate=30, quality=10, first_repeat = 40):
    image_files = [f for f in os.listdir(image_folder)]
    image_files = natsort.natsorted(image_files)
    
    writer = imageio.get_writer(output_video_path, fps=frame_rate, codec='libx264', quality=quality)

    for i, image_file in enumerate(image_files):
        img_path = os.path.join(image_folder, image_file)
        
        img = Image.open(img_path)

        if first_repeat > 0 and i == 0:
            for j in range(first_repeat):
                writer.append_data(np.array(img))
        else:
            writer.append_data(np.array(img))

        print(f"Adding {image_file} to video")

    writer.close()
    print(f"Video saved as {output_video_path}")

if __name__ == "__main__":
    input_folder = "./volume_imgs"
    output_folder = "./compressed" 
    video_output_path = "output_video.mp4"
    # convert_png_to_jpg(input_folder, output_folder, jpg_compress=False, quality=99)
    create_video_from_images(input_folder, video_output_path, frame_rate=30, quality=8)
