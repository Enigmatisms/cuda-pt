import os
import cv2
import natsort
from PIL import Image

def convert_png_to_jpg(input_folder, output_folder, jpg_compress=True, quality=97):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort()
    
    for png_file in png_files:
        png_path = os.path.join(input_folder, png_file)
        jpg_path = os.path.join(output_folder, png_file.replace('.png', '.jpg'))
        
        with Image.open(png_path) as img:
            if jpg_compress:
                img = img.convert("RGB")
                img.save(jpg_path, "JPEG", quality=quality)
            else:
                img.save(jpg_path, "JPEG")

            print(f"Converted {png_file} to {jpg_path}")

def create_video_from_images(image_folder, output_video_path, frame_rate=30, resolution=(1920, 1080), first_repeat = 30):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_files = natsort.natsorted(image_files)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, resolution)
    
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(image_folder, image_file)
        
        img = cv2.imread(img_path)
        
        img_resized = cv2.resize(img, resolution)
        
        if first_repeat > 0 and i == 0:
            for j in range(first_repeat):
                video_writer.write(img_resized)
        else:
            video_writer.write(img_resized)
        print(f"Adding {image_file} to video")
    
    video_writer.release()
    print(f"Video saved as {output_video_path}")

if __name__ == "__main__":
    input_folder = "./volume_imgs"
    output_folder = "./compressed" 
    video_output_path = "output_video.mp4"
    convert_png_to_jpg(input_folder, output_folder, jpg_compress=True, quality=95)
    create_video_from_images(output_folder, video_output_path, frame_rate=30, resolution=(1024, 1024))
