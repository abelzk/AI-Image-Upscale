import os
import urllib.request
import rarfile
import subprocess
from PIL import Image
import sys
from realesrgan import RealESRGANer

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_model(model_path):
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.4.0/RealESRGAN_x4plus.pth"
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print("Download completed.")

model_path = 'RealESRGAN_x4plus.pth'
if not os.path.exists(model_path):
    download_model(model_path)

model = RealESRGANer(
    scale=scale_factor,
    model_path=model_path,
    tile=256,
    tile_pad=10,
    pre_pad=0,
    half=False
)

def extract_rar(input_rar, extract_path, password=None):
    rarfile.UNRAR_TOOL = "unrar"
    with rarfile.RarFile(input_rar) as rar_ref:
        if password:
            rar_ref.extractall(path=extract_path, pwd=password)
        else:
            rar_ref.extractall(path=extract_path)

def compress_folder_to_rar(folder_path, output_rar, password=None):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    command = ["rar", "a", "-p" + password, output_rar]
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            command.append(file_path)
    
    subprocess.run(command, check=True)

def upscale_image(image_path, output_path, scale_factor=4):
    model_path = 'RealESRGAN_x4plus.pth'
    model = load_model(model_path, scale_factor)

    try:
        print(f"Starting to process: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input file not found: {image_path}")
            
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output, _ = model.enhance(img)
        output.save(output_path)
        print(f"Successfully saved upscaled image to: {output_path}")

    except Exception as e:
        print(f"ERROR processing {image_path}: {e}")
        raise

def process_images(input_folder, output_folder):
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                upscale_image(input_path, output_path)

    input_count = sum(1 for root, _, files in os.walk(input_folder) 
                     for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    output_count = sum(1 for root, _, files in os.walk(output_folder) 
                      for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    print(f"Input images found: {input_count}")
    print(f"Output images processed: {output_count}")

def main():
    work_dir = os.getcwd()
    input_rar = os.path.join(work_dir, 'input.rar')
    temp_dir = os.path.join(work_dir, 'temp')
    output_dir = os.path.join(work_dir, 'output')
    output_rar = os.path.join(work_dir, 'upscaled_images.rar')
    
    password = os.getenv('RAR_PASSWORD')
    if not password:
        print("Error: RAR_PASSWORD environment variable is not set.")
        sys.exit(1)

    create_directory(temp_dir)
    create_directory(output_dir)

    try:
        print("Extracting input RAR file...")
        extract_rar(input_rar, temp_dir, password=password)

        print("Processing images...")
        process_images(temp_dir, output_dir)

        print("Creating output RAR file...")
        compress_folder_to_rar(output_dir, output_rar, password=password)

        print("Process completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    finally:
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)

if __name__ == "__main__":
    main()
