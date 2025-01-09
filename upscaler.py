import os
import zipfile
from PIL import Image
import torch
from super_image import EdsrModel, ImageLoader
import sys

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_zip(input_zip, extract_path):
    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def compress_folder(folder_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def upscale_image(image_path, output_path, scale_factor=4):
    try:
        # Load model with weights_only=True
        model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale_factor, weights_only=True)
        
        # Rest of the function remains the same
        inputs = ImageLoader.load_image(image_path)
        preds = model(inputs)
        ImageLoader.save_image(preds, output_path)
        print(f"Successfully upscaled: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_images(input_folder, output_folder):
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                
                # Create necessary subdirectories
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                upscale_image(input_path, output_path)
                # Add this at the end of the function
    input_count = sum(1 for root, _, files in os.walk(input_folder) 
                     for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    output_count = sum(1 for root, _, files in os.walk(output_folder) 
                      for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    print(f"Input images found: {input_count}")
    print(f"Output images processed: {output_count}")

def main():
    # Setup directories
    work_dir = os.getcwd()
    input_zip = os.path.join(work_dir, 'input.zip')
    temp_dir = os.path.join(work_dir, 'temp')
    output_dir = os.path.join(work_dir, 'output')
    output_zip = os.path.join(work_dir, 'upscaled_images.zip')

    # Create necessary directories
    create_directory(temp_dir)
    create_directory(output_dir)

    try:
        # Extract input zip
        print("Extracting input zip file...")
        extract_zip(input_zip, temp_dir)

        # Process images
        print("Processing images...")
        process_images(temp_dir, output_dir)

        # Create output zip
        print("Creating output zip file...")
        compress_folder(output_dir, output_zip)

        print("Process completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)

if __name__ == "__main__":
    main()
