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
        print(f"Starting to process: {image_path}")
        
        # Verify input file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input file not found: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        
        # Load model
        model = EdsrModel.from_pretrained('edsr', scale=scale_factor)
        
        # Upscale image
        upscaled_image = model(image)
        
        # Save upscaled image
        upscaled_image.save(output_path)
        print(f"Upscaled image saved to: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_zip(input_zip, output_zip, temp_dir='temp', scale_factor=4):
    # Create temporary directory
    create_directory(temp_dir)
    
    # Extract zip file
    extract_zip(input_zip, temp_dir)
    
    # Process each image in the extracted folder
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            output_path = os.path.join(temp_dir, f"upscaled_{file}")
            upscale_image(file_path, output_path, scale_factor)
    
    # Compress the upscaled images into a new zip file
    compress_folder(temp_dir, output_zip)
    
    # Clean up temporary directory
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(temp_dir)

if __name__ == "__main__":
    input_zip = 'input.zip'
    output_zip = 'output.zip'
    process_zip(input_zip, output_zip)
