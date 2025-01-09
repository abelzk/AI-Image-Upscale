import os
import rarfile
import subprocess
from PIL import Image
import torch
from super_image import EdsrModel, ImageLoader
import sys

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_rar(input_rar, extract_path, password=None):
    rarfile.UNRAR_TOOL = "unrar"  # Ensure the `unrar` tool is installed on the system
    with rarfile.RarFile(input_rar) as rar_ref:
        if password:
            rar_ref.extractall(path=extract_path, pwd=password)
        else:
            rar_ref.extractall(path=extract_path)

def compress_folder_to_rar(folder_path, output_rar, password=None):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    # Use subprocess to call the `rar` command for creating a password-protected RAR archive
    command = ["rar", "a", "-p" + password, output_rar]
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            command.append(file_path)
    
    subprocess.run(command, check=True)

def upscale_image(image_path, output_path, scale_factor=4):
    try:
        print(f"Starting to process: {image_path}")
        
        # Verify input file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input file not found: {image_path}")
            
        # Load and verify input image
        try:
            with Image.open(image_path) as img:
                # Print image details for debugging
                print(f"Original image size: {img.size}, mode: {img.mode}")
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("Converted image to RGB mode")
        except Exception as img_error:
            raise Exception(f"Failed to open/process input image: {str(img_error)}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load model with error handling
        try:
            model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale_factor)
            print("Successfully loaded EDSR model")
        except Exception as model_error:
            raise Exception(f"Failed to load EDSR model: {str(model_error)}")

        # Process image with detailed error handling
        try:
            inputs = ImageLoader.load_image(img)
            print("Successfully loaded image into model format")
            
            preds = model(inputs)
            print("Successfully generated upscaled prediction")
            
            ImageLoader.save_image(preds, output_path)
            print(f"Successfully saved upscaled image to: {output_path}")
            
            # Verify output file exists and has size > 0
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("Output file is missing or empty")
                
        except Exception as process_error:
            raise Exception(f"Failed during image processing: {str(process_error)}")

    except Exception as e:
        print(f"ERROR processing {image_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        raise  # Re-raise the exception for the calling function

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

    input_count = sum(1 for root, _, files in os.walk(input_folder) 
                     for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    output_count = sum(1 for root, _, files in os.walk(output_folder) 
                      for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    print(f"Input images found: {input_count}")
    print(f"Output images processed: {output_count}")

def main():
    # Setup directories
    work_dir = os.getcwd()
    input_rar = os.path.join(work_dir, 'input.rar')
    temp_dir = os.path.join(work_dir, 'temp')
    output_dir = os.path.join(work_dir, 'output')
    output_rar = os.path.join(work_dir, 'upscaled_images.rar')
    
    # Get the password from environment variables
    password = os.getenv('RAR_PASSWORD')
    if not password:
        print("Error: RAR_PASSWORD environment variable is not set.")
        sys.exit(1)

    # Create necessary directories
    create_directory(temp_dir)
    create_directory(output_dir)

    try:
        # Extract input rar
        print("Extracting input RAR file...")
        extract_rar(input_rar, temp_dir, password=password)

        # Process images
        print("Processing images...")
        process_images(temp_dir, output_dir)

        # Create output rar
        print("Creating output RAR file...")
        compress_folder_to_rar(output_dir, output_rar, password=password)

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
