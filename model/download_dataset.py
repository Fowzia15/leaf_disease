import subprocess
import os
import shutil
import imghdr

def is_image_file(filepath):
    """Check if the file is a valid image."""
    try:
        return imghdr.what(filepath) in ['jpeg', 'jpg', 'png', 'gif']
    except Exception:
        return False

def download_plantdoc():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Define the temporary directory for git clone
    temp_dir = "temp_plantdoc"
    target_dir = 'data/leaf_diseases'
    
    # Clean up existing directories if they exist
    if os.path.exists(temp_dir):
        print(f"Cleaning up existing {temp_dir} directory...")
        shutil.rmtree(temp_dir)
    
    try:
        print("Downloading PlantDoc dataset from GitHub...")
        # Clone the repository
        subprocess.run(
            ["git", "clone", "https://github.com/pratikkayal/PlantDoc-Dataset.git", temp_dir], 
            check=True
        )
        
        # Organize the dataset into the required structure
        source_dir = os.path.join(temp_dir, 'train')
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Train directory not found in {temp_dir}")
        
        print("Organizing dataset...")
        # Create target directory
        if os.path.exists(target_dir):
            print(f"Cleaning up existing {target_dir} directory...")
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        
        # Move and organize files
        class_counts = {}
        for disease_dir in os.listdir(source_dir):
            # Skip hidden directories
            if disease_dir.startswith('.'):
                continue
                
            source_disease_path = os.path.join(source_dir, disease_dir)
            target_disease_path = os.path.join(target_dir, disease_dir.lower().replace(' ', '_'))
            
            if os.path.isdir(source_disease_path):
                # Create disease directory in target
                os.makedirs(target_disease_path, exist_ok=True)
                image_count = 0
                
                # Move all images
                for img_file in os.listdir(source_disease_path):
                    if img_file.startswith('.'):  # Skip hidden files
                        continue
                        
                    source_file = os.path.join(source_disease_path, img_file)
                    if not is_image_file(source_file):  # Skip non-image files
                        print(f"Skipping non-image file: {source_file}")
                        continue
                        
                    target_file = os.path.join(target_disease_path, img_file)
                    shutil.copy2(source_file, target_file)
                    image_count += 1
                
                class_counts[disease_dir] = image_count
        
        # Clean up
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
        print("\nDataset download and organization complete!")
        print("\nDataset structure:")
        total_images = 0
        for class_name, count in sorted(class_counts.items()):
            print(f"- {class_name}: {count} images")
            total_images += count
        print(f"\nTotal number of images: {total_images}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        # Clean up in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return
    except Exception as e:
        print(f"Error processing dataset: {e}")
        # Clean up in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return

if __name__ == "__main__":
    download_plantdoc() 