import shutil
from pathlib import Path
from src.preprocess import DataPreprocessor
import src.visualization as visualization
import src.utils as utils
import src.config as config
import datetime
import numpy as np
import random

def setup_directories(output_dir, vis_dir=None):
    temp_data_dir = output_dir / "temp_test_data"
    utils.ensure_dir(output_dir)
    
    if temp_data_dir.exists():
        shutil.rmtree(temp_data_dir)
        
    utils.ensure_dir(temp_data_dir)

    if vis_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        vis_dir = utils.ensure_dir(output_dir / "visualizations" / timestamp)
    else:
        utils.ensure_dir(vis_dir)
        
    return temp_data_dir, vis_dir

def copy_sample_files(data_dir, temp_data_dir, img_files):
    """
    Copy sample files to the temporary directory
    """
    # Create subdirectories based on FBP dataset structure
    utils.ensure_dir(temp_data_dir / "Image_16bit_BGRNir")
    utils.ensure_dir(temp_data_dir / "Annotation__index")
    
    # List to store tuples of (img_file, mask_file)
    original_files = []
    
    # Copy files to the temporary directory
    for img_file in img_files:
        if img_file.suffix.lower() == '.tiff':
            target_dir = "Image_16bit_BGRNir"
        
        mask_index_file = data_dir / "Annotation__index" / f"{img_file.stem}_24label.png"
        
        shutil.copy(img_file, temp_data_dir / target_dir / img_file.name)
        
        if mask_index_file.exists():
            shutil.copy(mask_index_file, temp_data_dir / "Annotation__index" / mask_index_file.name)
            mask_file = mask_index_file
        else:
            print(f"Warning: No annotation found for {img_file.name}")
            continue
            
        # Store the original file information for visualization
        original_files.append((img_file, mask_file))
        
    return original_files

def visualize_tiles(preprocessor, img_file, mask_file, vis_dir):
    img_id = img_file.stem
    
    tiles_dir = temp_data_dir / "tiles"
    tiles = list(tiles_dir.glob(f"{img_id}_*_*.npy"))
    
    if not tiles:
        print(f"No tiles found for {img_id}")
        return
    
    # Choose first tile for fixed visualization
    fixed_tile_path = tiles[0]
    fixed_tile_id = fixed_tile_path.stem
    
    visualization.generate_tile_visualization_from_file(
        preprocessor, 
        fixed_tile_id, 
        fixed_tile_path, 
        mask_file, 
        vis_dir, 
        suffix="_fixed"
    )

    # Choose random tile for second visualization
    ugs_tile_path = random.choice(tiles)
    ugs_tile_id = ugs_tile_path.stem
    
    visualization.generate_tile_visualization_from_file(
        preprocessor, 
        ugs_tile_id, 
        ugs_tile_path, 
        mask_file, 
        vis_dir, 
        suffix="_ugs"
    )

def test_preprocessing(data_dir, output_dir, vis_dir=None):
    # Make temp_data_dir accessible to visualize_tiles
    global temp_data_dir
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    temp_data_dir, vis_dir = setup_directories(output_dir, vis_dir)
    
    tiles_dir = utils.ensure_dir(temp_data_dir / "tiles")
    masks_dir = utils.ensure_dir(temp_data_dir / "masks")
    
    sat_files = list(data_dir.glob("Image_16bit_BGRNir/*.tiff"))
    if not sat_files:
        raise ValueError(f"No satellite images found in {data_dir}/Image_16bit_BGRNir")
    
    # Always use the first image file for consistent testing
    img_files = [sat_files[0]]
    print(f"Using satellite image: {img_files[0].name}")
    
    original_files = copy_sample_files(data_dir, temp_data_dir, img_files)
    if not original_files:
        raise ValueError("No valid files with annotations were found")
    
    # Create a preprocessor that saves tiles to the temp directory
    class TestPreprocessor(DataPreprocessor):
        def __init__(self):
            # Custom initializer that doesn't create split directories
            self.data_dir = temp_data_dir
            self.output_dir = temp_data_dir
            self.processed_files = set()
            self.processed_files_path = self.output_dir / "processed_files.json"
        
        def process(self):
            img_files = self.get_image_files()
            if not img_files:
                print("No image files found!")
                return {}
                
            # Take just the first file
            img_file = img_files[0]
            
            # Process the image directly without using parent class methods
            try:
                img = self.read_image(img_file)
                if img is None:
                    raise ValueError(f"Failed to read image: {img_file}")
                    
                mask_file = self.get_mask_file(img_file)
                if not mask_file.exists():
                    raise ValueError(f"Mask file not found for {img_file.name}")
                    
                multi_class_mask = self.create_multi_class_mask(mask_file)
                
                multi_class_mask = self._handle_dimension_mismatch(multi_class_mask, img, img_file.name)
                
                self.validate_data(img, multi_class_mask, img_file.name)
                
                img_with_indices = self.stack_channels(img)
                    
                tile_ids = self.create_tiles(img_with_indices, multi_class_mask, img_file.name)
                
                print(f"Created {len(tile_ids)} test tiles from {img_file.name}")
                
            except Exception as e:
                print(f"Error processing {img_file.name}: {str(e)}")
            
            return {"success": True}
        
        def create_tiles(self, img, multi_class_mask, filename):
            """
            Create two tiles from an image: one fixed and one containing UGS
            """
            tile_size = config.TILE_SIZE
            
            base_filename = Path(filename).stem
            
            tiles_created = []
            
            _, height, width = img.shape
            
            # Create a fixed tile - always at top-left corner (0,0)
            fixed_i, fixed_j = 0, 0
            
            # Extract tile from fixed position
            fixed_tile = img[:, fixed_i:fixed_i+tile_size, fixed_j:fixed_j+tile_size]
            fixed_mask_tile = multi_class_mask[fixed_i:fixed_i+tile_size, fixed_j:fixed_j+tile_size]
            
            # Generate a tile ID
            fixed_tile_id = f"{base_filename}_{fixed_i}_{fixed_j}"
            
            # Save the fixed tile
            fixed_tile_path = tiles_dir / f"{fixed_tile_id}.npy"
            fixed_mask_path = masks_dir / f"{fixed_tile_id}.npy"
            np.save(fixed_tile_path, fixed_tile)
            np.save(fixed_mask_path, fixed_mask_tile)
            
            tiles_created.append(fixed_tile_id)
            
            # Get second UGS tile
            step_size = tile_size // 2
            ugs_tiles = []
            
            # Search for one that contains both green space and urban areas
            for i in range(0, height - tile_size + 1, step_size):
                for j in range(0, width - tile_size + 1, step_size):
                    # Skip the fixed tile position
                    if i == fixed_i and j == fixed_j:
                        continue
                        
                    mask_tile = multi_class_mask[i:i+tile_size, j:j+tile_size]

                    has_green_space = np.any(mask_tile == config.GREEN_SPACE_INDEX)
                    has_urban = np.any(mask_tile == config.URBAN_INDEX)
                    
                    if has_green_space and has_urban:
                        green_space_ratio = np.mean(mask_tile == config.GREEN_SPACE_INDEX)
                        urban_ratio = np.mean(mask_tile == config.URBAN_INDEX)
                        
                        # Calculate score - higher for balanced tiles
                        balance_score = 1.0 - abs(green_space_ratio - urban_ratio)
                        content_score = green_space_ratio + urban_ratio
                        score = balance_score * content_score
                        
                        ugs_tiles.append({
                            'position': (i, j),
                            'score': score
                        })
            
            # Select the best UGS tile
            if ugs_tiles:
                ugs_tiles.sort(key=lambda x: x['score'], reverse=True)
                
                best_tile = ugs_tiles[0]
                ugs_i, ugs_j = best_tile['position']
                
                print(f"Found UGS test tile at position ({ugs_i}, {ugs_j}) with score {best_tile['score']:.2f}")
            else:
                # Use a random position if no UGS tiles found
                print(f"No UGS tiles found, using random position")
                ugs_i = random.randint(0, max(0, height - tile_size - 1))
                ugs_j = random.randint(0, max(0, width - tile_size - 1))
            
            # Extract tile from chosen position
            ugs_tile = img[:, ugs_i:ugs_i+tile_size, ugs_j:ugs_j+tile_size]
            ugs_mask_tile = multi_class_mask[ugs_i:ugs_i+tile_size, ugs_j:ugs_j+tile_size]
            
            # Generate tile ID
            ugs_tile_id = f"{base_filename}_{ugs_i}_{ugs_j}"
            
            # Save UGS tile
            ugs_tile_path = tiles_dir / f"{ugs_tile_id}.npy"
            ugs_mask_path = masks_dir / f"{ugs_tile_id}.npy"
            np.save(ugs_tile_path, ugs_tile)
            np.save(ugs_mask_path, ugs_mask_tile)
            
            tiles_created.append(ugs_tile_id)
            
            return tiles_created
    
    preprocessor = TestPreprocessor()
    preprocessor.process()
    
    # Create visualizations
    for img_file, mask_file in original_files:
        visualize_tiles(preprocessor, img_file, mask_file, vis_dir)
        visualization.generate_full_image_visualization(preprocessor, img_file, mask_file, vis_dir)
    
    shutil.rmtree(temp_data_dir)
    
    print(f"\nTest preprocessing completed. Visualizations saved to {vis_dir}")