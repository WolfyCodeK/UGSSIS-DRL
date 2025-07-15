import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
import src.utils as utils
import src.config as config

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

class DataPreprocessor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        for split in config.SPLITS:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            (split_dir / "tiles").mkdir(exist_ok=True)
            (split_dir / "masks").mkdir(exist_ok=True)
        
        self.processed_files_path = self.output_dir / "processed_files.json"
        self.processed_files = set()
        
        if self.processed_files_path.exists():
            print(f"Loading list of already processed files from {self.processed_files_path}")
            self.processed_files = set(utils.load_json(self.processed_files_path))
            print(f"Found {len(self.processed_files)} already processed files")
        
        self._save_processed_files() 
        
        self.split_counts = {split: 0 for split in config.SPLITS}
        self._initialise_split_counts()
    
    def _initialise_split_counts(self):
        for split in config.SPLITS:
            split_dir = self.output_dir / split
            tiles_dir = split_dir / "tiles"
            
            if tiles_dir.exists():
                self.split_counts[split] = len(list(tiles_dir.glob("*.npy")))
        
        split_counts_str = ", ".join([f"{split.capitalize()}: {count}" for split, count in self.split_counts.items()])
        
        print(f"Current split counts: {split_counts_str}")
    
    def create_multi_class_mask(self, mask_file):
        try:
            if mask_file.suffix.lower() in ['.tif', '.tiff']:
                mask_img = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
                if mask_img is None:
                    print(f"Error: Failed to read color mask: {mask_file}")
                    return None
                    
                mask = self._color_to_label(mask_img)
            else:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    print(f"Error: Failed to read index mask: {mask_file}")
                    return None
                    
            multi_class_mask = np.zeros_like(mask, dtype=np.uint8)
            
            for green_space_class in config.GREEN_SPACE_CLASSES:
                multi_class_mask[mask == green_space_class] = config.GREEN_SPACE_INDEX
                
            for urban_class in config.URBAN_CLASSES:
                multi_class_mask[mask == urban_class] = config.URBAN_INDEX
                
            return multi_class_mask
        except Exception as e:
            print(f"Error creating multi-class mask for {mask_file}: {str(e)}")
            return None
    
    def get_image_files(self):
        img_files = []
        
        img_dir_16bit = self.data_dir / "Image_16bit_BGRNir"
        
        if img_dir_16bit.exists():
            img_files = list(img_dir_16bit.glob("*.tiff"))
            print(f"Found {len(img_files)} 16-bit images in {img_dir_16bit}")
            
        else:
            print(f"Contents of data directory ({self.data_dir}):")
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    print(f"- {item.name}/ ({len(list(item.glob('*')))} files)")
                else:
                    print(f"- {item.name}")
            return []
        
        img_files = [f for f in img_files if str(f) not in self.processed_files]
        
        return img_files
    
    def create_tiles(self, img, multi_class_mask, filename):
        tile_size = config.TILE_SIZE
        step_size = config.TILE_SIZE 
        _, height, width = img.shape
        base_filename = Path(filename).stem
        saved_tile_ids = []
        
        for y in range(0, height - tile_size + 1, step_size):
            for x in range(0, width - tile_size + 1, step_size):
                img_tile = img[:, y:y + tile_size, x:x + tile_size]
                mask_tile = multi_class_mask[y:y + tile_size, x:x + tile_size]
                
                total_pixels = mask_tile.size
                if total_pixels == 0: continue
                
                green_pixels = np.sum(mask_tile == config.GREEN_SPACE_INDEX)
                urban_pixels = np.sum(mask_tile == config.URBAN_INDEX)
                
                green_ratio = green_pixels / total_pixels
                urban_ratio = urban_pixels / total_pixels
                
                has_sufficient_content = (green_ratio > config.MIN_MAJORITY_CONTENT_RATIO) or (urban_ratio > config.MIN_MAJORITY_CONTENT_RATIO)

                if has_sufficient_content:
                    tile_id = f"{base_filename}_tile_{y}_{x}"
                    
                    assigned_split = self._assign_split()
                    
                    save_dir = self.output_dir / assigned_split
                    tile_save_path = save_dir / "tiles" / f"{tile_id}.npy"
                    mask_save_path = save_dir / "masks" / f"{tile_id}.npy"
                    
                    np.save(tile_save_path, img_tile.astype(np.float32)) 
                    np.save(mask_save_path, mask_tile.astype(np.uint8))
                    
                    saved_tile_ids.append(tile_id)
                    self.split_counts[assigned_split] += 1
                    
        return saved_tile_ids 
    
    def validate_data(self, img, mask, filename):
        if img is None or mask is None:
            print(f"Error: Empty data for {filename}")
            return False
        
        img_height, img_width = img.shape[1:] if img.ndim > 2 else img.shape
        mask_height, mask_width = mask.shape
        
        if img_height != mask_height or img_width != mask_width:
            height_diff = abs(img_height - mask_height)
            width_diff = abs(img_width - mask_width)
            
            if height_diff <= 2 and width_diff <= 2:
                print(f"Warning: Small dimension mismatch for {filename}. Image: ({img.shape}), Mask: ({mask.shape})")
                return True
            else:
                print(f"Error: Dimension mismatch for {filename}. Image: ({img.shape}), Mask: ({mask.shape})")
                return False
            
        return True
    
    def _handle_dimension_mismatch(self, multi_class_mask, img, filename):
        if multi_class_mask is None:
            print(f"Error: Mask is None for {filename}")
            return None
            
        if img.shape[1:] != multi_class_mask.shape:
            img_height, img_width = img.shape[1], img.shape[2]
            mask_height, mask_width = multi_class_mask.shape
            
            height_diff = abs(img_height - mask_height)
            width_diff = abs(img_width - mask_width)
            
            if height_diff <= 2 and width_diff <= 2:
                resised_mask = cv2.resize(multi_class_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                return resised_mask
            else:
                print(f"Error: Dimension mismatch for {filename}. Image: ({img.shape}), Mask: ({multi_class_mask.shape})")
                return None
                
        return multi_class_mask
    
    def _initialise_processing_stats(self):
        stats = {
            'total_images': 0,
            'total_ugs_tiles': 0
        }

        for split in config.SPLITS:
            stats[split] = 0
            
        return stats
    
    def _process_single_image(self, img_file, stats):
        try:
            if str(img_file) in self.processed_files:
                print(f"Skipping already processed file: {img_file.name}")
                return False
                
            img = self.read_image(img_file)
            if img is None:
                raise ValueError(f"Failed to read image: {img_file}")
                
            mask_file = self.get_mask_file(img_file)
            if not mask_file.exists():
                raise ValueError(f"Mask file not found for {img_file.name}")
                
            multi_class_mask = self.create_multi_class_mask(mask_file)
            multi_class_mask = self._handle_dimension_mismatch(multi_class_mask, img, img_file.name)
            
            if multi_class_mask is None:
                raise ValueError(f"Mask processing failed for {img_file.name}.")
            
            self.validate_data(img, multi_class_mask, img_file.name) 
            
            evi = utils.get_evi(img)
            
            img_with_evi = np.zeros((5, img.shape[1], img.shape[2]), dtype=np.float32)
            img_with_evi[:4] = img.astype(np.float32)
            img_with_evi[4] = evi.numpy().astype(np.float32)
                
            tile_ids = self.create_tiles(img_with_evi, multi_class_mask, img_file.name)
            
            self.processed_files.add(str(img_file))
            self._save_processed_files()
            
            self._update_split_counts(tile_ids)
            self._update_run_statistics(tile_ids, stats) 
            
            return True
            
        except Exception as e:
            print(f"Error processing {img_file.name}: {str(e)}")
            return False

    def _update_split_counts(self, new_tile_ids):
        for tile_id in new_tile_ids:
            found = False
            for split in config.SPLITS:
                tile_path = self.output_dir / split / "tiles" / f"{tile_id}.npy"
                mask_path = self.output_dir / split / "masks" / f"{tile_id}.npy"
                if tile_path.exists() or mask_path.exists():
                    found = True
                    break
            if not found:
                print(f"Warning: Could not determine split for newly created tile {tile_id}")

    def _update_run_statistics(self, new_tile_ids, stats):
        new_tiles_count = len(new_tile_ids)
        stats['total_ugs_tiles'] += new_tiles_count 

    def _print_processing_summary(self, stats):
        print("\nProcessing run summary")
        print(f"Images processed in this run: {stats['total_images']}")
        print(f"Tiles created in this run: {stats['total_ugs_tiles']}") 
        print("\nOverall dataset distribution")
        
        total_tiles_overall = sum(self.split_counts.values())
        print(f"Total tiles across all runs: {total_tiles_overall}")
        
        if total_tiles_overall > 0:
            for split in config.SPLITS:
                count = self.split_counts[split]
                ratio = count / total_tiles_overall * 100
                print(f"  {split.capitalize()}: {count} tiles ({ratio:.1f}%)")
        else:
            print("No tiles found in the output directory.")

    def process(self):
        self.sample_tiles = {}
        
        img_files = self.get_image_files()
        
        print(f"Processing {len(img_files)} images.")
        
        existing_stats = self._calculate_existing_stats()
        
        stats = self._initialise_processing_stats()
        
        for _, img_file in enumerate(tqdm(img_files)):
            self._process_single_image(img_file, stats)
                
        combined_stats = self._combine_stats(stats, existing_stats)
                
        self._print_processing_summary(combined_stats)
        
        self._save_processed_files()
    
        return combined_stats
        
    def _calculate_existing_stats(self):
        existing_stats = {
            'total_images': 0,
            'total_ugs_tiles': 0
        }
        
        total_tile_count = 0
        
        for split in config.SPLITS:
            tiles_dir = self.output_dir / split / "tiles"
            
            if not tiles_dir.exists():
                existing_stats[split] = 0
                continue
                
            tile_count = len(list(tiles_dir.glob("*.npy")))
            existing_stats[split] = tile_count
            total_tile_count += tile_count
        
        existing_stats['total_ugs_tiles'] = total_tile_count
        existing_stats['total_images'] = len(self.processed_files)
        
        return existing_stats
        
    def _combine_stats(self, new_stats, existing_stats):
        combined_stats = {
            'total_images': new_stats['total_images'] + existing_stats['total_images'],
            'total_ugs_tiles': new_stats['total_ugs_tiles'] + existing_stats['total_ugs_tiles']
        }
        
        for split in config.SPLITS:
            combined_stats[split] = new_stats[split] + existing_stats[split]
        
        return combined_stats
        
    def _color_to_label(self, color_mask):
        height, width, _ = color_mask.shape
        label_mask = np.zeros((height, width), dtype=np.uint8)
        
        color_mask_id = color_mask[:,:,0].astype(np.int32) * config.COLOR_R_MULTIPLIER + color_mask[:,:,1].astype(np.int32) * config.COLOR_G_MULTIPLIER + color_mask[:,:,2].astype(np.int32) * config.COLOR_B_MULTIPLIER
        
        color_id_to_label = {}
        
        for class_id in config.CLASS_INFO.keys():
            color = config.get_bgr_color(class_id)
            color_id = color[0] * config.COLOR_R_MULTIPLIER + color[1] * config.COLOR_G_MULTIPLIER + color[2] * config.COLOR_B_MULTIPLIER
            color_id_to_label[color_id] = class_id
        
        unique_color_ids = np.unique(color_mask_id)
        
        for color_id in unique_color_ids:
            if color_id in color_id_to_label:
                label = color_id_to_label[color_id]
                label_mask[color_mask_id == color_id] = label
        
        return label_mask
    
    def get_mask_file(self, img_file):
        img_stem = img_file.stem

        mask_path = self.data_dir / "Annotation__index" / f"{img_stem}_24label.png"
        
        if mask_path.exists():
            return mask_path
        else:
            print(f"Index mask file not found: {mask_path}")
            return mask_path
    
    def read_image(self, img_file):
        try:
            with rasterio.open(str(img_file)) as src:
                img = src.read()
                return img
            
        except Exception as e:
            print(f"Error reading {img_file}: {e}")
            return None
        
    def _save_processed_files(self):
        utils.save_json(list(self.processed_files), self.processed_files_path)

    def _assign_split(self):
        total_tiles = sum(self.split_counts.values())
        
        current_ratios = {split: count / total_tiles if total_tiles > 0 else 0 
                        for split, count in self.split_counts.items()}
        
        target_ratios = dict(zip(config.SPLITS, config.SPLIT_RATIO))
        
        split_deficit = {split: target_ratios[split] - current_ratios[split] 
                        for split in config.SPLITS}
        
        assigned_split = max(split_deficit.keys(), key=lambda x: split_deficit[x])
        
        return assigned_split