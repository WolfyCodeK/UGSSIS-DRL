import numpy as np
from pathlib import Path
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
import re
import src.config as config
import logging

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger, handler

def ensure_dir(directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_mask_path(tile_path):
    return Path(str(tile_path).replace("/tiles/", "/masks/").replace("\\tiles\\", "\\masks\\"))

def get_evi(img):
    if torch.is_tensor(img):
        blue = img[0].float()
        red = img[2].float()
        nir = img[3].float()
        
        img_max_val = torch.max(img[:4].reshape(4, -1), dim=1).values.max()
        
        if img_max_val > 1.0:
            
            if img_max_val > config.UINT16_MAX:
                scale_factor = float(config.UINT16_MAX)
            else:
                scale_factor = float(config.UINT8_MAX)
                
            blue = blue / scale_factor
            red = red / scale_factor
            nir = nir / scale_factor
        
        denominator = nir + 6 * red - 7.5 * blue + 1.0
        
        evi = torch.zeros_like(nir)
        
        # Calculate EVI only for valid pixels
        valid_pixels = torch.abs(denominator) > 1e-6
        evi[valid_pixels] = 2.5 * (nir[valid_pixels] - red[valid_pixels]) / denominator[valid_pixels]
        
        evi = torch.clamp(evi, -1.0, 1.0)
        
    else: 
        # Input is NumPy array
        blue = img[0].astype(np.float32)
        red = img[2].astype(np.float32)
        nir = img[3].astype(np.float32)
        
        # Determine max value for scaling
        img_max_val = img[:4].max()
        
        if img_max_val > 1.0:
            
            if img_max_val > config.UINT16_MAX:
                scale_factor = float(config.UINT16_MAX)
            else:
                scale_factor = float(config.UINT8_MAX)
                
            blue = blue / scale_factor
            red = red / scale_factor
            nir = nir / scale_factor
        
        denominator = nir + 6 * red - 7.5 * blue + 1.0
    
        evi = np.zeros_like(nir)
        
        # Calculate EVI only for valid pixels
        valid_pixels = np.abs(denominator) > 1e-6
        evi[valid_pixels] = 2.5 * (nir[valid_pixels] - red[valid_pixels]) / denominator[valid_pixels]
        
        evi = np.clip(evi, -1.0, 1.0)
    
    return evi

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(input_path):
    with open(input_path, 'r') as f:
        return json.load(f)

def calculate_class_weights(dataset, num_classes):
    # Calculate class weights based on the pixel frequency in the dataset
    print(f"Calculating class weights for {num_classes} classes.")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    progress_bar = tqdm(loader, desc="Counting Class Pixels", leave=False)
    
    for batch in progress_bar:
        # Skip empty batches
        if batch is None: continue
        
        masks = batch['mask']
        masks_np = masks.cpu().numpy()
        
        # Count occurrences of each class index
        unique, counts = np.unique(masks_np, return_counts=True)
        
        # Add counts to the total, handling cases where a class might not be in a batch
        for index, count in zip(unique, counts):

            if 0 <= index < num_classes:
                class_counts[index] += count
                
        total_pixels += masks.numel() # Total pixels in the batch

    if total_pixels == 0:
        print("No pixels found in dataset to calculate class weights.")
        return None
        
    if np.any(class_counts == 0):
        print(f"One or more classes have zero pixels in the dataset: {class_counts}")
        # Avoid division by zero - replace 0 count with 1
        class_counts[class_counts == 0] = 1 

    # Calculate inverse frequency weights to give weight to rarer classes
    frequencies = class_counts / total_pixels
    median_freq = np.median(frequencies[frequencies > 0])
    weights = median_freq / frequencies
    
    weights_tensor = torch.from_numpy(weights).float()
    
    print(f"Calculated Class Counts: {class_counts}")
    print(f"Calculated Class Weights: {weights_tensor.tolist()}")
    
    return weights_tensor

def calculate_iou_binary(predicted_logits, targets):
    # Calculate IoU for the positive class in binary segmentation
    predicted_classes = torch.argmax(predicted_logits, dim=1)
    targets = targets.long().to(predicted_classes.device)

    # Binary segmentation so 1 is always the positive class
    predicted_indicies = (predicted_classes == 1)
    target_indicies = (targets == 1)

    intersection = (predicted_indicies & target_indicies).sum().float()
    union = (predicted_indicies | target_indicies).sum().float()

    # Handle division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.item()

def convert_windows_path_to_wsl(path_str):
    is_wsl = 'WSL_DISTRO_NAME' in os.environ or 'WSL_INTEROP' in os.environ

    if not is_wsl or not path_str:
        return path_str

    windows_path_match = re.match(r"^([a-zA-Z]):[\\/](.*)", path_str)
    
    if windows_path_match:
        
        drive_letter = windows_path_match.group(1).lower()
        
        rest_of_path = windows_path_match.group(2).replace("\\", "/")
        
        wsl_path = f"/mnt/{drive_letter}/{rest_of_path}"
        
        print(f"Converted Windows path '{path_str}' to WSL path '{wsl_path}'")
        
        return wsl_path
    else:
        return path_str

def parse_tile_id(tile_id):
    try:
        parts = tile_id.split('_tile_')
        
        if len(parts) != 2:
            raise ValueError("Tile ID format incorrect: missing '_tile_'")
        
        base_filename = parts[0]
        coords_part = parts[1]
        
        coord_parts = coords_part.split('_')
        
        if len(coord_parts) != 2:
            raise ValueError("Tile ID format incorrect: coordinate part invalid")
            
        y_coord = int(coord_parts[0])
        x_coord = int(coord_parts[1])
        
        return base_filename, y_coord, x_coord
    
    except Exception as e:
        print(f"Failed to parse tile ID '{tile_id}': {e}")
        return None, None, None

def calculate_entropy_from_logits(logits):
    if logits is None or logits.numel() == 0:
        return torch.tensor([])
        
    probs = torch.softmax(logits, dim=1)
    
    # Calculate entropy per pixel: H = - sum(p * log(p + ε))
    # ε (small value) added to prevent log(0)
    entropy_per_pixel = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    
    avg_entropy_per_image = torch.mean(entropy_per_pixel, dim=[1, 2])
    
    return avg_entropy_per_image
