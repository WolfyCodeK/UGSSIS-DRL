from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re
import src.config as config
import matplotlib.patches as mpatches
import cv2
from collections import Counter
from src.utils import ensure_dir, get_mask_path, setup_logger
import logging
import random
from src.dataset import UGSDataset
from tqdm import tqdm
import torch
import math

logger, _ = setup_logger()

def normalize_16bit(img):
    img = img.astype(np.float32)
    
    if img.max() > config.UINT8_MAX:
        return img / float(config.UINT16_MAX)
    
    return img / float(config.UINT8_MAX)

def apply_contrast_enhancement(img):
    # Apply percentile contrast enhancement to an image
    p_low_val = np.percentile(img, 2)
    p_high_val = np.percentile(img, 98)
    
    if p_high_val > p_low_val:
        return np.clip((img - p_low_val) / (p_high_val - p_low_val), 0, 1)
    
    return img

def create_natural_rgb(img):
    # BGR -> RGB
    rgb_img = np.stack([img[2], img[1], img[0]], axis=2)
    
    rgb_img = normalize_16bit(rgb_img)
    rgb_img = apply_contrast_enhancement(rgb_img)
    
    natural_rgb = rgb_img.copy()
    
    # Apply color correction to counteract blue tint in satellite images
    natural_rgb[:,:,0] *= config.RED_CORRECTION  
    natural_rgb[:,:,1] *= config.GREEN_CORRECTION 
    natural_rgb[:,:,2] *= config.BLUE_CORRECTION
    
    # Normalise
    max_val = np.max(natural_rgb)
    if max_val > 0:
        natural_rgb = natural_rgb / max_val
    
    natural_rgb = np.clip(natural_rgb, 0, 1)
    
    # Apply gamma correction
    natural_rgb = np.power(natural_rgb, 1/config.GAMMA_CORRECTION)
    
    return natural_rgb

def create_multi_class_mask_rgb(multi_class_mask):
    # Create RGB visualization
    mask_rgb = np.zeros((*multi_class_mask.shape, 3), dtype=np.float32)
    
    for cls, col in enumerate(config.MASK_COLORS.items()):
        mask_rgb[multi_class_mask == cls] = col[1]
    
    return mask_rgb

def create_mask_legend():
    return [
        mpatches.Patch(color=config.MASK_COLORS['background'], label='Background'),
        mpatches.Patch(color=config.MASK_COLORS['green_space'], label='Green Space'),
        mpatches.Patch(color=config.MASK_COLORS['urban'], label='Urban')
    ]

def create_mask_overlay(natural_rgb, multi_class_mask):
    alpha = 0.4
    overlay_img = natural_rgb.copy()
    
    # Create color overlays
    green_space_overlay = np.zeros_like(overlay_img)
    green_space_overlay[multi_class_mask == config.GREEN_SPACE_INDEX] = config.MASK_COLORS['green_space']
    
    urban_overlay = np.zeros_like(overlay_img)
    urban_overlay[multi_class_mask == config.URBAN_INDEX] = config.MASK_COLORS['urban']
    
    # Apply alpha blending
    overlay_img = overlay_img * (1 - alpha * (multi_class_mask > 0).astype(float)[:,:,np.newaxis])
    overlay_img += alpha * green_space_overlay + alpha * urban_overlay
    
    overlay_img = np.clip(overlay_img, 0, 1)
    
    has_green_space = np.any(multi_class_mask == config.GREEN_SPACE_INDEX)
    has_urban = np.any(multi_class_mask == config.URBAN_INDEX)
    
    has_ugs = has_green_space and has_urban
    
    return overlay_img, has_ugs

def render_channel_panels(img, multi_class_mask, augmented_img=None, augmented_mask=None, fig=None, axs=None):
    # Create figure and axes if not provided
    if fig is None or axs is None:
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        axs = axs.flatten()
    
    if augmented_img is not None:
        display_img = augmented_img
        display_mask = augmented_mask
    else:
        display_img = img
        display_mask = multi_class_mask
    
    # RGB
    natural_rgb = create_natural_rgb(display_img)
    axs[0].imshow(natural_rgb)
    axs[0].set_title('RGB (Augmented)' if augmented_img is not None else 'RGB (Natural Colors)', fontsize=10)
    axs[0].axis('off')
    
    # NIR
    nir = display_img[3]
    nir_normalized = normalize_16bit(nir)
    nir_enhanced = apply_contrast_enhancement(nir_normalized)
    nir_img = axs[1].imshow(nir_enhanced, cmap='inferno')
    axs[1].set_title('Near Infrared (NIR)' + (' (Augmented)' if augmented_img is not None else ''), fontsize=10)
    axs[1].axis('off')
    plt.colorbar(nir_img, ax=axs[1], fraction=0.046, pad=0.04)
    
    # EVI
    evi = display_img[4]
    evi_min = np.percentile(evi, 5)
    evi_max = np.percentile(evi, 95)
    
    evi_normalized = (evi - evi_min) / max(evi_max - evi_min, 1e-5)
    evi_normalized = np.clip(evi_normalized, 0, 1)
    evi_img = axs[2].imshow(evi_normalized, cmap='RdYlGn')
    axs[2].set_title('Enhanced Vegetation Index (EVI)' + (' (Augmented)' if augmented_img is not None else ''), fontsize=10)
    
    axs[2].axis('off')
    plt.colorbar(evi_img, ax=axs[2], fraction=0.046, pad=0.04)
    
    # Multi-class mask
    mask_rgb = create_multi_class_mask_rgb(display_mask)
    axs[3].imshow(mask_rgb)
    axs[3].set_title('Multi-class Mask' + (' (Augmented)' if augmented_mask is not None else ''), fontsize=10)
    axs[3].axis('off')
    
    legend_elements = create_mask_legend()
    axs[3].legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    overlay_img, has_ugs = create_mask_overlay(natural_rgb, display_mask)
    
    title = 'RGB with Mask Overlay'
    
    if has_ugs:
        title += " (Potential UGS)"
    if augmented_img is not None:
        title += " (Augmented)"
    
    axs[4].imshow(overlay_img)
    axs[4].set_title(title, fontsize=10)
    axs[4].axis('off')
    
    # Create the class distribution visualisation panel
    if display_mask is None:
        return fig, axs
    height, width = display_mask.shape
    mask_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    class_colors = {
        0: config.CLASS_INFO[0]["rgb"], # Background
        1: config.CLASS_INFO[9]["rgb"], # Green Space
        2: config.CLASS_INFO[12]["rgb"] # Urban
    }
    
    for class_id, color in class_colors.items():
        mask_img[display_mask == class_id] = color
    
    counter = Counter(display_mask.flatten())
    
    total_pixels = height * width
    percentages = {class_id: (count / total_pixels) * 100 for class_id, count in counter.items()}
    
    legend_elements = []
    class_names = {0: 'Background', 1: 'Green Space', 2: 'Urban'}
    
    for class_id in sorted(counter.keys()):
        if class_id in class_names:
            color = tuple(c/config.UINT8_MAX for c in class_colors[class_id])
            legend_elements.append(
                mpatches.Patch(
                    color=color,
                    label=f"{class_names[class_id]} ({percentages[class_id]:.1f}%)"
                )
            )
    
    # Original RGB
    if augmented_img is not None and len(axs) > 6:
        original_rgb = create_natural_rgb(img)
        axs[6].imshow(original_rgb)
        axs[6].set_title('Original RGB (Natural Colors)', fontsize=10)
        axs[6].axis('off')
        
        # Hide any unused panels
        for i in range(7, len(axs)):
            axs[i].axis('off')
            axs[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig, axs

def create_class_distribution_legend(counter, class_colors, class_names):
    legend_elements = []
    
    total_pixels = sum(counter.values())
    
    # Create a legend entry for each class
    for class_id in sorted(counter.keys()):
        
        if class_id in class_names:
            color = class_colors[class_id]
            
            # Normalise to 0-1 range
            if isinstance(color[0], int):
                color = tuple(c/config.UINT8_MAX for c in color)
                
            percentage = (counter[class_id] / total_pixels) * 100 if total_pixels > 0 else 0
            
            legend_elements.append(
                mpatches.Patch(
                    color=color,
                    label=f"{class_names[class_id]} ({percentage:.1f}%)"
                )
            )
    
    return legend_elements

def render_tile_with_original_mask(img, multi_class_mask, original_mask_file, tile_id=None, save_path=None, augmented_img=None, augmented_mask=None):
    """
    Visualize a tile with both the simplified 3-class mask and the original full 24-class mask
    """
    if tile_id is not None:
        tile_coords = extract_tile_coords_from_id(tile_id)
        
        if tile_coords is None:
            raise ValueError(f"Could not extract coordinates from tile ID: {tile_id}")
            
    fig, axs = render_channel_panels(img, multi_class_mask, augmented_img=augmented_img, augmented_mask=augmented_mask)
    
    # Replace 6th panel with original 24-class mask tile
    tile_mask, counter = extract_tile_from_mask(original_mask_file, tile_coords)
    axs[5].clear()
    axs[5].imshow(tile_mask)
    axs[5].set_title('Original 24-Class Mask', fontsize=10)
    axs[5].axis('off')
    
    # Create legend for top classes
    top_classes = counter.most_common(3)
    
    if top_classes:
        class_colors = {class_id: config.get_rgb_color(class_id) for class_id, _ in top_classes}
        class_names = {class_id: config.get_class_name(class_id) for class_id, _ in top_classes}
        
        legend_elements = create_class_distribution_legend(
            Counter({class_id: count for class_id, count in top_classes}),
            class_colors,
            class_names
        )
        
        axs[5].legend(
            handles=legend_elements, 
            loc='lower right', 
            fontsize=8, 
            title="Top Classes"
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close(fig)

def render_full_image_with_classes(img, mask_file, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    rgb = create_natural_rgb(img)
    axs[0].imshow(rgb)
    axs[0].set_title("Full RGB", fontsize=14)
    axs[0].axis('off')
    
    # Full 24-class mask
    if mask_file.suffix.lower() in ['.tif', '.tiff']:
        # Read directly with cv2
        mask = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
        if mask is None:
            logger.error(f"Failed to read mask file: {mask_file}")
            return
        # Convert to RGB color space for matplotlib
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
    else:
        # Read and convert to colormap
        mask_index = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        if mask_index is None:
            logger.error(f"Failed to read mask file: {mask_file}")
            return
        
        mask_height, mask_width = mask_index.shape
        mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        
        for class_id in np.unique(mask_index):
            color = config.get_rgb_color(class_id)
            mask[mask_index == class_id] = color
    
    # Get class distribution for legend
    counter = get_mask_class_counter(mask_file)
    total_pixels = mask.shape[0] * mask.shape[1]
    top_classes = counter.most_common(5)
    
    class_colors = {}
    class_names = {}
    
    for class_id, count in top_classes:
        class_colors[class_id] = config.get_rgb_color(class_id)
        class_names[class_id] = config.get_class_name(class_id)
    
    # Create legend
    legend_elements = []
    for class_id, count in top_classes:
        # Scale to 0-1 for matplotlib
        color = tuple(c/config.UINT8_MAX for c in class_colors[class_id])  
        name = class_names[class_id]
        
        percentage = (count / total_pixels) * 100
        
        legend_elements.append(
            mpatches.Patch(
                color=color,
                label=f"{name}: {percentage:.1f}%"
            )
        )
    
    axs[1].imshow(mask)
    axs[1].set_title("Original 24-Class Mask", fontsize=14)
    axs[1].legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=10,
        title="Top 5 Classes"
    )
    axs[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close(fig)

def get_mask_class_counter(mask_file, mask_img=None):
    if mask_img is None:

        if mask_file.suffix.lower() in ['.tif', '.tiff']:
            # Color mask - use cv2 to read
            mask_img = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
            if mask_img is None:
                return Counter()
            # Convert to RGB color space for matplotlib
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        else:
            # Read index mask
            mask_data = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            if mask_data is None:
                return Counter()
            return Counter(mask_data.flatten())

    flat_mask = mask_img.reshape(-1, 3)
    
    # Convert colors to class indices
    color_ids = flat_mask[:,0].astype(np.int32) * config.COLOR_R_MULTIPLIER + flat_mask[:,1].astype(np.int32) * config.COLOR_G_MULTIPLIER + flat_mask[:,2].astype(np.int32) * config.COLOR_B_MULTIPLIER
    
    # Map color IDs to class labels
    color_to_label = {}
    for class_id in config.CLASS_INFO.keys():
        color = config.get_rgb_color(class_id)
        color_id = color[0] * config.COLOR_R_MULTIPLIER + color[1] * config.COLOR_G_MULTIPLIER + color[2] * config.COLOR_B_MULTIPLIER
        color_to_label[color_id] = class_id
    
    counter = Counter()
    for color_id in color_ids:
        if color_id in color_to_label:
            counter[color_to_label[color_id]] += 1
            
    return counter

def extract_tile_from_mask(original_mask_file, tile_coords):
    tile_size = config.TILE_SIZE
        
    i, j = tile_coords
    
    if original_mask_file.suffix.lower() in ['.tif', '.tiff']:
        # Color mask - use cv2 to read
        full_mask_img = cv2.imread(str(original_mask_file), cv2.IMREAD_COLOR)
        if full_mask_img is None:
            return np.zeros((tile_size, tile_size, 3), dtype=np.uint8), Counter()
        # Convert to RGB color space for matplotlib
        full_mask_img = cv2.cvtColor(full_mask_img, cv2.COLOR_BGR2RGB)
        
        # Extract the tile region
        if i+tile_size <= full_mask_img.shape[0] and j+tile_size <= full_mask_img.shape[1]:
            tile_mask = full_mask_img[i:i+tile_size, j:j+tile_size]
        else:
            # Create a blank tile if coordinates are out of bounds
            tile_mask = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # Handle dimension mismatches
        if tile_mask.shape[:2] != (tile_size, tile_size):
            tile_mask = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        counter = get_mask_class_counter(original_mask_file, tile_mask)
    else:
        # Read index mask
        full_mask = cv2.imread(str(original_mask_file), cv2.IMREAD_UNCHANGED)
        if full_mask is None:
            return np.zeros((tile_size, tile_size, 3), dtype=np.uint8), Counter()
        full_mask_img = np.zeros((full_mask.shape[0], full_mask.shape[1], 3), dtype=np.uint8)
        
        # Create color visualization from index mask
        for class_id in config.CLASS_INFO.keys():
            color = config.get_rgb_color(class_id)
            full_mask_img[full_mask == class_id] = color
        
        # Extract the tile regions safely
        if i+tile_size <= full_mask.shape[0] and j+tile_size <= full_mask.shape[1]:
            tile_mask_index = full_mask[i:i+tile_size, j:j+tile_size]
        else:
            tile_mask_index = np.zeros((tile_size, tile_size), dtype=np.uint8)
        
        if i+tile_size <= full_mask_img.shape[0] and j+tile_size <= full_mask_img.shape[1]:
            tile_mask = full_mask_img[i:i+tile_size, j:j+tile_size]
        else:
            tile_mask = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # Check dimension and create blank if necessary
        if tile_mask.shape[:2] != (tile_size, tile_size):
            tile_mask = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tile_mask_index = np.zeros((tile_size, tile_size), dtype=np.uint8)
            
        counter = Counter(tile_mask_index.flatten())
    
    # Remove background class
    if 0 in counter:
        del counter[0]
        
    return tile_mask, counter

def extract_tile_coords_from_id(tile_id):
    # Match pattern with image_name_i_j where i and j are coordinates
    coords_match = re.search(r'_(\d+)_(\d+)$', tile_id)
    
    if coords_match:
        return int(coords_match.group(1)), int(coords_match.group(2))
    else:
        print(f"Warning: Could not extract coordinates from tile ID: {tile_id}")
        return None

def generate_tile_visualization_from_file(preprocessor, tile_id, tile_path, mask_file, vis_dir, suffix):
    try:
        tile = np.load(tile_path)
        
        mask_path = get_mask_path(tile_path)
        
        if not mask_path.exists():
            print(f"Mask file not found: {mask_path}")
            return None
            
        mask = np.load(mask_path)
        
    except Exception as e:
        print(f"Error loading tile/mask for {tile_id}: {str(e)}")
        return None
        
    tile_vis_path = vis_dir / f"{tile_id}{suffix}_tile.png"
    augmented_tile, augmented_mask = preprocessor.augment_tile(tile.copy(), mask.copy())
    
    render_tile_with_original_mask(
        tile, 
        mask, 
        mask_file, 
        tile_id=tile_id,
        save_path=tile_vis_path,
        augmented_img=augmented_tile,
        augmented_mask=augmented_mask
    )
    
    print(f"Created visualization for tile {tile_id}{suffix}")
    
    return tile_vis_path

def generate_full_image_visualization(preprocessor, img_file, mask_file, vis_dir):
    img = preprocessor.read_image(img_file)
    
    if img is None:
        print(f"Failed to read image {img_file}")
        return None
        
    vis_save_path = vis_dir / f"{img_file.stem}_full_classes.png"
    render_full_image_with_classes(img, mask_file, save_path=vis_save_path)
    
    print(f"Created visualization for {img_file.name}")
    
    return vis_save_path

def generate_tile_visualization_from_arrays(tile_data, mask_data, tile_id, output_dir, suffix):
    ensure_dir(output_dir)
    
    output_base = f"{tile_id}{suffix}"
    output_file = output_dir / f"{output_base}.png"
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten() 

    render_channel_panels(tile_data, mask_data, fig=fig, axs=axs)
    
    fig.suptitle(f"Tile: {tile_id}{suffix}", fontsize=14)
    
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    plt.savefig(output_file)
    plt.close(fig)
    
    print(f"Created visualization: {output_file}")
    
    return output_file

def preview_datasets():
    num_samples = 5
    
    logger.info(f"Loading datasets for preview (showing {num_samples} samples per set)...")

    try:
        train_dataset_preview = UGSDataset(split='train', transform=None)
        val_dataset_preview = UGSDataset(split='val', transform=None)
        test_dataset_preview = UGSDataset(split='test', transform=None)
        
    except FileNotFoundError as e:
        logger.error(f"Failed to load datasets for preview: {e}")
        logger.error("Please ensure preprocessing has been run.")
        return
    
    except Exception as e:
        logger.error(f"An error occurred loading datasets: {e}")
        return

    if len(train_dataset_preview) > 0:
        display_dataset_preview(train_dataset_preview, f"Training Set ({len(train_dataset_preview)} tiles)", num_samples)
    else:
        logger.warning("Training dataset is empty, skipping preview.")

    if len(val_dataset_preview) > 0:
        display_dataset_preview(val_dataset_preview, f"Validation Set ({len(val_dataset_preview)} tiles)", num_samples)
    else:
        logger.warning("Validation dataset is empty, skipping preview.")

    if len(test_dataset_preview) > 0:
        display_dataset_preview(test_dataset_preview, f"Test Set ({len(test_dataset_preview)} tiles)", num_samples)
    else:
        logger.warning("Test dataset is empty, skipping preview.")

def display_dataset_preview(dataset, dataset_name, num_samples):
    logger.info(f"Displaying {num_samples} random samples visually from: {dataset_name}")
    sample_indices_to_display = random.sample(range(len(dataset)), num_samples)

    try:
        # Switch to TkAgg backend for interactive preview
        import matplotlib
        matplotlib.use('TkAgg', force=True) 
        import matplotlib.pyplot as plt
        
        logger.debug("Switched matplotlib backend to TkAgg for preview.")
        
    except Exception as e:
        logger.warning(f"Could not switch matplotlib backend: {e}. Interactive preview might not work.")

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4.5), squeeze=False)
    axes = axes.flatten()

    # Visual preview loop
    for i, index in enumerate(sample_indices_to_display):
        try:
            sample = dataset[index]
            
            if sample is None:
                logger.warning(f"Skipping visual preview for sample index {index} in {dataset_name} due to loading error.")
                axes[i].set_title(f"Error loading index {index}")
                axes[i].axis('off')
                continue

            image_five_channel = sample['image'].numpy()
            mask_three_class = sample['mask'].numpy()
            tile_id = sample['id']

            # Calculate tile stats for displaying
            total_pixels_tile = mask_three_class.size
            
            bg_count_tile = np.sum(mask_three_class == config.UNCLASSIFIED_INDEX)
            gs_count_tile = np.sum(mask_three_class == config.GREEN_SPACE_INDEX)
            ur_count_tile = np.sum(mask_three_class == config.URBAN_INDEX)
            
            if total_pixels_tile > 0:
                bg_pct_tile = (bg_count_tile / total_pixels_tile) * 100
                gs_pct_tile = (gs_count_tile / total_pixels_tile) * 100
                ur_pct_tile = (ur_count_tile / total_pixels_tile) * 100

            rgb_image = create_natural_rgb(image_five_channel)
            
            # Create overlay
            overlay_image, _ = create_mask_overlay(rgb_image, mask_three_class)
            
            # Plot overlay instead of RGB
            ax = axes[i]
            ax.imshow(overlay_image)
            
            title_str = f"{tile_id}\nBG:{bg_pct_tile:.1f}% GS:{gs_pct_tile:.1f}% UR:{ur_pct_tile:.1f}%"
            
            ax.set_title(title_str, fontsize=8)
            ax.axis('off')
            
        except Exception as e:
            logger.error(f"Error processing visual sample index {index} from {dataset_name}: {e}", exc_info=True)
            
            axes[i].set_title(f"Error processing index {index}")
            axes[i].axis('off')

    fig.suptitle(f"Visual Preview: {dataset_name} ({num_samples} random samples)")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

    logger.info(f"Calculating overall class distribution for {dataset_name} ({len(dataset)} tiles)...")
    
    total_bg_pixels = 0
    total_gs_pixels = 0
    total_ur_pixels = 0
    total_all_pixels = 0
    calculation_errors = 0
    
    progress_bar = tqdm(range(len(dataset)), desc=f"Calculating stats for {dataset_name}", leave=False)
    
    for index in progress_bar:
        try:
            sample = dataset[index]
            
            if sample is None:
                calculation_errors += 1
                continue 

            mask_three_class = sample['mask'].numpy()
            total_all_pixels += mask_three_class.size
            
            total_bg_pixels += np.sum(mask_three_class == config.UNCLASSIFIED_INDEX)
            total_gs_pixels += np.sum(mask_three_class == config.GREEN_SPACE_INDEX)
            total_ur_pixels += np.sum(mask_three_class == config.URBAN_INDEX)
            
        except Exception as e:
            logger.warning(f"Error calculating stats for index {index} from {dataset_name}: {e}")
            calculation_errors += 1

    if calculation_errors > 0:
        logger.warning(f"Could not process {calculation_errors} samples while calculating overall stats for {dataset_name}.")

    # Calculate overall percentages
    if total_all_pixels > 0:
        overall_bg_pct = (total_bg_pixels / total_all_pixels) * 100
        overall_gs_pct = (total_gs_pixels / total_all_pixels) * 100
        overall_ur_pct = (total_ur_pixels / total_all_pixels) * 100

    logger.info(f"--- Overall Class Distribution ({dataset_name} | {len(dataset)} tiles) ---")
    logger.info(f"  Background: {overall_bg_pct:.2f}% ({total_bg_pixels:,} px)")
    logger.info(f"  Green Space: {overall_gs_pct:.2f}% ({total_gs_pixels:,} px)")
    logger.info(f"  Urban Area: {overall_ur_pct:.2f}% ({total_ur_pixels:,} px)")
    print("---")

def save_validation_prediction_binary(model, device, sample, epoch_or_cycle, output_dir, norm_stats, model_type):
    
    model.eval()
    img_unnormalized = sample['image']
    original_mask_three_class = sample.get('original_mask')
    tile_id = sample['id']

    if original_mask_three_class is None:
        logger.error(f"Cannot generate ground truth visualization for {tile_id}: 'original_mask' missing from sample.")
        return 
    
    if model_type == 'green':
        target_class_index = config.GREEN_SPACE_INDEX
        target_class_name = "Green Space"
    elif model_type == 'urban':
        target_class_index = config.URBAN_INDEX
        target_class_name = "Urban Space"
    else:
        logger.error(f"Unknown model_type '{model_type}' for validation visualization.")
        return

    # Prepare image for model
    mean = torch.tensor(norm_stats['mean'], dtype=img_unnormalized.dtype).view(-1, 1, 1)
    std = torch.tensor(norm_stats['std'], dtype=img_unnormalized.dtype).view(-1, 1, 1)
    
    # Avoid division by zero
    std = torch.where(std == 0, torch.tensor(1.0), std) 
    
    img_normalized = (img_unnormalized - mean) / std
    img_normalized = img_normalized.unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        logits = model(img_normalized)
        pred_mask_binary = torch.argmax(logits, dim=1).squeeze(0).cpu()

    # Convert tensors to numpy for plotting
    img_np = img_unnormalized.numpy() 
    gt_mask_binary_np = (original_mask_three_class.numpy() == target_class_index).astype(np.uint8)
    pred_mask_np = pred_mask_binary.numpy().astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    rgb_image = create_natural_rgb(img_np)
    axs[0].imshow(rgb_image)
    axs[0].set_title(f"Input Image (Tile: {tile_id})")
    axs[0].axis('off')

    # Create a model-specific colormap
    background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    if model_type == 'green':
        target_color = np.array(config.MASK_COLORS['green_space'], dtype=np.float32)
        
    elif model_type == 'urban':
        target_color = np.array(config.MASK_COLORS['urban'], dtype=np.float32)
        
    else:
        raise ValueError(f"Unknown model_type '{model_type}' for validation visualization.")
        
    gt_color_map = np.array([background_color, target_color])
    gt_mask_rgb = gt_color_map[gt_mask_binary_np]
    axs[1].imshow(gt_mask_rgb)
    axs[1].set_title(f"Ground Truth ({model_type.capitalize()})")
    axs[1].axis('off')
    
    gt_patches = [
        mpatches.Patch(color=tuple(background_color), label='Background/Other'),
        mpatches.Patch(color=tuple(target_color), label=target_class_name)
    ]
    axs[1].legend(handles=gt_patches, loc='lower right', fontsize=9)

    # Predicted binary mask
    pred_mask_rgb = gt_color_map[pred_mask_np]
    axs[2].imshow(pred_mask_rgb)
    axs[2].set_title(f"Prediction ({model_type.capitalize()})")
    axs[2].axis('off')
    
    pred_patches = [
        mpatches.Patch(color=tuple(background_color), label='Background/Other'),
        mpatches.Patch(color=tuple(target_color), label=target_class_name)
    ]
    axs[2].legend(handles=pred_patches, loc='lower right', fontsize=9)

    ensure_dir(output_dir)
    save_path = Path(output_dir) / f"val_pred_{model_type}_cycle_{epoch_or_cycle}_{tile_id}.png"
    
    plt.suptitle(f"Validation Prediction ({model_type.capitalize()}) - Cycle/Epoch {epoch_or_cycle}")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    except Exception as e:
        logger.error(f"Failed to save validation prediction plot: {e}")
        
    finally:
        plt.close(fig)

def save_selected_tile_collage(full_dataset_untransformed, selected_indices, green_selected_indices, urban_selected_indices, cycle, output_dir):
    if not selected_indices:
        logger.info(f"Cycle {cycle}: No tiles selected, skipping collage generation.")
        return

    num_tiles = len(selected_indices)
    
    # Calculate grid size
    ncols = int(math.ceil(math.sqrt(num_tiles)))
    nrows = int(math.ceil(num_tiles / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5), squeeze=False)
    axes = axes.flatten()

    collage_dir = ensure_dir(Path(output_dir) / "selected_tiles_collage")
    save_path = collage_dir / f"cycle_{cycle}_selection.png"

    logger.info(f"Generating tile collage for cycle {cycle} ({num_tiles} tiles) -> {save_path}")

    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        
        try:
            sample = full_dataset_untransformed[idx]
            if sample is None:
                ax.set_title(f"Index {idx}\nError Loading")
                ax.axis('off')
                continue

            img_np = sample['image'].numpy()
            tile_id = sample['id']
            mask_three_class = sample['mask'].numpy() 

            rgb_image = create_natural_rgb(img_np)
            
            overlay_image, _ = create_mask_overlay(rgb_image, mask_three_class)
            
            ax.imshow(overlay_image) 

            selected_by_green = idx in green_selected_indices
            selected_by_urban = idx in urban_selected_indices
            
            title = f"Idx: {idx}\n{tile_id[:15]}..."
            border_color = 'gray'
            linewidth = 1

            if selected_by_green and selected_by_urban:
                title += "\nSource: Both"
                border_color = 'purple' 
                linewidth = 3
                
            elif selected_by_green:
                title += "\nSource: Green"
                border_color = 'lime'
                linewidth = 3
                
            elif selected_by_urban:
                title += "\nSource: Urban"
                border_color = 'red'
                linewidth = 3
            else:
                title += "\nSource: Random Fill"
            
            ax.set_title(title, fontsize=7)
            
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(linewidth)
            
            ax.set_xticks([])
            ax.set_yticks([])

        except Exception as e:
            logger.error(f"Error processing tile index {idx} for collage: {e}", exc_info=True)
            
            ax.set_title(f"Index {idx}\nError Plotting")
            ax.axis('off')

    for j in range(num_tiles, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Selected Tiles - Cycle {cycle}", fontsize=16)
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    try:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    except Exception as e:
        logger.error(f"Failed to save selected tiles collage plot: {e}")
    finally:
        plt.close(fig)