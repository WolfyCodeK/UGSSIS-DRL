#!/usr/bin/env python
"""
Interactive Tile Labeling Script for UGS Presence

Displays tiles (RGB + 3-Class GT Mask) one by one and allows the user
to label them at the tile level (Yes/No/Skip) for UGS presence using key presses.

Called via `python main.py --label`.
Labels the 'train' split and saves to 'expert_tile_labels.csv'.
"""

import matplotlib
matplotlib.use('TkAgg') # Use a backend that supports interactive events
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import logging
import sys
import torch
from tqdm import tqdm
from src.utils import convert_windows_path_to_wsl

# Ensure src is in path for imports if running script directly
# Keep this logic in case script is run directly for debugging
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.dataset import UGSDataset
from src.visualization import create_natural_rgb, create_multi_class_mask_rgb, create_mask_legend

# --- Global State ---
dataset = None
labelable_indices = [] # Indices of tiles meeting the criteria (contain both green and urban)
all_tile_ids_map = {} # Map: dataset_index -> tile_id (for all tiles)
labels = {} # {tile_id: contains_ugs (1/0)} # Stores ALL labels (manual Y/N, automatic N)
current_labelable_list_pos = 0 # Position in the labelable_indices list
output_csv_path = Path('expert_tile_labels.csv')
labelable_indices_path = Path('labelable_indices.json') # Path for the marker file
manual_labels = {} # Labels loaded directly from the file
fig = None
axs = None

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(split='train', checkpoint_path_str=None, output_csv_path_str='expert_tile_labels.csv'):
    """Loads dataset, filters based on checkpoint (for train) AND GT content, loads/infers labels."""
    # Ensure assignment updates the global map for later use
    global dataset, labelable_indices, all_tile_ids_map, current_labelable_list_pos, labels, manual_labels 
    output_csv_path = Path(output_csv_path_str)
    is_train_split = (split == 'train')
    
    # --- 1. Load Checkpoint (Only if split is 'train') --- 
    labeled_indices_from_checkpoint = None
    checkpoint_path = None
    if is_train_split:
        if checkpoint_path_str is None:
            logger.error("Checkpoint path is required for labeling the 'train' split.")
            return False
        checkpoint_path = Path(convert_windows_path_to_wsl(checkpoint_path_str))
        logger.info(f"Loading checkpoint for 'train' split filtering: {checkpoint_path}")
        try:
            if not checkpoint_path.exists():
                 logger.error(f"Checkpoint file not found: {checkpoint_path}")
                 raise FileNotFoundError
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint_data, dict) and 'labeled_indices' in checkpoint_data:
                labeled_indices_from_checkpoint = checkpoint_data['labeled_indices']
                logger.info(f"Extracted {len(labeled_indices_from_checkpoint)} labeled indices from checkpoint.")
                if not labeled_indices_from_checkpoint:
                     logger.warning("Checkpoint 'labeled_indices' list is empty for train split.")
            else:
                logger.error("Could not find 'labeled_indices' key in checkpoint. Cannot proceed with train split labeling.")
                return False
        except FileNotFoundError:
             logger.error(f"Processed data directory for split '{split}' not found.")
             return False
        except Exception as e:
             logger.error(f"Error loading checkpoint {checkpoint_path}: {e}", exc_info=True)
             return False
    else:
         logger.info(f"Running labeler for '{split}' split. Checkpoint indices will not be used for filtering.")

    # --- 2. Load Full Dataset --- 
    logger.info(f"Loading full dataset for split: {split}...")
    try:
        dataset = UGSDataset(split=split, transform=None, target_class_index=None)
        if len(dataset) == 0:
            logger.error(f"Dataset split '{split}' is empty. Cannot label.")
            return False
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        return False

    # --- 3. Load Existing Manual/Previous Labels --- 
    manual_labels = {}
    if output_csv_path.exists():
        logger.info(f"Loading existing label file: {output_csv_path}...")
        try:
            existing_df = pd.read_csv(output_csv_path)
            if {'tile_id', 'contains_ugs'}.issubset(existing_df.columns):
                existing_df = existing_df.dropna(subset=['tile_id', 'contains_ugs'])
                existing_df['contains_ugs'] = existing_df['contains_ugs'].astype(int)
                manual_labels = dict(zip(existing_df['tile_id'], existing_df['contains_ugs']))
                logger.info(f"Loaded {len(manual_labels)} labels from file.")
            else:
                logger.warning(f"CSV file {output_csv_path} missing required columns. Ignoring.")
        except Exception as e:
            logger.error(f"Error loading or parsing {output_csv_path}: {e}. Ignoring existing file.")
    else:
        logger.info(f"No existing label file found at {output_csv_path}. Starting fresh.")

    # --- 4. Filter Indices by Checkpoint (if train) & GT Content & Build Maps --- 
    logger.info(f"Filtering tiles for split '{split}' based on GT content (Green & Urban)... ")
    # Clear the global map before repopulating locally
    all_tile_ids_map.clear() 
    current_all_tile_ids_map = {} 
    allowed_tile_ids = set() 
    indices_to_check = labeled_indices_from_checkpoint if is_train_split else list(range(len(dataset)))
    
    if is_train_split and labeled_indices_from_checkpoint is not None:
         logger.info(f"Also filtering based on {len(labeled_indices_from_checkpoint)} indices from checkpoint.")
    elif not is_train_split:
         logger.info("Filtering based only on GT content for this split.")
    
    # Build the full index->ID map first
    for i in range(len(dataset)):
         current_all_tile_ids_map[i] = Path(dataset.tile_files[i]).stem
         
    # Determine the set of allowed IDs (all for test, checkpoint-filtered for train)
    if is_train_split and labeled_indices_from_checkpoint is not None:
        allowed_tile_ids = {current_all_tile_ids_map.get(idx) 
                             for idx in labeled_indices_from_checkpoint 
                             if 0 <= idx < len(dataset) and current_all_tile_ids_map.get(idx) is not None}
    else: # For test split, all tiles are initially allowed
         allowed_tile_ids = set(current_all_tile_ids_map.values())

    # Iterate through allowed indices and check GT content
    current_labelable_indices = [] # Initialize the list here
    progress_bar = tqdm(indices_to_check, desc=f"Filtering {split} Tiles")
    num_skipped_out_of_bounds = 0
    # Temporarily store indices with both green and urban
    for index in progress_bar:
        if not (0 <= index < len(dataset)): 
            num_skipped_out_of_bounds += 1
            continue # Skip invalid indices from checkpoint
        
        # Check GT content
        sample = dataset[index]
        if sample is None: continue
        mask = sample['mask']
        unique_classes = torch.unique(mask)
        if config.GREEN_SPACE_INDEX in unique_classes and config.URBAN_INDEX in unique_classes:
            current_labelable_indices.append(index)
            
    if num_skipped_out_of_bounds > 0:
        logger.warning(f"Skipped {num_skipped_out_of_bounds} indices from checkpoint that were out of bounds.")

    # --- 4.1. Apply Urban Ratio Filter (New Step) ---
    logger.info(f"Applying Urban Ratio filter (Threshold: {config.MIN_MAJORITY_CONTENT_RATIO:.1%})...")
    final_labelable_indices = []
    auto_labeled_low_urban = {} # Track tiles auto-labeled here
    progress_bar_ratio = tqdm(current_labelable_indices, desc=f"Checking Urban Ratio")
    for index in progress_bar_ratio:
        try:
            sample = dataset[index]
            if sample is None: continue
            mask = sample['mask'].numpy() # Need numpy for calculation
            total_pixels = mask.size
            if total_pixels == 0: continue
            
            urban_pixels = np.sum(mask == config.URBAN_INDEX)
            urban_ratio = urban_pixels / total_pixels

            tile_id = current_all_tile_ids_map.get(index)
            if tile_id is None: continue

            if urban_ratio < config.MIN_MAJORITY_CONTENT_RATIO:
                # Auto-label as 0 (No UGS) and skip manual labeling
                auto_labeled_low_urban[tile_id] = 0 
            else:
                # Keep this index for potential manual labeling
                final_labelable_indices.append(index)
        except Exception as e:
             logger.error(f"Error calculating urban ratio for index {index}: {e}", exc_info=True)
             # Optionally skip this tile if ratio calculation fails

    labelable_indices = final_labelable_indices # This is the list the user will see
    logger.info(f"Found {len(labelable_indices)} tiles remaining for potential manual labeling after Urban Ratio filter.")
    if auto_labeled_low_urban:
        logger.info(f"Automatically labeled {len(auto_labeled_low_urban)} tiles as 'No UGS' due to low urban content.")

    # --- 5. Create Comprehensive Labels Dictionary --- 
    final_labels = {} 
    # Start with labels from file
    final_labels.update(manual_labels)
    # Add automatic labels from the low-urban filter (will overwrite manual if conflicts, which is okay)
    final_labels.update(auto_labeled_low_urban)
    
    # Add automatic labels for tiles that were allowed by checkpoint/split but didn't meet *any* filter criteria (original green/urban or new urban ratio)
    num_auto_labeled_other = 0
    for index, tile_id in all_tile_ids_map.items():
        if tile_id in allowed_tile_ids and tile_id not in final_labels:
             # Check if it was excluded by original green/urban filter OR the urban ratio filter
             is_in_final_labelable = index in labelable_indices # labelable_indices now only contains those passing urban ratio
             if not is_in_final_labelable:
                final_labels[tile_id] = 0
                num_auto_labeled_other += 1

    labels = final_labels
    if num_auto_labeled_other > 0:
        logger.info(f"Additionally auto-labeled {num_auto_labeled_other} tiles as 'No UGS' (e.g., missing Green/Urban GT, or failed ratio check). These won't be shown.")

    # --- Update Global Map Explicitly ---
    all_tile_ids_map.update(current_all_tile_ids_map) # Update the global dict

    # --- 6. Determine Resume Position --- 
    current_labelable_list_pos = 0
    # Check against MANUALLY loaded labels to determine where to resume interaction
    labeled_ids_set = set(manual_labels.keys()) 
    first_unlabeled_pos = -1
    for list_pos, dataset_idx in enumerate(labelable_indices):
        tile_id = all_tile_ids_map.get(dataset_idx)
        if tile_id and tile_id not in labeled_ids_set:
            first_unlabeled_pos = list_pos
            break
    if first_unlabeled_pos != -1:
        current_labelable_list_pos = first_unlabeled_pos
        resume_tile_id = all_tile_ids_map.get(labelable_indices[current_labelable_list_pos])
        logger.info(f"Resuming manual labeling from position {current_labelable_list_pos} in the labelable list (Tile ID: {resume_tile_id}) | Total to label in '{split}': {len(labelable_indices)}")
    else:
        # If first_unlabeled_pos is still -1, it means all tiles in labelable_indices
        # were found in manual_labels. Start review from beginning.
        logger.info(f"All labelable tiles for split '{split}' already have manual labels. Starting review from beginning.")
        current_labelable_list_pos = 0
        # Don't set the all_labeled flag here
        # all_labeled = True 
            
    # Return success status only
    return True # Removed all_labeled flag

def display_tile():
    """Displays the current tile's RGB and GT mask."""
    global current_labelable_list_pos, fig, axs, labelable_indices, all_tile_ids_map, dataset, labels, manual_labels

    if not (0 <= current_labelable_list_pos < len(labelable_indices)):
        logger.info("Reached end of labelable tiles list or invalid list position.")
        if fig:
             plt.close(fig) # Close window if we are out of bounds
        return

    current_dataset_index = labelable_indices[current_labelable_list_pos]
    # Get tile_id using the map
    tile_id = all_tile_ids_map.get(current_dataset_index, "UNKNOWN_ID")
    logger.debug(f"Displaying labelable list position {current_labelable_list_pos}, dataset index {current_dataset_index}, tile: {tile_id}")

    if dataset is None:
        logger.error("Dataset is not loaded. Cannot display tile.")
        return
        
    try:
        sample = dataset[current_dataset_index]
        if sample is None:
            logger.warning(f"Could not load data for tile {tile_id} (index {current_dataset_index}). Skipping.")
            current_labelable_list_pos += 1
            display_tile()
            return

        img_tensor = sample['image']
        mask_3cls_tensor = sample['mask']
        img_np = img_tensor.numpy()
        mask_3cls_np = mask_3cls_tensor.numpy()
        rgb_image = create_natural_rgb(img_np)
        mask_rgb = create_multi_class_mask_rgb(mask_3cls_np)

        if fig is None or axs is None or not plt.fignum_exists(fig.number):
             logger.info("Recreating plot window.")
             fig, axs = plt.subplots(1, 2, figsize=(10, 5))
             fig.canvas.mpl_connect('key_press_event', on_key_press)

        axs[0].clear()
        axs[1].clear()
        axs[0].imshow(rgb_image)
        axs[0].set_title("Input RGB")
        axs[0].axis('off')
        axs[1].imshow(mask_rgb)
        axs[1].set_title("Ground Truth (3-Class)")
        axs[1].axis('off')
        legend_elements = create_mask_legend()
        axs[1].legend(handles=legend_elements, loc='lower right', fontsize=8)

        label_status = "Unknown"
        if tile_id in labels:
            label_value = labels[tile_id]
            is_manual = tile_id in manual_labels # Check if it came from the file
            label_source = "Manual" if is_manual else "Auto"
            label_text = "Yes" if label_value == 1 else "No"
            label_status = f"Labeled: {label_text} ({label_source})"
        else:
            label_status = "Not Labeled"
            
        total_labelable = len(labelable_indices)
        fig.suptitle(f"Tile {current_labelable_list_pos + 1}/{total_labelable}: {tile_id} (Dataset Index: {current_dataset_index})\nLabel: '{label_status}'\n[Y] Yes (Contains UGS) | [N] No (No UGS) | [S] Skip | [Q] Quit & Save", fontsize=10)
        fig.canvas.draw_idle()

    except Exception as e:
        logger.error(f"Error displaying tile {tile_id} (index {current_dataset_index}): {e}", exc_info=True)

def save_labels(output_csv_path_str='expert_tile_labels.csv'):
    """Saves the collected labels to the specified CSV file."""
    global labels
    output_csv_path = Path(output_csv_path_str) # Use passed path
    if not labels:
        logger.info("No labels collected/generated to save.") # Modified message
        return

    logger.info(f"Saving {len(labels)} total labels (manual + automatic) to {output_csv_path}...")
    try:
        label_list = [{'tile_id': tid, 'contains_ugs': lbl} for tid, lbl in labels.items()]
        df = pd.DataFrame(label_list)
        df = df[['tile_id', 'contains_ugs']]
        df.to_csv(output_csv_path, index=False)
        logger.info("Labels saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save labels to {output_csv_path}: {e}")

def on_key_press(event):
    """Handles key press events for labeling."""
    global current_labelable_list_pos, labels, labelable_indices, all_tile_ids_map, output_csv_path_arg

    if not (0 <= current_labelable_list_pos < len(labelable_indices)):
        logger.warning("Key press ignored: Index out of bounds.")
        return
        
    current_dataset_index = labelable_indices[current_labelable_list_pos]
    # Get tile_id using the map
    tile_id = all_tile_ids_map.get(current_dataset_index, "UNKNOWN_ID")
    if tile_id == "UNKNOWN_ID":
         logger.error("Cannot find tile ID for current index. Aborting key press.")
         return

    key = event.key.lower()
    proceed_to_next = False

    if key == 'y':
        logger.info(f"Labeled '{tile_id}' as YES (Contains UGS)")
        labels[tile_id] = 1
        proceed_to_next = True
    elif key == 'n':
        logger.info(f"Labeled '{tile_id}' as NO (No UGS)")
        labels[tile_id] = 0
        proceed_to_next = True
    elif key == 's':
        logger.info(f"Skipped '{tile_id}'")
        # Remove if previously labeled (allows correcting skips/autos)
        if tile_id in labels:
            logger.debug(f"Removing previous label for skipped tile '{tile_id}'")
            del labels[tile_id]
        proceed_to_next = True
    elif key == 'q':
        logger.info("Quit requested.")
        save_labels(output_csv_path_arg)
        plt.close(event.canvas.figure)
        return
    else:
        return # Ignore other keys
        
    if proceed_to_next:
        current_labelable_list_pos += 1
        if current_labelable_list_pos < len(labelable_indices):
            display_tile()
        else:
            logger.info("Finished all labelable tiles!")
            save_labels(output_csv_path_arg)
            plt.close(event.canvas.figure)

def run_labeler(checkpoint_path=None, split='train', output='expert_tile_labels.csv'):
    """Main function to run the interactive labeler. Requires checkpoint path."""
    global fig, axs, output_csv_path_arg # Store output path for keypress handler
    
    # Add check if called without checkpoint (e.g., from main.py)
    if checkpoint_path is None:
         logger.error("run_labeler called without checkpoint_path. Please use standalone execution with --checkpoint or modify main.py integration.")
         # Alternatively, prompt user here, or use a hardcoded default path
         # For now, just exit.
         return
         
    output_csv_path_arg = output # Store for on_key_press

    # Pass split and output path to load_data, only capture success status
    load_success = load_data(split=split, 
                             checkpoint_path_str=checkpoint_path, 
                             output_csv_path_str=output)
    
    if not load_success:
        return # Exit if data loading failed

    # Check if there are actually tiles to label AFTER loading/filtering
    if not labelable_indices:
        logger.info(f"No tiles to label in split '{split}' based on checkpoint and GT content filters.")
        save_labels(output) # Save potentially generated automatic labels for this split
        return
        
    # If we have labelable indices, proceed to interactive session
    logger.info(f"Presenting {len(labelable_indices)} tiles from split '{split}' for manual labeling (starting at index {current_labelable_list_pos})...")

    # Setup matplotlib figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # Display the first tile
    display_tile()

    logger.info("Starting interactive labeling. Press keys in the plot window:")
    logger.info("  [Y] Yes (Contains UGS) | [N] No (No UGS) | [S] Skip Tile | [Q] Quit & Save")
    plt.show() # Start the event loop, blocks until window is closed

    logger.info("Labeling session ended.")