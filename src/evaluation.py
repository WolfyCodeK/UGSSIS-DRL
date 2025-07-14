import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import json 
import pandas as pd 

import src.config as config
from src.dataset import UGSDataset
from src.segmentation_model import ResNetDeepLab
from src.tile_classifier_model import TileClassifier
from src.utils import convert_windows_path_to_wsl, ensure_dir, load_json, save_json, setup_logger

logger, _ = setup_logger()

def load_model(checkpoint_data, device, model_type):
    print(f"Loading {model_type} model state dict from checkpoint data...")

    model = ResNetDeepLab(
        num_classes=2,
        backbone_name=config.SEG_MODEL_BACKBONE,
        pretrained_backbone=False
    ).to(device)

    model_key = f'model_{model_type}'
    if isinstance(checkpoint_data, dict) and model_key in checkpoint_data:
        model_state_dict = checkpoint_data[model_key]
        
    else:
        fallback_key = f'model_{model_type}_state_dict'
        
        if isinstance(checkpoint_data, dict) and fallback_key in checkpoint_data:
            print(f"Found state dict under fallback key '{fallback_key}'. Consider updating checkpoint saving/loading consistency.")
            model_state_dict = checkpoint_data[fallback_key]
            
        else:
            raise KeyError(f"Could not find state dict for {model_type} in checkpoint.")
    try:
        load_result = model.load_state_dict(model_state_dict, strict=False)
        
        if load_result.missing_keys:
            print(f"Missing keys when loading {model_type} state_dict: {load_result.missing_keys}")
            
        if load_result.unexpected_keys:
            print(f"Unexpected keys when loading {model_type} state_dict: {load_result.unexpected_keys}")

    except Exception as e:
        raise RuntimeError(f"Failed to load state dict for {model_type}.")

    print(f"{model_type.capitalize()} model loaded successfully.")
    model.eval()
    return model

def predict_binary(model, dataloader, device, model_type, return_images=True):
    model.eval()
    all_preds_binary = []
    
    if return_images:
        all_images_norm = []
    else:
        all_images_norm = None
        
    all_ids = []

    print(f"Running {model_type} inference...")
    
    progress_bar = tqdm(dataloader, desc=f"Predicting ({model_type.capitalize()})", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            
            if batch is None: 
                continue

            images_norm = batch['image'].to(device)
            ids = batch['id']

            logits = model(images_norm)
            preds_binary = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds_binary.append(preds_binary)
            
            if return_images:
                all_images_norm.append(batch['image'].cpu().numpy())
                
            all_ids.extend(ids)

    print(f"{model_type.capitalize()} inference complete.")
    all_preds_binary_np = np.concatenate(all_preds_binary, axis=0)
    
    if return_images:
        all_images_norm_np = np.concatenate(all_images_norm, axis=0)
    else:
        all_images_norm_np = None
    
    return all_preds_binary_np, all_images_norm_np, all_ids

def calculate_binary_metrics(binary_preds, binary_targets, target_class=1, smooth=1e-6):
    preds_flat = (binary_preds == target_class).flatten()
    targets_flat = (binary_targets == target_class).flatten()

    tp = np.sum(preds_flat & targets_flat)
    fp = np.sum(preds_flat & ~targets_flat)
    fn = np.sum(~preds_flat & targets_flat)

    intersection = tp
    union = tp + fp + fn

    iou = (intersection + smooth) / (union + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def denormalise_image(normalised_image_tensor, mean, std):
    mean_tensor = torch.tensor(mean, dtype=normalised_image_tensor.dtype, device=normalised_image_tensor.device).view(-1, 1, 1)
    
    std_tensor = torch.tensor(std, dtype=normalised_image_tensor.dtype, device=normalised_image_tensor.device).view(-1, 1, 1)
    
    std_tensor = torch.where(std_tensor == 0, torch.tensor(1.0, device=std_tensor.device), std_tensor)
    
    denormalised_image = normalised_image_tensor * std_tensor + mean_tensor
    
    return denormalised_image

def calculate_class_percentages(mask):
    total_pixels = mask.size
    if total_pixels == 0:
        return {0: 0.0, 1: 0.0, 2: 0.0}

    background_pct = np.mean(mask == config.UNCLASSIFIED_INDEX) * 100
    green_pct = np.mean(mask == config.GREEN_SPACE_INDEX) * 100
    urban_pct = np.mean(mask == config.URBAN_INDEX) * 100
    
    return {0: background_pct, 1: green_pct, 2: urban_pct}

def evaluate_model(model_checkpoint, eval_output_dir_name="evaluation_results"):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model_checkpoint = convert_windows_path_to_wsl(model_checkpoint)
    
    checkpoint_path = Path(model_checkpoint)
    run_name = checkpoint_path.parent.name
    
    if not run_name or run_name == '.':
        run_name = checkpoint_path.stem + "_eval" 
        print(f"Using fallback run_name '{run_name}' for evaluation output.")
        
    eval_output_dir = ensure_dir(Path(eval_output_dir_name) / run_name)
    print(f"Evaluation results will be saved to: {eval_output_dir}")

    run_name_from_checkpoint = Path(model_checkpoint).parent.name
    norm_stats_path = Path(config.LOG_DIR) / run_name_from_checkpoint / "norm_stats.json"
    
    if not norm_stats_path.exists():

        norm_stats_path_fallback = Path(config.LOG_DIR) / "manual-ugs" / "norm_stats.json" 
        
        if norm_stats_path_fallback.exists():
            norm_stats_path = norm_stats_path_fallback
        else:
            # Handle error as before if neither exists
            error_msg = f"Normalisation constants file not found at {norm_stats_path} or {norm_stats_path_fallback}"
            print(error_msg)
            raise FileNotFoundError(error_msg)

    try:
        norm_stats = load_json(norm_stats_path)
        norm_mean = norm_stats['mean']
        norm_std = norm_stats['std']
        print(f"Loaded normalisation constants from {norm_stats_path}")
        
    except FileNotFoundError as e:
        print(f"Error loading normalisation constants from {norm_stats_path}: {e}")
        raise Exception(f"Error loading normalisation constants from {norm_stats_path}: {e}")
    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {norm_stats_path}: {e}")
        raise Exception(f"Error decoding JSON from {norm_stats_path}: {e}")

    labeled_indices_from_checkpoint = None
    model_green = None
    model_urban = None
    
    try:
        
        print(f"Loading full checkpoint data from: {model_checkpoint}")
        if not os.path.exists(model_checkpoint):
            print(f"Checkpoint file not found: {model_checkpoint}")
            raise FileNotFoundError(f"Checkpoint file not found: {model_checkpoint}")

        checkpoint_data = torch.load(model_checkpoint, map_location=device, weights_only=False)
        
        if isinstance(checkpoint_data, dict) and 'labeled_indices' in checkpoint_data:
            
            labeled_indices_from_checkpoint = checkpoint_data['labeled_indices']
            
            if isinstance(labeled_indices_from_checkpoint, list):
                print(f"Extracted {len(labeled_indices_from_checkpoint)} labeled indices from checkpoint.")
            else:
                print("'labeled_indices' found in checkpoint but is not a list. Ignoring.")
                labeled_indices_from_checkpoint = None
                
        else:
            print("Could not find 'labeled_indices' list in checkpoint. Tile Classifier training (if needed) will use all available labeled inputs.")
            labeled_indices_from_checkpoint = None
            
        model_green = load_model(checkpoint_data, device, model_type='green')
        model_urban = load_model(checkpoint_data, device, model_type='urban')
        
    except FileNotFoundError:
        print(f"Checkpoint file {model_checkpoint} not found.")
        return
    
    except KeyError as e:
        print(f"Error loading model state dict key from checkpoint {model_checkpoint}: {e}")
        print("Checkpoint dictionary structure might be incorrect (expecting 'model_green'/'model_urban').")
        return
    
    except Exception as e:
        print(f"An unexpected error occurred loading checkpoint {model_checkpoint}: {e}", exc_info=True)
        return

    tile_classifier_model_path = Path(config.TC_CHECKPOINT_DIR) / config.TC_SAVE_NAME 
    tile_classifier_model = None

    if tile_classifier_model_path.exists():
        print(f"Attempting to load existing Tile Classifier model from: {tile_classifier_model_path}")
        try:
            tile_classifier_model = TileClassifier().to(device)
            tile_classifier_model.load_state_dict(torch.load(tile_classifier_model_path, map_location=device))
            tile_classifier_model.eval()
            print("Existing Tile Classifier model loaded successfully. Skipping input generation and training.")
            
        except Exception as e:
            print(f"Failed to load existing tile classifier model state dict from {tile_classifier_model_path}: {e}", exc_info=True)
            print("Cannot proceed with evaluation using the faulty classifier checkpoint.")
            return
    else:
        print(f"Tile Classifier model checkpoint not found at {tile_classifier_model_path}. Generating inputs and training...")
        
        print("Generating tile classifier inputs from train set.")
        
        input_files_generated = generate_tile_classifier_inputs(
            model_green=model_green,
            model_urban=model_urban,
            device=device,
            norm_stats=norm_stats,
            labeled_indices=labeled_indices_from_checkpoint,
            output_dir=Path(config.TC_TILE_CLASSIFIER_INPUT_DIR) / 'train',
            expert_labels_file=Path(config.TC_EXPERT_LABELS) 
        )

        if input_files_generated:
            print("Training Tile Classifier model...")
        try:
            from src.train_tile_classifier import train_tile_classifier_model
                
            saved_path = train_tile_classifier_model()
                
            if saved_path and Path(saved_path).exists():
                print("Tile Classifier training complete. Loading newly trained model.")
                
                tile_classifier_model = TileClassifier().to(device)
                tile_classifier_model.load_state_dict(torch.load(saved_path, map_location=device))
                
                tile_classifier_model.eval()
                tile_classifier_model_path = saved_path
            else:
                print("Tile Classifier training failed or did not save a model. Cannot proceed with evaluation.")
                return 
            
        except Exception as e:
            print(f"An unexpected error occurred during tile classifier training: {e}", exc_info=True)
            return 

    if tile_classifier_model is None:
        print("Tile Classifier model is None after load/train attempt. Aborting.")
        return 

    print("Loading test data (raw images and GT masks)...")
    
    test_dataset_raw = UGSDataset(split='test', transform=None, target_class_index=None)
    if len(test_dataset_raw) == 0:
        print("Test dataset is empty. Cannot evaluate.")
        return

    test_loader_eval = DataLoader(test_dataset_raw, batch_size=config.SEG_BATCH_SIZE, shuffle=False, num_workers=0) 

    all_ids = []
    all_masks_3cls_gt_list = []
    all_preds_green_list = []
    all_preds_urban_list = []
    all_classifier_tile_preds_list = []
    all_images_norm_list = [] 

    print("Performing inference and collecting results batch-by-batch.")
    with torch.no_grad():
        model_green.eval()
        model_urban.eval()
        
        tile_classifier_model.eval() 
        
        progress_bar = tqdm(test_loader_eval, desc="Evaluating Test Set", total=len(test_loader_eval))
        for batch in progress_bar:
            if batch is None: continue
            
            raw_images = batch['image']
            masks_3cls_gt = batch['mask'].numpy()
            ids_batch = batch['id']
            

            all_masks_3cls_gt_list.append(masks_3cls_gt)
            all_ids.extend(ids_batch)
            
            mean_t = torch.tensor(norm_mean, dtype=raw_images.dtype, device=device).view(1, -1, 1, 1)
            std_t = torch.tensor(norm_std, dtype=raw_images.dtype, device=device).view(1, -1, 1, 1)
            
            std_t = torch.where(std_t == 0, torch.tensor(1.0, device=device), std_t)
            
            norm_images = (raw_images.to(device) - mean_t) / std_t
            
            all_images_norm_list.append(norm_images.cpu().numpy())


            logits_green = model_green(norm_images)
            
            batch_preds_green = torch.argmax(logits_green, dim=1).cpu().numpy().astype(np.uint8)
            all_preds_green_list.append(batch_preds_green)

            logits_urban = model_urban(norm_images)
            batch_preds_urban = torch.argmax(logits_urban, dim=1).cpu().numpy().astype(np.uint8)
            all_preds_urban_list.append(batch_preds_urban)

            classifier_input = torch.from_numpy(np.stack([batch_preds_green, batch_preds_urban], axis=1)).float().to(device)
            
            classifier_logits = tile_classifier_model(classifier_input) # (N, 1)
            batch_classifier_preds = (torch.sigmoid(classifier_logits) > 0.5).cpu().numpy().flatten().astype(int)
            all_classifier_tile_preds_list.extend(batch_classifier_preds)

    print("Concatenating results.")
    masks_3cls_gt_np = np.concatenate(all_masks_3cls_gt_list, axis=0)
    preds_green_np = np.concatenate(all_preds_green_list, axis=0)
    preds_urban_np = np.concatenate(all_preds_urban_list, axis=0)
    tile_level_preds_np = np.array(all_classifier_tile_preds_list)
    images_norm = np.concatenate(all_images_norm_list, axis=0)

    print("Calculating binary Green/Urban test metrics (Prediction vs GT).")
    
    gt_green_binary = (masks_3cls_gt_np == config.GREEN_SPACE_INDEX).astype(np.uint8)
    gt_urban_binary = (masks_3cls_gt_np == config.URBAN_INDEX).astype(np.uint8)
    
    metrics_green_binary = calculate_binary_metrics(preds_green_np, gt_green_binary)
    metrics_urban_binary = calculate_binary_metrics(preds_urban_np, gt_urban_binary)
    
    print(f"Binary test metrics (Green Model vs GT)")
    print(f"  IoU: {metrics_green_binary['iou']:.4f}, Precision: {metrics_green_binary['precision']:.4f}, Recall: {metrics_green_binary['recall']:.4f}, F1: {metrics_green_binary['f1']:.4f}")
    print(f"Binary test metrics (Urban Model vs GT)")
    print(f"  IoU: {metrics_urban_binary['iou']:.4f}, Precision: {metrics_urban_binary['precision']:.4f}, Recall: {metrics_urban_binary['recall']:.4f}, F1: {metrics_urban_binary['f1']:.4f}")

    binary_ugs_preds_final = np.zeros_like(preds_green_np, dtype=np.uint8) 
    print("Constructing final UGS mask based on tile classifier decisions.")
    
    for i in range(len(tile_level_preds_np)):
        if tile_level_preds_np[i] == 1:
            is_green_pred = (preds_green_np[i] == 1)
            binary_ugs_preds_final[i][is_green_pred] = 1

    print("Generating TARGET UGS masks for test set using expert labels...")
    
    expert_test_labels_path = Path('expert_test_labels.csv')
    expert_test_labels = {}
    
    if expert_test_labels_path.exists():
        try:
            df_expert_test = pd.read_csv(expert_test_labels_path)
            
            if {'tile_id', 'contains_ugs'}.issubset(df_expert_test.columns):
                df_expert_test = df_expert_test.dropna(subset=['tile_id', 'contains_ugs'])
                df_expert_test['contains_ugs'] = df_expert_test['contains_ugs'].astype(int)
                
                expert_test_labels = dict(zip(df_expert_test['tile_id'], df_expert_test['contains_ugs']))
                print(f"Loaded {len(expert_test_labels)} expert labels for test set.")
                
            else:
                print(f"Expert test labels file {expert_test_labels_path} missing required columns. Using rule-based GT instead.")
                
        except Exception as e:
            print(f"Error loading expert test labels from {expert_test_labels_path}: {e}. Using rule-based GT instead.")
            
    else:
        print(f"Expert test labels file not found: {expert_test_labels_path}. Using rule-based GT instead.")
        
    binary_ugs_masks_gt_final = np.zeros_like(gt_green_binary, dtype=np.uint8)
    
    if expert_test_labels:
        num_missing_expert = 0
        for i, tile_id in enumerate(all_ids):
            
            expert_label_val = expert_test_labels.get(tile_id)
            
            final_label = None
            if expert_label_val is not None:
                try:
                    final_label = int(expert_label_val)
                except (ValueError, TypeError):
                    print(f"Could not convert expert label '{expert_label_val}' to int for tile {tile_id}. Treating as missing.")
                    final_label = None

            if final_label == 1:
                binary_ugs_masks_gt_final[i] = gt_green_binary[i]
            elif final_label == 0:
                pass 
            else:
                num_missing_expert += 1
                pass
        if num_missing_expert > 0:
            print(f"Could not find or use expert labels for {num_missing_expert} test tiles. Ground truth for these tiles is set to 0.")
        print("Generated final ground truth UGS masks based on expert test labels.")
    else: 
        raise Exception("Failed to load expert test labels. Cannot proceed with expert-derived GT.")

    print("Calculating final binary UGS metrics (Tile Classifier Pred vs Expert-Derived GT).")
    metrics_final_ugs = calculate_binary_metrics(binary_ugs_preds_final, binary_ugs_masks_gt_final, target_class=1)

    print(f"Final evaluation metrics (Binary UGS - Tile Classifier)")
    print(f"  IoU:       {metrics_final_ugs['iou']:.4f}")
    print(f"  Precision: {metrics_final_ugs['precision']:.4f}")
    print(f"  Recall:    {metrics_final_ugs['recall']:.4f}")
    print(f"  F1-Score:  {metrics_final_ugs['f1']:.4f}")

    all_metrics = {
        "binary_green_metrics (vs GT)": metrics_green_binary,
        "binary_urban_metrics (vs GT)": metrics_urban_binary,
        "final_binary_ugs_metrics (derived)": metrics_final_ugs
    }
    metrics_path = eval_output_dir / "evaluation_metrics_dual.json"
    try:
        save_json(all_metrics, metrics_path)
        print(f"Metrics saved to {metrics_path}")
        
    except Exception as e:
        print(f"Failed to save metrics: {e}")

    print("Evaluation complete.")

def generate_tile_classifier_inputs(model_green, model_urban, device, norm_stats, labeled_indices, output_dir, expert_labels_file):
    try:
        ensure_dir(output_dir)

        if not expert_labels_file.exists():
            print(f"Expert labels file not found at {expert_labels_file}. Cannot determine relevant tiles.")
            return False

        try:
            df_expert = pd.read_csv(expert_labels_file)
            
            if not {'tile_id', 'contains_ugs'}.issubset(df_expert.columns):
                raise ValueError("Expert labels CSV must contain 'tile_id' and 'contains_ugs' columns.")
            
            expert_tile_ids = set(df_expert['tile_id'].astype(str).tolist())
            print(f"Loaded {len(expert_tile_ids)} unique tile IDs from expert labels file.")

        except Exception as e:
            print(f"Error reading or parsing expert labels file {expert_labels_file}: {e}")
            return False


        print("Loading base UGSDataset (train split) for input generation.")
        base_train_dataset = UGSDataset(split='train', transform=None)
        if not base_train_dataset:
            print("Failed to load base training dataset.")
            return False

        if labeled_indices is not None:
            print(f"Processing subset of {len(labeled_indices)} tiles based on checkpoint labeled_indices.")

            num_base_samples = len(base_train_dataset)
            valid_indices = [idx for idx in labeled_indices if 0 <= idx < num_base_samples]
            invalid_count = len(labeled_indices) - len(valid_indices)
            
            if invalid_count > 0:
                print(f"Ignoring {invalid_count} indices from checkpoint that are out of bounds for the current dataset (size {num_base_samples}).")
                
            if not valid_indices:
                print("No valid labeled indices found after bounds check. Cannot generate inputs.")
                return False
            dataset_to_process = Subset(base_train_dataset, valid_indices)
            indices_to_iterate = valid_indices
        else:
            print("Processing the *entire* 'train' split as labeled_indices were not found in checkpoint.")
            dataset_to_process = base_train_dataset
            indices_to_iterate = list(range(len(base_train_dataset)))

        input_gen_loader = DataLoader(dataset_to_process, batch_size=config.SEG_BATCH_SIZE * 4, shuffle=False, num_workers=2)

        model_green.eval()
        model_urban.eval()

        mean = torch.tensor(norm_stats['mean'], dtype=torch.float32, device=device).view(-1, 1, 1)
        std = torch.tensor(norm_stats['std'], dtype=torch.float32, device=device).view(-1, 1, 1)
        std = torch.where(std == 0, torch.tensor(1.0, device=device), std)

        num_saved = 0
        processed_ids_subset = set()

        print(f"Generating inputs for {len(dataset_to_process)} tiles.")
        progress_bar = tqdm(input_gen_loader, desc="Generating Classifier Inputs", leave=False)

        with torch.no_grad():
            batch_start_idx = 0
            for batch in progress_bar:
                if batch is None: continue

                images_unnormalised = batch['image'].to(device)
                
                current_batch_indices_in_list = indices_to_iterate[batch_start_idx : batch_start_idx + len(images_unnormalised)]
                
                batch_tile_ids = [Path(base_train_dataset.tile_files[orig_idx]).stem for orig_idx in current_batch_indices_in_list]
                
                batch_start_idx += len(images_unnormalised)

                images = (images_unnormalised - mean) / std

                logits_green = model_green(images)
                probs_green = F.softmax(logits_green, dim=1)[:, 1, :, :]

                logits_urban = model_urban(images)
                probs_urban = F.softmax(logits_urban, dim=1)[:, 1, :, :]

                stacked_probs = torch.stack([probs_green, probs_urban], dim=1).cpu().numpy()

                for i, tile_id in enumerate(batch_tile_ids):
                    if tile_id is None:
                        print(f"Skipping sample at index {current_batch_indices_in_list[i]} due to missing tile ID.")
                        continue
                        
                    if tile_id not in expert_tile_ids:
                        logger.debug(f"Skipping {tile_id} - present in dataset split but not in expert labels CSV.")
                        continue

                    if tile_id in processed_ids_subset:
                        print(f"Tile ID {tile_id} encountered again in this input generation run. Skipping duplicate save.")
                        continue
                        
                    output_file = output_dir / f"{tile_id}.npy"
                    try:
                        np.save(output_file, stacked_probs[i].astype(np.float16))
                        num_saved += 1
                        processed_ids_subset.add(tile_id)
                    except Exception as e:
                        print(f"Failed to save input for tile {tile_id} to {output_file}: {e}")

        if num_saved == 0 and len(dataset_to_process) > 0:
            print(f"Processed {len(dataset_to_process)} dataset samples but failed to save any valid inputs matching expert labels to {output_dir}.")
            print("Check expert label matching, file permissions, and intermediate processing steps.")
            
            return False
            
        print(f"Successfully saved {num_saved}/{len(processed_ids_subset)} tile classifier input files to {output_dir}.")
        print("Tile classifier input generation finished")
        return True

    except Exception as e:
        print(f"An unexpected error occurred during tile classifier input generation: {e}", exc_info=True)
        
        return False