import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import src.config as config
import glob
import pandas as pd

class UGSDataset(Dataset):
    def __init__(self, split='train', transform=None, target_class_index=None):
        self.output_dir = Path(config.OUTPUT_DIR)
        self.split = split
        self.target_class_index = target_class_index
        self.transform = transform

        self.tiles_dir = self.output_dir / split / "tiles"
        self.masks_dir = self.output_dir / split / "masks"

        if not self.tiles_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found for split '{split}' at {self.output_dir / split}")

        self.tile_files = sorted(glob.glob(str(self.tiles_dir / '*.npy')))

        if not self.tile_files:
            print(f"No .npy files found in {self.tiles_dir}")
        else:
            print(f"Found {len(self.tile_files)} tiles for split '{split}' (Target Class: {self.target_class_index})")

    def __len__(self):
        return len(self.tile_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tile_path = Path(self.tile_files[idx])
        tile_id = tile_path.stem
        mask_path = self.masks_dir / f"{tile_id}.npy"

        try:
            tile = np.load(tile_path)
            mask_three_class = np.load(mask_path)

            original_mask_np = mask_three_class.astype(np.uint8)

            if self.target_class_index is not None:
                mask = (mask_three_class == self.target_class_index).astype(np.uint8)
            else:
                mask = mask_three_class.astype(np.uint8)

        except Exception as e:
            print(f"Error loading data for tile_id {tile_id} (idx {idx}): {e}", exc_info=True)
            return None
        
        tile_tensor = torch.from_numpy(tile).float()
        mask_tensor = torch.from_numpy(mask).long()

        sample = {'image': tile_tensor, 'mask': mask_tensor, 'id': tile_id, 'index': idx}

        if self.transform is None:
            sample['original_mask'] = torch.from_numpy(original_mask_np).long()

        if self.transform:
            sample = self.transform(sample)
            if 'index' not in sample:
                sample['index'] = idx
            if 'id' not in sample:
                sample['id'] = tile_id

        return sample
    
class TileClassifierDataset(Dataset):
    def __init__(
        self, 
        split='train', 
        input_dir='tile_classifier_inputs', 
        expert_labels_file='expert_train_labels.csv',
        allowed_tile_ids=None, 
        transform=None
    ):
        self.input_base_dir = Path(input_dir)
        self.split = split
        self.data_dir = self.input_base_dir / split
        self.expert_labels_file = Path(expert_labels_file)
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Tile Classifier input directory not found: {self.data_dir}")
        
        if not self.expert_labels_file.exists():
            raise FileNotFoundError(f"Expert labels file not found: {self.expert_labels_file}")
        
        try:
            df_expert = pd.read_csv(self.expert_labels_file)
            
            if not {'tile_id', 'contains_ugs'}.issubset(df_expert.columns):
                raise ValueError("Expert labels CSV must contain 'tile_id' and 'contains_ugs' columns.")
            
            df_expert = df_expert.dropna(subset=['tile_id', 'contains_ugs'])
            
            df_expert['contains_ugs'] = df_expert['contains_ugs'].astype(int)
            self.expert_labels = dict(zip(df_expert['tile_id'], df_expert['contains_ugs']))
            
            print(f"Loaded {len(self.expert_labels)} expert tile labels from {self.expert_labels_file}")
            
        except Exception as e:
            print(f"Error loading expert labels from {self.expert_labels_file}: {e}")
            raise
        
        self.input_files = []
        
        all_npy_files = sorted(glob.glob(str(self.data_dir / '*.npy')))
        
        num_skipped_allowed = 0
        num_skipped_expert = 0
        
        for f_path_str in all_npy_files:
            tile_id = Path(f_path_str).stem
            
            is_allowed = allowed_tile_ids is None or tile_id in allowed_tile_ids
            has_expert_label = tile_id in self.expert_labels
            
            if is_allowed and has_expert_label:
                self.input_files.append(f_path_str)
                
            elif not is_allowed:
                num_skipped_allowed += 1
                print(f"Skipping {tile_id} - not in allowed_tile_ids set.")
                
            elif not has_expert_label:
                num_skipped_expert += 1
                print(f"Skipping {tile_id} - not found in expert labels file.")
                
        print(f"Skipped {num_skipped_allowed} files not in allowed set (from checkpoint labeled_indices)." if allowed_tile_ids is not None else "Allowed ID filter not provided.")
        
        print(f"Skipped {num_skipped_expert} files missing from expert labels CSV.")
        
        if not self.input_files:
            print(f"No input .npy files found in {self.data_dir} corresponding to expert labels and allowed IDs.")
            
        else:
            print(f"Found {len(self.input_files)} input files for split '{split}' matching expert labels and allowed IDs.")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = Path(self.input_files[idx])
        tile_id = input_path.stem

        try:
            input_stack = np.load(input_path).astype(np.float32)

            expert_label = self.expert_labels.get(tile_id)
            if expert_label is None:
                print(f"Missing expert label for {tile_id} during getitem!")
                return None 
            
            target_label = float(expert_label)

        except Exception as e:
            print(f"Error loading or processing data for tile_id {tile_id} (idx {idx}): {e}", exc_info=True)
            return None

        input_tensor = torch.from_numpy(input_stack)
        target_tensor = torch.tensor(target_label, dtype=torch.float32)

        sample = {'input': input_tensor, 'label': target_tensor, 'id': tile_id}

        if self.transform:
            transformed_input = self.transform(sample['input']) 
            sample['input'] = transformed_input
            
            if 'label' not in sample: 
                sample['label'] = target_tensor
                
            if 'id' not in sample: 
                sample['id'] = tile_id

        return sample