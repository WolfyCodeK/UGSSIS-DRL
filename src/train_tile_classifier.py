import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import sys
from sklearn.metrics import f1_score, accuracy_score

# Make sure src is in path if running script directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.dataset import TileClassifierDataset 
from src.tile_classifier_model import TileClassifier
from src.utils import ensure_dir

def train_tile_classifier_epoch(model, dataloader, criterion, optimiser, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Classifier Train Epoch", leave=False)
    
    for batch in progress_bar:
        if batch is None: continue
        
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        
        optimiser.zero_grad()
        logits = model(inputs)
        
        loss = criterion(logits, labels) 
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    if num_batches > 0:
        avg_loss = total_loss / num_batches
    else:
        avg_loss = 0.0
    
    return avg_loss

def validate_tile_classifier_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Classifier Validation", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            if batch is None: continue
            
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            batch_size = inputs.shape[0]
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * batch_size
            
            probs = torch.sigmoid(logits)
            threshold = 0.5
            preds = (probs > threshold).cpu().numpy().astype(int)
            
            all_preds.extend(preds.flatten()) 
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))
            num_samples += batch_size
            
    if num_samples > 0:
        avg_loss = total_loss / num_samples
    else:
        avg_loss = 0.0
    
    if num_samples > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division="warn")
    else:
        accuracy = 0.0
        f1 = 0.0
            
    return avg_loss, accuracy, f1

def train_tile_classifier_model():
    run_name = f"tile_classifier_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = ensure_dir(Path(config.TC_LOG_DIR) / run_name)
    checkpoint_dir = ensure_dir(Path(config.TC_CHECKPOINT_DIR))
    save_path = checkpoint_dir / config.TC_SAVE_NAME
    
    print(f"Starting tile classifier training")
    print(f"Logs: {log_dir}")
    print(f"Saving final model to: {save_path}")
    
    writer = SummaryWriter(log_dir=str(log_dir))

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading TileClassifierDataset.")
    
    try:
        full_dataset = TileClassifierDataset(
            split='train', 
            input_dir=config.TC_TILE_CLASSIFIER_INPUT_DIR, 
            expert_labels_file=config.TC_EXPERT_LABELS,
            allowed_tile_ids=None
        )
        
    except Exception as e:
        print(f"Failed to load dataset for tile classifier training: {e}")
        writer.close()
        return None
        
    if len(full_dataset) == 0:
        print("TileClassifierDataset is empty. Aborting training.")
        writer.close()
        return None

    val_size = int(config.TC_VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    if val_size < 1 or train_size < 1:
        print(f"Dataset size ({len(full_dataset)}) too small for validation split ({config.TC_VAL_SPLIT}). Training without validation.")
        train_dataset = full_dataset
        val_dataset = None
        
    else:
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Split dataset: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    print("Calculating target label distribution...")

    try:
        all_labels = []
        
        for i in range(len(full_dataset)):
            sample = full_dataset[i]

            if sample is None:
                continue
            
            all_labels.append(sample['label'].item()) 
        
        label_counts = np.bincount(np.array(all_labels).astype(int))
        
        if len(label_counts) < 2:
            label_counts = np.pad(label_counts, (0, 2 - len(label_counts))) 
        
        print(f"Dataset Label Distribution: Class 0 (Non UGS): {label_counts[0]}, Class 1 (UGS): {label_counts[1]}")
        
        if label_counts[1] == 0:
            print("Training data contains no positive (UGS) examples.")
            
    except Exception as e:
        print(f"Could not calculate label distribution: {e}")
        
    criterion = nn.BCEWithLogitsLoss().to(device) 

    train_loader = DataLoader(train_dataset, batch_size=config.TC_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.TC_BATCH_SIZE * 2, shuffle=False, num_workers=2) if val_dataset else None

    print("Initialising Tile Classifier Model.")
    model = TileClassifier().to(device)
    optimiser = optim.Adam(model.parameters(), lr=config.TC_LR)

    print(f"Starting training for {config.TC_EPOCHS} epochs...")
    best_val_f1 = -1.0
    start_time = time.time()

    for epoch in range(config.TC_EPOCHS):
        epoch_start_time = time.time()
        train_loss = train_tile_classifier_epoch(model, train_loader, criterion, optimiser, device)
        writer.add_scalar('Loss_Train', train_loss, epoch)
        
        val_log_msg = ""
        current_val_f1 = -1.0
        
        if val_loader:
            val_loss, val_acc, val_f1 = validate_tile_classifier_model(model, val_loader, criterion, device)
            
            writer.add_scalar('Loss_Validation', val_loss, epoch)
            writer.add_scalar('Accuracy_Validation', val_acc, epoch)
            writer.add_scalar('F1_Validation', val_f1, epoch)
            
            val_log_msg = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            current_val_f1 = val_f1
            
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                torch.save(model.state_dict(), save_path)
                print(f"Epoch {epoch+1}: New best validation F1: {best_val_f1:.4f}. Checkpoint saved.")
                
            # Save last epoch checkpoint
            if (epoch + 1) % 10 == 0 or epoch == config.TC_EPOCHS - 1:
                torch.save(model.state_dict(), save_path)
                print(f"Epoch {epoch+1}: Checkpoint saved (no validation). ")
                
        else:
            if (epoch + 1) % 10 == 0 or epoch == config.TC_EPOCHS - 1:
                torch.save(model.state_dict(), save_path)
                print(f"Epoch {epoch+1}: Checkpoint saved (no validation). ")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{config.TC_EPOCHS} - Train Loss: {train_loss:.4f}{val_log_msg} ({epoch_duration:.2f}s)")
        
        writer.add_scalar('LearningRate', optimiser.param_groups[0]['lr'], epoch)

    total_training_time = time.time() - start_time
    print(f"Tile classifier training finished. Total time: {total_training_time:.2f}s")
    
    if val_loader:
        print(f"Best Validation F1: {best_val_f1:.4f}")
        
    print(f"Final model saved to: {save_path}")
    writer.close()
    
    return save_path 