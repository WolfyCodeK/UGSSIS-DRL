import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import src.config as config
from src.dataset import UGSDataset
from src.segmentation_model import ResNetDeepLab
from src.active_learning_agent import DQNAgent
from src.utils import ensure_dir, calculate_class_weights, calculate_iou_binary, calculate_entropy_from_logits
from src.transforms import JointTransform, PhotometricAugmentation

class NormaliseSampleDict:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

        self.std_tensor = torch.tensor(std, dtype=torch.float32)
        self.mean_tensor = torch.tensor(mean, dtype=torch.float32)
        
        self.std_tensor[self.std_tensor == 0] = 1.0 

    def __call__(self, sample):
        image = sample['image']
        
        mean = self.mean_tensor.to(image.dtype).view(-1, 1, 1)
        std = self.std_tensor.to(image.dtype).view(-1, 1, 1)
        
        sample['image'] = transforms.Normalize(mean=mean, std=std)(image)
        
        return sample

def calculate_normalisation_constants(dataset):
    loader = DataLoader(dataset, batch_size=config.SEG_BATCH_SIZE * 2, shuffle=False, num_workers=2)
    
    num_channels = dataset[0]['image'].shape[0]
    sum_channels = torch.zeros(num_channels)
    sum_sq_channels = torch.zeros(num_channels)
    num_pixels = 0
    num_samples = 0

    print("Calculating normalisation constants (mean/std) from training data.")
    
    progress_bar = tqdm(loader, desc="Calculating Stats", leave=False)
    for batch in progress_bar:
        if batch is None: 
            continue
        
        images = batch['image']
        images = images.float()
        
        sum_channels += torch.sum(images, dim=[0, 2, 3])
        sum_sq_channels += torch.sum(images**2, dim=[0, 2, 3])
        
        batch_pixels = images.shape[0] * images.shape[2] * images.shape[3]
        num_pixels += batch_pixels
        num_samples += images.shape[0]
        
    if num_pixels == 0:
        raise Exception("No pixels found to calculate normalisation constants.")
    
    mean = sum_channels / num_pixels
    std = torch.sqrt((sum_sq_channels / num_pixels) - mean**2)

    mean_list = mean.tolist()
    std_list = std.tolist()

    print(f"Calculated Mean: {mean_list}")
    print(f"Calculated Std Dev: {std_list}")

    return mean_list, std_list

def train_segmentation_epoch(model, dataloader, criterion, optimiser, device, model_type):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    num_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Train Epoch ({model_type.capitalize()})", leave=False)
    for batch in progress_bar:
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        batch_size = images.shape[0]
        
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item() * batch_size 
        
        with torch.no_grad():
            batch_iou = calculate_iou_binary(outputs, masks)
        total_iou += batch_iou * batch_size
        
        num_batches += 1
        num_samples += batch_size
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{batch_iou:.4f}'})
        
    if num_samples > 0:
        avg_loss = total_loss / num_samples
        avg_iou = total_iou / num_samples
    else:
        avg_loss = 0.0
        avg_iou = 0.0
    
    return avg_loss, avg_iou

def validate_segmentation_model(model, dataloader, criterion, device, model_type, norm_stats, writer=None, global_step=None):
    
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Validation ({model_type.capitalize()})", leave=False)
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            if batch is None: continue
            
            images_unnormalised = batch['image'].to(device)
            masks = batch['mask'].to(device)
            original_masks = batch.get('original_mask')

            if original_masks is not None:
                original_masks = original_masks.to(device)

            batch_size = images_unnormalised.shape[0]

            if norm_stats is None:
                print("Normalisation stats not provided")
                return float('nan'), float('nan')
            
            mean = torch.tensor(norm_stats['mean'], dtype=images_unnormalised.dtype, device=device).view(-1, 1, 1)
            
            std = torch.tensor(norm_stats['std'], dtype=images_unnormalised.dtype, device=device).view(-1, 1, 1)
            
            std = torch.where(std == 0, torch.tensor(1.0, device=device), std)
            images = (images_unnormalised - mean) / std
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item() * batch_size

            batch_iou = calculate_iou_binary(outputs, masks)
            total_iou += batch_iou * batch_size
            
            num_samples += batch_size
            progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

    if num_samples > 0:
        avg_loss = total_loss / num_samples
        avg_iou = total_iou / num_samples
    else:
        avg_loss = 0.0
        avg_iou = 0.0

    print(f"Validation ({model_type.capitalize()}) - Step: {global_step} - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")

    if writer is not None and global_step is not None:
        writer.add_scalar(f'Loss_Validation_{model_type.capitalize()}', avg_loss, global_step)
        writer.add_scalar(f'IoU_Validation_{model_type.capitalize()}', avg_iou, global_step)

    return avg_loss, avg_iou

def score_unlabeled_samples(model, dataset, indices, device):
    model.eval()
    scores = {}

    score_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.SEG_BATCH_SIZE * 2, 
        shuffle=False, 
        num_workers=2
    )

    with torch.no_grad():
        for batch in tqdm(score_loader, desc="Scoring Candidates", leave=False):
            
            if batch is None: 
                continue
            
            images = batch['image'].to(device)
            batch_indices = batch['index'].tolist()

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            entropy_per_pixel = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            
            avg_entropy_per_image = torch.mean(entropy_per_pixel, dim=[1, 2])

            for i, score in enumerate(avg_entropy_per_image.cpu().tolist()):
                original_index = batch_indices[i]
                scores[original_index] = score
                
    model.train()
    
    return scores

def select_samples_blended(
    cycle, initial_unlabeled_count, current_val_iou_green, current_val_iou_urban, current_val_loss_green, current_val_loss_urban, agent_green, agent_urban, model_green, model_urban, full_dataset, unlabeled_indices, num_candidates, batch_size, device, norm_stats, writer
):
    if not unlabeled_indices:
        print("No unlabeled indices to select from.")
        return [], np.zeros(config.DQN_STATE_SIZE), set(), set()


    num_available = len(unlabeled_indices)
    num_to_sample = min(num_candidates, num_available)
    if num_to_sample == 0:
        print("No candidates available for uncertainty calculation.")

        state = np.array([
            current_val_iou_green, 
            current_val_iou_urban, 
            current_val_loss_green, 
            current_val_loss_urban, 
            0.0, 0.0, 0.0, 0.0,
            cycle / config.AL_CYCLES if config.AL_CYCLES > 0 else 0.0, 
            0.0
        ], dtype=np.float32)
        
        return [], state, set(), set()
        
    candidate_indices = random.sample(unlabeled_indices, num_to_sample)
    print(f"Calculating uncertainty stats over {len(candidate_indices)} candidate samples.")

    candidate_subset = Subset(full_dataset, candidate_indices)
    scoring_batch_size = config.SEG_BATCH_SIZE * 4 
    
    candidate_loader = DataLoader(candidate_subset, batch_size=scoring_batch_size, shuffle=False, num_workers=2)

    model_green.eval()
    model_urban.eval()
    
    entropies_green_list = []
    entropies_urban_list = []
    processed_indices_for_state = []
    
    if norm_stats is None:
        print("Normalisation stats missing.")
        return [], np.zeros(config.DQN_STATE_SIZE), set(), set()
    
    mean = torch.tensor(norm_stats['mean'], dtype=torch.float32, device=device).view(-1, 1, 1)
    
    std = torch.tensor(norm_stats['std'], dtype=torch.float32, device=device).view(-1, 1, 1)
    
    std = torch.where(std == 0, torch.tensor(1.0, device=device), std)
    
    with torch.no_grad():
        for batch in candidate_loader:
            if batch is None: continue
            
            images_unnormalised = batch['image'].to(device)
            batch_indices = batch['index']
            images = (images_unnormalised - mean) / std
            
            outputs_green = model_green(images)
            entropy_green_batch = calculate_entropy_from_logits(outputs_green)
            entropies_green_list.extend(entropy_green_batch.cpu().numpy())

            outputs_urban = model_urban(images)
            entropy_urban_batch = calculate_entropy_from_logits(outputs_urban)
            entropies_urban_list.extend(entropy_urban_batch.cpu().numpy())
            
            processed_indices_for_state.extend(batch_indices.cpu().numpy())

    model_green.train()
    model_urban.train()

    mean_uncert_g, std_uncert_g, mean_uncert_u, std_uncert_u = 0.0, 0.0, 0.0, 0.0
    
    if entropies_green_list:
        entropies_green_state = np.array(entropies_green_list)
        mean_uncert_g = np.mean(entropies_green_state)
        
        if len(entropies_green_state) > 1:
            std_uncert_g = np.std(entropies_green_state)
        else:
            std_uncert_g = 0.0
            
    if entropies_urban_list:
        entropies_urban_state = np.array(entropies_urban_list)
        mean_uncert_u = np.mean(entropies_urban_state)
        
        if len(entropies_urban_state) > 1:
            
            std_uncert_u = np.std(entropies_urban_state)
        else:
            std_uncert_u = 0.0
            
    if config.AL_CYCLES > 0:
        cycle_norm = cycle / config.AL_CYCLES
    else:
        cycle_norm = 0.0

    if initial_unlabeled_count > 0:
        budget_norm = len(unlabeled_indices) / initial_unlabeled_count
    else:
        budget_norm = 0.0

    state = np.array([
        current_val_iou_green, current_val_iou_urban,
        current_val_loss_green, current_val_loss_urban,
        mean_uncert_g, std_uncert_g,
        mean_uncert_u, std_uncert_u,
        cycle_norm, budget_norm
    ], dtype=np.float32)

    if len(state) != config.DQN_STATE_SIZE:
        print(f"State size mismatch. Expected {config.DQN_STATE_SIZE}, got {len(state)}.")
        
        state = np.zeros(config.DQN_STATE_SIZE)

    q_raw_g = agent_green.get_q_value(state)
    q_raw_u = agent_urban.get_q_value(state)

    min_q_g, max_q_g = agent_green.get_q_range()
    min_q_u, max_q_u = agent_urban.get_q_range()

    range_g = max_q_g - min_q_g
    range_u = max_q_u - min_q_u
    
    if range_g > 0:
        q_norm_g = (q_raw_g - min_q_g) / (range_g + 1e-6)
    else:
        q_norm_g = 0.5
        
    if range_u > 0:
        q_norm_u = (q_raw_u - min_q_u) / (range_u + 1e-6)
    else:
        q_norm_u = 0.5

    alpha_green = np.clip(q_norm_g, 0.0, 1.0)
    alpha_urban = np.clip(q_norm_u, 0.0, 1.0)

    print(f"Cycle {cycle} - Q(G): {q_raw_g:.4f} (Range: [{min_q_g:.4f}, {max_q_g:.4f}] -> Norm: {q_norm_g:.4f}) -> AlphaG: {alpha_green:.3f}")
    
    print(f"Cycle {cycle} - Q(U): {q_raw_u:.4f} (Range: [{min_q_u:.4f}, {max_q_u:.4f}] -> Norm: {q_norm_u:.4f}) -> AlphaU: {alpha_urban:.3f}")

    if writer is not None:
        writer.add_scalar('AL/Q_Value_Green_Raw', q_raw_g, cycle)
        writer.add_scalar('AL/Q_Value_Urban_Raw', q_raw_u, cycle)
        writer.add_scalar('AL/Q_Range_Min_Green', min_q_g, cycle)
        writer.add_scalar('AL/Q_Range_Max_Green', max_q_g, cycle)
        writer.add_scalar('AL/Q_Range_Min_Urban', min_q_u, cycle)
        writer.add_scalar('AL/Q_Range_Max_Urban', max_q_u, cycle)
        writer.add_scalar('AL/Alpha_Weight_Green', alpha_green, cycle)
        writer.add_scalar('AL/Alpha_Weight_Urban', alpha_urban, cycle)
        
        writer.add_scalar('AL/State_Mean_Uncertainty_Green', mean_uncert_g, cycle)
        writer.add_scalar('AL/State_Std_Uncertainty_Green', std_uncert_g, cycle)
        writer.add_scalar('AL/State_Mean_Uncertainty_Urban', mean_uncert_u, cycle)
        writer.add_scalar('AL/State_Std_Uncertainty_Urban', std_uncert_u, cycle)
        writer.add_scalar('AL/State_Cycle_Norm', cycle_norm, cycle)
        writer.add_scalar('AL/State_Budget_Norm', budget_norm, cycle)

    if not processed_indices_for_state:
        print("No candidates were processed during state calculation so cannot select tiles.")
        
        return [], state, set(), set()

    processed_indices = np.array(processed_indices_for_state)
    entropies_green = np.array(entropies_green_list) 
    entropies_urban = np.array(entropies_urban_list)

    eps_norm = 1e-6
    
    min_h_g, max_h_g = np.min(entropies_green), np.max(entropies_green)
    min_h_u, max_h_u = np.min(entropies_urban), np.max(entropies_urban)
    
    range_h_g = max_h_g - min_h_g
    range_h_u = max_h_u - min_h_u
    
    if range_h_g > 0:
        norm_entropies_green = (entropies_green - min_h_g) / (range_h_g + eps_norm)
    else:
        norm_entropies_green = np.zeros_like(entropies_green)
        
    if range_h_u > 0:
        norm_entropies_urban = (entropies_urban - min_h_u) / (range_h_u + eps_norm)
    else:
        norm_entropies_urban = np.zeros_like(entropies_urban)
    
    norm_entropies_green = np.clip(norm_entropies_green, 0, 1)
    norm_entropies_urban = np.clip(norm_entropies_urban, 0, 1)

    random_scores = np.random.rand(len(processed_indices))
    
    scores_green = alpha_green * norm_entropies_green + (1 - alpha_green) * random_scores
    
    scores_urban = alpha_urban * norm_entropies_urban + (1 - alpha_urban) * random_scores

    candidate_results = list(zip(processed_indices.tolist(), scores_green, scores_urban))

    num_to_select_each = batch_size // 2
    
    candidate_results.sort(key=lambda x: x[1], reverse=True)
    top_indices_green = {index for index, _, _ in candidate_results[:num_to_select_each]}
    
    candidate_results.sort(key=lambda x: x[2], reverse=True) 
    top_indices_urban = {index for index, _, _ in candidate_results[:num_to_select_each]}

    selected_indices_set = top_indices_green | top_indices_urban
    
    needed = batch_size - len(selected_indices_set)
    
    if needed > 0:
        remaining_candidates = [index for index in processed_indices if index not in selected_indices_set]
        
        fill_count = min(needed, len(remaining_candidates))
        
        if fill_count > 0:
            random_fill = random.sample(remaining_candidates, fill_count)
            selected_indices_set.update(random_fill)

    selected_indices = list(selected_indices_set)
    
    if len(selected_indices) > batch_size:
        selected_indices = random.sample(selected_indices, batch_size)
    elif len(selected_indices) < batch_size:
        print(f"Could only select {len(selected_indices)} unique samples from {len(processed_indices)} candidates, target was {batch_size}")

    print(f"Selected {len(selected_indices)} samples using blended strategy (AlphaG: {alpha_green:.3f}, AlphaU: {alpha_urban:.3f}).")

    return selected_indices, state, top_indices_green, top_indices_urban 

def run_training():
    run_name = config.RUN_NAME or f"UGS_AL_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = Path(config.LOG_DIR) / run_name
    checkpoint_dir = Path(config.CHECKPOINT_DIR) / run_name
    
    ensure_dir(log_dir)
    ensure_dir(checkpoint_dir)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading full training dataset.")
    
    full_train_dataset = UGSDataset(split='train', transform=None)
    
    if not full_train_dataset:
        print("Training dataset failed to load.")
        return

    num_train_samples = len(full_train_dataset)
    all_indices = list(range(num_train_samples))
    
    random.shuffle(all_indices)
    
    initial_pool_size = int(config.INITIAL_POOL_RATIO * num_train_samples)
    initial_labeled_indices = all_indices[:initial_pool_size]
    
    norm_calc_subset = Subset(full_train_dataset, initial_labeled_indices)
    
    norm_mean, norm_std = calculate_normalisation_constants(norm_calc_subset)
    norm_stats = {'mean': norm_mean, 'std': norm_std}
    
    val_dataset = UGSDataset(split='val', transform=None) 
    
    if not val_dataset:
        print("Validation dataset failed to load.")
        val_loader_green, val_loader_urban = None, None
    else:
        val_dataset_green = UGSDataset(split='val', target_class_index=config.GREEN_SPACE_INDEX)
        val_dataset_urban = UGSDataset(split='val', target_class_index=config.URBAN_INDEX)
        
        val_loader_green = DataLoader(val_dataset_green, batch_size=config.SEG_BATCH_SIZE * 2, shuffle=False, num_workers=2)
        val_loader_urban = DataLoader(val_dataset_urban, batch_size=config.SEG_BATCH_SIZE * 2, shuffle=False, num_workers=2)

    normalise_transform = NormaliseSampleDict(mean=norm_stats['mean'], std=norm_stats['std'])
    
    train_transform = transforms.Compose([
        JointTransform(flip_prob=config.AUG_FLIP_PROB), 
        PhotometricAugmentation(
            blur_prob=config.AUG_BLUR_PROB, 
            bgr_noise_strength=config.AUG_BGR_NOISE_STRENGTH, 
            nir_noise_strength=config.AUG_NIR_NOISE_STRENGTH
        ), 
        normalise_transform
    ])

    print("Initialising segmentation models.")
    
    model_green = ResNetDeepLab(
        num_classes=2,
        backbone_name=config.SEG_MODEL_BACKBONE,
        pretrained_backbone=config.PRETRAINED_BACKBONE
    ).to(device)
    model_urban = ResNetDeepLab(
        num_classes=2,
        backbone_name=config.SEG_MODEL_BACKBONE,
        pretrained_backbone=config.PRETRAINED_BACKBONE
    ).to(device)

    optimiser_green = optim.Adam(model_green.parameters(), lr=config.SEG_LR)
    optimiser_urban = optim.Adam(model_urban.parameters(), lr=config.SEG_LR)
    
    scheduler_green = optim.lr_scheduler.ExponentialLR(optimiser_green, gamma=config.LR_SCHEDULER_GAMMA)
    scheduler_urban = optim.lr_scheduler.ExponentialLR(optimiser_urban, gamma=config.LR_SCHEDULER_GAMMA)

    print(f"Initialising DQN agents with state size {config.DQN_STATE_SIZE}")
    agent_green = DQNAgent(device=device)
    agent_urban = DQNAgent(device=device)
    
    criterion_green = nn.CrossEntropyLoss().to(device) 
    criterion_urban = nn.CrossEntropyLoss().to(device)

    print("Creating base training datasets with transforms.")
    train_base_dataset_green = UGSDataset(
        split='train', 
        transform=train_transform, 
        target_class_index=config.GREEN_SPACE_INDEX
    )
    train_base_dataset_urban = UGSDataset(
        split='train', 
        transform=train_transform, 
        target_class_index=config.URBAN_INDEX
    )
    
    if not train_base_dataset_green or not train_base_dataset_urban:
        print("Failed to create base training datasets.")
        return

    num_samples = len(train_base_dataset_green)
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)

    initial_pool_size = int(config.INITIAL_POOL_RATIO * num_samples)
    initial_pool_size = max(initial_pool_size, config.SEG_BATCH_SIZE)
    
    initial_labeled_indices = all_indices[:initial_pool_size]
    unlabeled_indices = all_indices[initial_pool_size:]
    initial_unlabeled_count = len(unlabeled_indices)
    
    print(f"Initial setup: {len(initial_labeled_indices)} labeled, {len(unlabeled_indices)} unlabeled.")

    print("Calculating initial class weights.")
    
    initial_labeled_subset_green = Subset(train_base_dataset_green, initial_labeled_indices)
    initial_labeled_subset_urban = Subset(train_base_dataset_urban, initial_labeled_indices)
    
    weights_green = calculate_class_weights(initial_labeled_subset_green, num_classes=2)
    
    weights_urban = calculate_class_weights(initial_labeled_subset_urban, num_classes=2)
    
    criterion_green = nn.CrossEntropyLoss(weight=weights_green).to(device)
    criterion_urban = nn.CrossEntropyLoss(weight=weights_urban).to(device)

    print("Starting initial training phase.")
    
    initial_train_loader_green = DataLoader(initial_labeled_subset_green, batch_size=config.SEG_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    initial_train_loader_urban = DataLoader(initial_labeled_subset_urban, batch_size=config.SEG_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    for epoch in range(config.EPOCHS_PER_CYCLE):
        print(f"Initial Epoch {epoch+1}/{config.EPOCHS_PER_CYCLE}")
        
        train_loss_g, train_iou_g = train_segmentation_epoch(model_green, initial_train_loader_green, criterion_green, optimiser_green, device, config.MODEL_GREEN)
        
        train_loss_u, train_iou_u = train_segmentation_epoch(model_urban, initial_train_loader_urban, criterion_urban, optimiser_urban, device, config.MODEL_URBAN)
        
        writer.add_scalar('Loss_Train_Initial_Green', train_loss_g, epoch)
        writer.add_scalar('IoU_Train_Initial_Green', train_iou_g, epoch)
        writer.add_scalar('Loss_Train_Initial_Urban', train_loss_u, epoch)
        writer.add_scalar('IoU_Train_Initial_Urban', train_iou_u, epoch)
        
    print("Initial training phase complete.")

    print("Running initial validation.")
    
    if val_loader_green and val_loader_urban:
        current_val_loss_green, current_val_iou_green = validate_segmentation_model(
            model_green, val_loader_green, criterion_green, device, config.MODEL_GREEN, norm_stats, writer, 0
        )
        
        current_val_loss_urban, current_val_iou_urban = validate_segmentation_model(
            model_urban, val_loader_urban, criterion_urban, device, config.MODEL_URBAN, norm_stats, writer, 0
        )
        
        print(f"Initial Validation - Green: Loss={current_val_loss_green:.4f}, IoU={current_val_iou_green:.4f}")
        
        print(f"Initial Validation - Urban: Loss={current_val_loss_urban:.4f}, IoU={current_val_iou_urban:.4f}")
    else:
        print("Skipping initial validation as validation loaders are not available.")
        current_val_loss_green, current_val_iou_green = 0.0, 0.0
        current_val_loss_urban, current_val_iou_urban = 0.0, 0.0

    prev_val_iou_green = current_val_iou_green 
    prev_val_iou_urban = current_val_iou_urban

    print("Starting active learning loop.")
    total_al_time = 0

    for cycle in range(1, config.AL_CYCLES + 1):
        cycle_start_time = time.time()
        print(f"AL Cycle {cycle}/{config.AL_CYCLES}")
        print(f"Labeled pool size: {len(initial_labeled_indices)}, Unlabeled pool size: {len(unlabeled_indices)}")

        selection_start_time = time.time()
        if not unlabeled_indices:
            print("No more unlabeled samples available. Stopping AL loop.")
            break

        selected_indices_batch, current_cycle_state, selected_by_green, selected_by_urban = select_samples_blended(
            cycle=cycle,
            initial_unlabeled_count=initial_unlabeled_count,
            current_val_iou_green=current_val_iou_green,
            current_val_iou_urban=current_val_iou_urban,
            current_val_loss_green=current_val_loss_green,
            current_val_loss_urban=current_val_loss_urban,
            agent_green=agent_green,
            agent_urban=agent_urban,
            model_green=model_green,
            model_urban=model_urban,
            full_dataset=full_train_dataset,
            unlabeled_indices=unlabeled_indices,
            num_candidates=config.AL_CANDIDATES_PER_CYCLE,
            batch_size=config.BATCH_ACQUISITION_SIZE,
            device=device,
            norm_stats=norm_stats,
            writer=writer
        )
        
        selection_time = time.time() - selection_start_time
        print(f"Sample selection took {selection_time:.2f}s")

        if not selected_indices_batch:
            print(f"Cycle {cycle}: No samples selected. Skipping training and validation.")
            continue

        initial_labeled_indices.extend(selected_indices_batch)
        unlabeled_indices = [index for index in unlabeled_indices if index not in selected_indices_batch]
        print(f"Updated pools: {len(initial_labeled_indices)} labeled, {len(unlabeled_indices)} unlabeled.")

        retrain_start_time = time.time()
        print(f"Retraining models on {len(initial_labeled_indices)} samples...")
        
        current_labeled_subset_green = Subset(train_base_dataset_green, initial_labeled_indices)
        current_labeled_subset_urban = Subset(train_base_dataset_urban, initial_labeled_indices)
        
        weights_green = calculate_class_weights(current_labeled_subset_green, num_classes=2)
        weights_urban = calculate_class_weights(current_labeled_subset_urban, num_classes=2)
        criterion_green = nn.CrossEntropyLoss(weight=weights_green).to(device)
        criterion_urban = nn.CrossEntropyLoss(weight=weights_urban).to(device)

        current_train_loader_green = DataLoader(current_labeled_subset_green, batch_size=config.SEG_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        
        current_train_loader_urban = DataLoader(current_labeled_subset_urban, batch_size=config.SEG_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        
        for epoch in range(config.EPOCHS_PER_CYCLE):
            epoch_num = (cycle - 1) * config.EPOCHS_PER_CYCLE + epoch + 1 + config.EPOCHS_PER_CYCLE
            
            print(f"Cycle {cycle} - Retraining Epoch {epoch+1}/{config.EPOCHS_PER_CYCLE} (Overall Epoch {epoch_num})")
            
            train_loss_g, train_iou_g = train_segmentation_epoch(model_green, current_train_loader_green, criterion_green, optimiser_green, device, config.MODEL_GREEN)
            
            writer.add_scalar('Loss_Train_Green', train_loss_g, cycle)
            writer.add_scalar('IoU_Train_Green', train_iou_g, cycle)

            train_loss_u, train_iou_u = train_segmentation_epoch(model_urban, current_train_loader_urban, criterion_urban, optimiser_urban, device, config.MODEL_URBAN)
            
            writer.add_scalar('Loss_Train_Urban', train_loss_u, cycle)
            writer.add_scalar('IoU_Train_Urban', train_iou_u, cycle)
            
            writer.add_scalar('LR/Green', optimiser_green.param_groups[0]['lr'], cycle)
            writer.add_scalar('LR/Urban', optimiser_urban.param_groups[0]['lr'], cycle)
            
        retrain_time = time.time() - retrain_start_time
        print(f"Retraining took {retrain_time:.2f}s")

        validation_start_time = time.time()
        reward_g = 0.0
        reward_u = 0.0
        delta_iou_g = 0.0
        delta_iou_u = 0.0

        if cycle % config.VALIDATION_FREQ == 0:
            print(f"Running validation for Cycle {cycle}...")
            
            val_loss_g, val_iou_g = validate_segmentation_model(
                model_green, val_loader_green, criterion_green, device, config.MODEL_GREEN, norm_stats, writer, cycle
            )
            
            val_loss_u, val_iou_u = validate_segmentation_model(
                model_urban, val_loader_urban, criterion_urban, device, config.MODEL_URBAN, norm_stats, writer, cycle
            )
            
            current_val_loss_green = val_loss_g
            current_val_iou_green = val_iou_g
            current_val_loss_urban = val_loss_u
            current_val_iou_urban = val_iou_u

            delta_iou_g = current_val_iou_green - prev_val_iou_green
            delta_iou_u = current_val_iou_urban - prev_val_iou_urban
            
            reward_g = np.clip(delta_iou_g, -0.1, 0.1)
            reward_u = np.clip(delta_iou_u, -0.1, 0.1)
            
            print(f"Cycle {cycle} Validation Results & Rewards:")
            
            print(f"  Green - IoU: {current_val_iou_green:.4f} (delta-IoU: {delta_iou_g:+.4f}), Loss: {current_val_loss_green:.4f}, Reward: {reward_g:.4f}")
            
            print(f"  Urban - IoU: {current_val_iou_urban:.4f} (delta-IoU: {delta_iou_u:+.4f}), Loss: {current_val_loss_urban:.4f}, Reward: {reward_u:.4f}")

            prev_val_iou_green = current_val_iou_green
            prev_val_iou_urban = current_val_iou_urban

            # Update agent 
            agent_green.step(current_cycle_state, reward_g)
            agent_urban.step(current_cycle_state, reward_u)
            print(f"Agents updated with state from start of cycle and calculated rewards.")

            writer.add_scalar('AL/Delta_IoU_Green', delta_iou_g, cycle)
            writer.add_scalar('AL/Delta_IoU_Urban', delta_iou_u, cycle)
            writer.add_scalar('AL/Reward_Green', reward_g, cycle)
            writer.add_scalar('AL/Reward_Urban', reward_u, cycle)
            
            # Update learning rate scheduler
            if cycle >= config.LR_SCHEDULER_START_CYCLE:
                scheduler_green.step()
                scheduler_urban.step()
                print(f"LR Scheduler step performed at cycle {cycle}")

        else:
            print(f"Skipping validation and agent update for Cycle {cycle} (Validation Freq: {config.VALIDATION_FREQ})")

            writer.add_scalar(f'Loss_Validation_Green', current_val_loss_green, cycle)
            writer.add_scalar(f'IoU_Validation_Green', current_val_iou_green, cycle)
            writer.add_scalar(f'Loss_Validation_Urban', current_val_loss_urban, cycle)
            writer.add_scalar(f'IoU_Validation_Urban', current_val_iou_urban, cycle)
        
        validation_time = time.time() - validation_start_time

        min_q_g, max_q_g = agent_green.get_q_range()
        min_q_u, max_q_u = agent_urban.get_q_range()
        
        writer.add_scalar('AL/Q_Range_Min_Green', min_q_g, cycle)
        writer.add_scalar('AL/Q_Range_Max_Green', max_q_g, cycle)
        writer.add_scalar('AL/Q_Range_Min_Urban', min_q_u, cycle)
        writer.add_scalar('AL/Q_Range_Max_Urban', max_q_u, cycle)

        # Save checkpoint 
        if cycle % config.VALIDATION_FREQ == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{cycle}.pth"
            
            checkpoint_data = {
                'cycle': cycle,
                'model_green': model_green.state_dict(),
                'model_urban': model_urban.state_dict(),
                'optimiser_green': optimiser_green.state_dict(),
                'optimiser_urban': optimiser_urban.state_dict(),
                'labeled_indices': initial_labeled_indices,
            }
            torch.save(checkpoint_data, checkpoint_path)
            
            print(f"Saved checkpoint for cycle {cycle} to {checkpoint_path}")

        cycle_end_time = time.time()
        cycle_duration = cycle_end_time - cycle_start_time
        total_al_time += cycle_duration
        
        print(f"AL cycle {cycle} finished in {cycle_duration:.2f}s")
        
        writer.add_scalar('Time/Cycle_Duration_s', cycle_duration, cycle)
        writer.add_scalar('Time/Selection_s', selection_time, cycle)
        writer.add_scalar('Time/Retraining_s', retrain_time, cycle)
        writer.add_scalar('Time/Validation_Update_s', validation_time, cycle)

    total_al_duration_h = total_al_time / 3600
    
    print(f"Active learning finished - Total time: {total_al_time:.2f}s ({total_al_duration_h:.2f} hours)")
    
    writer.close()