import torch
import random
from torchvision import transforms
import src.config as config
import src.utils as utils

class JointTransform:
    def __init__(self, flip_prob=config.AUG_FLIP_PROB):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        # Horizontal Flip
        if random.random() < self.flip_prob:
            image = torch.fliplr(image)
            mask = torch.fliplr(mask.unsqueeze(0))# Add channel dim for fliplr
            mask = mask.squeeze(0) # Remove channel afterwards

        # Vertical Flip
        if random.random() < self.flip_prob:
            image = torch.flipud(image)
            mask = torch.flipud(mask.unsqueeze(0)) # Add channel dim for flipud
            mask = mask.squeeze(0) # Remove channel afterwards

        # Rotation (0°, 90°, 180°, 270°)
        k = random.randint(0, 3)
        
        if k > 0:
            image = torch.rot90(image, k=k, dims=(1, 2))
            mask = torch.rot90(mask, k=k, dims=(0, 1))

        sample['image'] = image
        sample['mask'] = mask
        
        return sample

class PhotometricAugmentation:
    """
    Applies noise and blur to images and recalculates EVI
    """
    def __init__(
        self,
        bgr_noise_strength=config.AUG_BGR_NOISE_STRENGTH,
        nir_noise_strength=config.AUG_NIR_NOISE_STRENGTH,
        blur_prob=config.AUG_BLUR_PROB
    ):
        self.bgr_noise_strength = bgr_noise_strength
        self.nir_noise_strength = nir_noise_strength
        self.blur_prob = blur_prob
        blur_kernel_size = 3
            
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=blur_kernel_size)

    def __call__(self, sample):
        image = sample['image']

        original_evi = utils.get_evi(image[:4]) 
        
        # Apply Noise to BGRNir
        image_bgrnir = image[:4].clone() # Work on a copy
        if self.bgr_noise_strength > 0:
            bgr_noise = torch.randn_like(image_bgrnir[:3]) * self.bgr_noise_strength
            image_bgrnir[:3] += bgr_noise
        
        if self.nir_noise_strength > 0:
            nir_noise = torch.randn_like(image_bgrnir[3]) * self.nir_noise_strength
            image_bgrnir[3] += nir_noise
        
        # Apply Blur to BGRNir
        if random.random() < self.blur_prob:
            blurred_bgrnir = torch.zeros_like(image_bgrnir)
            
            for i in range(4):
                channel_to_blur = image_bgrnir[i].unsqueeze(0).unsqueeze(0)
                blurred_channel = self.gaussian_blur(channel_to_blur)
                blurred_bgrnir[i] = blurred_channel.squeeze(0).squeeze(0)
                
            image_bgrnir = blurred_bgrnir
        
        # Reconstruct the image tensor
        augmented_image = torch.cat((image_bgrnir, original_evi.unsqueeze(0)), dim=0)
        sample['image'] = augmented_image

        image[4] = original_evi
        sample['image'] = image 

        return sample