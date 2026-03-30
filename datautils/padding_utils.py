"""
Padding utility functions - pad images/masks to a multiple of 32.

For IRDST and TSIRMT datasets:
- Original size: 150×200 (H×W)
- After padding: 160×224
- Padding mode: border replication (cv2.BORDER_REPLICATE)
- Padding location: right and bottom
"""

import cv2
import numpy as np
import torch
from PIL import Image


def pad_to_multiple_of_32(img, mask=None, fill_mode='replicate'):
    """
    Pad an image (and optional mask) to a multiple of 32.
    
    Args:
        img: input image, one of:
            - numpy array (H, W) or (H, W, C)
            - PIL Image
            - torch.Tensor (C, H, W) or (H, W)
        mask: optional mask with the same type/shape conventions as img
        fill_mode: padding fill mode
            - 'replicate': replicate border values (recommended for IR images)
            - 'constant': constant padding (value 0)
    
    Returns:
        padded_img: padded image (same type as input)
        padded_mask: padded mask (if provided)
        padding_info: dict containing {
            'original_size': (H, W),
            'padded_size': (H_new, W_new),
            'pad_bottom': int,
            'pad_right': int
        }
    """
    # 1. Convert to numpy array
    input_type = None
    if isinstance(img, Image.Image):
        input_type = 'pil'
        img_np = np.array(img)
    elif isinstance(img, torch.Tensor):
        input_type = 'torch'
        if img.ndim == 3:  # (C, H, W)
            img_np = img.permute(1, 2, 0).cpu().numpy()
        else:  # (H, W)
            img_np = img.cpu().numpy()
    else:
        input_type = 'numpy'
        img_np = img
    
    # 2. Get original size
    if img_np.ndim == 2:
        original_h, original_w = img_np.shape
        is_gray = True
    else:
        original_h, original_w = img_np.shape[:2]
        is_gray = False
    
    # 3. Compute target size (multiple of 32)
    target_h = ((original_h + 31) // 32) * 32
    target_w = ((original_w + 31) // 32) * 32
    
    # 4. Compute padding amounts
    pad_bottom = target_h - original_h
    pad_right = target_w - original_w
    
    # 5. Apply padding
    if fill_mode == 'replicate':
        border_type = cv2.BORDER_REPLICATE
    else:
        border_type = cv2.BORDER_CONSTANT
    
    if pad_bottom > 0 or pad_right > 0:
        # Image padding
        padded_img_np = cv2.copyMakeBorder(
            img_np, 
            top=0, bottom=pad_bottom,
            left=0, right=pad_right,
            borderType=border_type,
            value=0
        )
        
        # Mask padding (if provided)
        if mask is not None:
            if isinstance(mask, Image.Image):
                mask_np = np.array(mask)
            elif isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask
            
            # Masks use constant padding (0=background)
            padded_mask_np = cv2.copyMakeBorder(
                mask_np,
                top=0, bottom=pad_bottom,
                left=0, right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
        else:
            padded_mask_np = None
    else:
        padded_img_np = img_np
        padded_mask_np = mask if mask is not None else None
    
    # 6. Convert back to original type
    if input_type == 'pil':
        padded_img = Image.fromarray(padded_img_np.astype(np.uint8))
        if padded_mask_np is not None:
            padded_mask = Image.fromarray(padded_mask_np.astype(np.uint8))
        else:
            padded_mask = None
    elif input_type == 'torch':
        if is_gray:
            padded_img = torch.from_numpy(padded_img_np)
        else:
            padded_img = torch.from_numpy(padded_img_np).permute(2, 0, 1)
        
        if padded_mask_np is not None:
            padded_mask = torch.from_numpy(padded_mask_np)
        else:
            padded_mask = None
    else:
        padded_img = padded_img_np
        padded_mask = padded_mask_np
    
    # 7. Return padding info
    padding_info = {
        'original_size': (original_h, original_w),
        'padded_size': (target_h, target_w),
        'pad_bottom': pad_bottom,
        'pad_right': pad_right
    }
    
    return padded_img, padded_mask, padding_info


def crop_to_original_size(img, padding_info):
    """
    Crop a padded image back to the original size.
    
    Args:
        img: padded image
        padding_info: padding info returned by pad_to_multiple_of_32
    
    Returns:
        cropped_img: image cropped back to the original size
    """
    original_h, original_w = padding_info['original_size']
    
    # Handle different input types
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:  # (B, C, H, W)
            return img[:, :, :original_h, :original_w]
        elif img.ndim == 3:  # (C, H, W) or (B, H, W)
            return img[:, :original_h, :original_w]
        else:  # (H, W)
            return img[:original_h, :original_w]
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:  # (H, W)
            return img[:original_h, :original_w]
        else:  # (H, W, C)
            return img[:original_h, :original_w]
    elif isinstance(img, Image.Image):
        return img.crop((0, 0, original_w, original_h))
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


# ========== Test code ==========
if __name__ == '__main__':
    print("="*60)
    print("Padding utility function test")
    print("="*60)
    
    # Test 1: numpy array
    print("\nTest 1: Numpy array (150×200)")
    img_np = np.random.randint(0, 255, (150, 200), dtype=np.uint8)
    mask_np = np.random.randint(0, 2, (150, 200), dtype=np.uint8) * 255
    
    padded_img, padded_mask, info = pad_to_multiple_of_32(img_np, mask_np, fill_mode='replicate')
    print(f"  Original size: {info['original_size']}")
    print(f"  Padded size:   {info['padded_size']}")
    print(f"  Padding: bottom={info['pad_bottom']}, right={info['pad_right']}")
    print(f"  Result shapes: img={padded_img.shape}, mask={padded_mask.shape}")
    
    # Test cropping
    cropped = crop_to_original_size(padded_img, info)
    print(f"  Cropped: {cropped.shape}")
    print(f"  Crop correct: {np.array_equal(img_np, cropped)}")
    
    # Test 2: torch tensor
    print("\nTest 2: Torch tensor (150×200)")
    img_torch = torch.randn(150, 200)
    padded_torch, _, info = pad_to_multiple_of_32(img_torch, fill_mode='replicate')
    print(f"  Padded: {padded_torch.shape}")
    cropped_torch = crop_to_original_size(padded_torch, info)
    print(f"  Cropped: {cropped_torch.shape}")
    print(f"  Crop correct: {torch.allclose(img_torch, cropped_torch)}")
    
    # Test 3: PIL Image
    print("\nTest 3: PIL Image (150×200)")
    img_pil = Image.fromarray(img_np)
    padded_pil, _, info = pad_to_multiple_of_32(img_pil, fill_mode='replicate')
    print(f"  Padded: {padded_pil.size}")  # PIL.size is (W, H)
    cropped_pil = crop_to_original_size(padded_pil, info)
    print(f"  Cropped: {cropped_pil.size}")
    print(f"  Crop correct: {np.array_equal(img_np, np.array(cropped_pil))}")
    
    print("\n" + "="*60)
    print("All tests passed.")
    print("="*60)
