import os
import cv2
import random

import numpy as np
# import kornia.feature as KF
# import flowiz as fz

from PIL import Image, ImageOps

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from .padding_utils import pad_to_multiple_of_32

class BUPTMIRSDTSet(Dataset):
    """
    LMIRSTD dataset loader.
    Uses the original resolution 640×512 (no resize/padding).
    """

    def __init__(self, mode, dataset_dir, folder_id, base_size=256, crop_size=256, channel_size=3):
        super(BUPTMIRSDTSet, self).__init__()

        self.mode = mode
        self.imagedir = dataset_dir
        # base_size and crop_size are ignored; this dataset always uses the original resolution
        self.base_size = base_size
        self.crop_size = crop_size
        
        self.in_channels = channel_size

        # (Optional) feature extractor for background alignment

        if mode == 'train':
            self.sampling_strategy = 'Interval'
        else:
            self.sampling_strategy = 'Interval'

        self._items, self._mask_paths = [], []
        
        # Build index entries
        def _collect_sequence_files(folder_name):
            sequence_dir = os.path.join(self.imagedir, 'images', folder_name)
            mask_dir = os.path.join(self.imagedir, 'masks', folder_name)

            image_exts = ('.png', '.bmp', '.jpg', '.jpeg')
            mask_exts = ('.png', '.jpg', '.jpeg', '.bmp')

            image_files = sorted([f for f in os.listdir(sequence_dir) if f.lower().endswith(image_exts)])

            mask_map = {}
            for img_name in image_files:
                base = os.path.splitext(img_name)[0]
                mask_file = None
                for ext in mask_exts:
                    candidate = base + ext
                    if os.path.exists(os.path.join(mask_dir, candidate)):
                        mask_file = candidate
                        break
                if mask_file is None:
                    print(f"Warning: cannot find mask for image {img_name} (folder: {folder_name})")
                    continue
                mask_map[img_name] = mask_file

            # Return only images that have a valid mask
            valid_image_files = [img for img in image_files if img in mask_map]
            return valid_image_files, mask_map

        if self.mode == 'train':
            for folder in folder_id:
                img_paths, mask_map = _collect_sequence_files(folder)
                file_num = len(img_paths)
                
                # Sliding-window sampling
                for i in range(channel_size-1, file_num):
                    img_pair, mask_pair = [], []
                    for j in range(i - channel_size + 1, i + 1):
                        img_name = img_paths[j]
                        img_pair.append(f"images/{folder}/{img_name}")
                        mask_pair.append(f"masks/{folder}/{mask_map[img_name]}")
                    self._items.append(img_pair)
                    self._mask_paths.append(mask_pair)
                    
        elif self.mode == 'test':
            for folder in folder_id:
                img_paths, mask_map = _collect_sequence_files(folder)
                file_num = len(img_paths)
                
                # In testing, process every frame
                for i in range(0, file_num):
                    img_pair, mask_pair = [], []
                    if i < channel_size - 1:
                        # For the first few frames, repeat the first frame
                        first_img = img_paths[0]
                        img_pair = [f"images/{folder}/{first_img}"] * (channel_size - i - 1)
                        mask_pair = [f"masks/{folder}/{mask_map[first_img]}"] * (channel_size - i - 1)
                        for j in range(i+1):
                            img_name = img_paths[j]
                            img_pair.append(f"images/{folder}/{img_name}")
                            mask_pair.append(f"masks/{folder}/{mask_map[img_name]}")
                    else:
                        for j in range(i - channel_size + 1, i + 1):
                            img_name = img_paths[j]
                            img_pair.append(f"images/{folder}/{img_name}")
                            mask_pair.append(f"masks/{folder}/{mask_map[img_name]}")
                    self._items.append(img_pair)
                    self._mask_paths.append(mask_pair)

    def _sync_transform(self, imgs, mask, trajectory):
        """
        Training-time transform for LMIRSTD.
        Keeps original resolution (no resize/padding) and only applies random mirroring.
        """
        num_img = len(imgs)
        
        # Random mirror (the only augmentation)
        if random.random() < 0.5:
            for i in range(num_img):
                imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            trajectory = trajectory.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Keep original resolution (no resize/crop/padding)
        # Final transform
        for i in range(num_img):
            imgs[i] = np.array(imgs[i])
        imgs, mask, trajectory = np.array(imgs, dtype=np.float32), np.array(mask, dtype=np.float32), np.array(trajectory, dtype=np.float32)
        return imgs, mask, trajectory

    
    def _val_sync_transform(self, imgs, mask, trajectory):
        """
        Validation/testing transform for LMIRSTD.
        Keeps the original resolution (no resize/padding).
        """
        num_img = len(imgs)
        
        # Keep original resolution (no resize/padding)
        for i in range(num_img):
            imgs[i] = np.array(imgs[i])
        imgs, mask, trajectory = np.array(imgs, dtype=np.float32), np.array(mask, dtype=np.float32), np.array(trajectory, dtype=np.float32)
        
        return imgs, mask, trajectory

    def __getitem__(self, idx):
        img_ids = self._items[idx]
        mask_ids = self._mask_paths[idx]

        # Load image sequence
        imgs = []
        for img_path in img_ids:
            full_path = os.path.join(self.imagedir, img_path)
            try:
                img = Image.open(full_path).convert('L')
                imgs.append(img)
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                # Fallback to a blank image if loading fails
                imgs.append(Image.new('L', (256, 256), 0))

        # Build mask and keypoints
        # Load mask for the current frame
            mask_path = os.path.join(self.imagedir, mask_ids[-1])
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    mask = np.zeros(imgs[-1].size[::-1], dtype=np.float32)
                    targets = np.array([[0, 0]])
                else:
                    # Use the original GT mask (normalized to 0-1)
                    mask = mask_img.astype(np.float32) / 255.0

                    # Extract target centroids for evaluation/statistics
                    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    targets = []
                    for contour in contours:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            targets.append([cx, cy])
                    targets = np.array(targets) if targets else np.array([[0, 0]])
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                mask = np.zeros(imgs[-1].size[::-1], dtype=np.float32)
                targets = np.array([[0, 0]])

        # Background alignment
        sample_ids = img_ids

        group_homo = []
        # Do not use SIFT alignment here; return identity matrices
        for _ in range(len(img_ids)):
            group_homo.append(np.eye(3, dtype=np.float32))
        group_homo = np.array(group_homo).astype('float32')

        # Trajectory map (placeholder)
        trajectory = np.zeros(imgs[-1].size[::-1], dtype=np.float32)

        # Convert to PIL for transforms
        mask = Image.fromarray(mask)
        trajectory = Image.fromarray(trajectory)

        # Apply transforms
        if self.mode == 'train':
            imgs, mask, trajectory = self._sync_transform(imgs, mask, trajectory)
        elif self.mode == 'test' or self.mode == 'vis':
            imgs, mask, trajectory = self._val_sync_transform(imgs, mask, trajectory)

        mask = np.expand_dims(mask, axis=0).astype('float32')
        trajectory = np.expand_dims(trajectory, axis=0).astype('float32')

        sample_ids = img_ids

        return torch.from_numpy(imgs), torch.from_numpy(np.concatenate([mask, trajectory], axis=0)).clamp(0, 1), torch.from_numpy(group_homo), sample_ids

    def __len__(self):
        return len(self._items)

class TSIRMTSet(Dataset):

    def __init__(self, mode, dataset_dir, folder_id, base_size=256, crop_size=256, channel_size=3, load_single_frame=False, resize_mode='resize'):
        super(TSIRMTSet, self).__init__()

        self.mode = mode
        self.dataset_dir = dataset_dir
        self.imagedir = dataset_dir  # Dataset root directory
        self.folder_id = folder_id
        self.base_size = base_size
        self.crop_size = crop_size
        self.load_single_frame = load_single_frame  # Whether to load only the last frame
        self.resize_mode = resize_mode  # 'resize' or 'padding'
        # Use binary masks only (no Gaussian heatmap option)
        
        self.in_channels = channel_size

        # (Optional) feature extractor for background alignment

        if mode == 'train':
            self.sampling_strategy = 'Interval'
        else:
            self.sampling_strategy = 'Interval'

        self._items, self._mask_paths = [], []
        
        # Build index entries
        if self.mode == 'train':
            for folder in folder_id:
                sequence_dir = os.path.join(self.imagedir, 'images', folder)
                mask_dir = os.path.join(self.imagedir, 'masks', folder)
                
                # Auto-detect image file extensions
                all_files = sorted(os.listdir(sequence_dir))
                img_paths = []
                for f in all_files:
                    if f.endswith(('.png', '.jpg', '.bmp', '.jpeg')):
                        img_paths.append(f)
                
                # Detect mask file extension
                mask_files = sorted(os.listdir(mask_dir)) if os.path.exists(mask_dir) else []
                mask_ext = '.png'  # default
                if mask_files:
                    for mf in mask_files:
                        if mf.endswith('.jpg') or mf.endswith('.jpeg'):
                            mask_ext = '.jpg'
                            break
                        elif mf.endswith('.bmp'):
                            mask_ext = '.bmp'
                            break
                
                # Print detected formats for the first folder
                if folder == folder_id[0]:
                    print(f"  Dataset file format detection ({folder}):")
                    print(f"    Images: {os.path.splitext(img_paths[0])[1] if img_paths else 'N/A'}")
                    print(f"    Masks:  {mask_ext}")
                
                file_num = len(img_paths)
                
                # Sliding-window sampling
                for i in range(channel_size-1, file_num):
                    img_pair, mask_pair = [], []
                    for j in range(i - channel_size + 1, i + 1):
                        img_name = os.path.splitext(img_paths[j])[0]
                        img_pair.append(f"images/{folder}/{img_paths[j]}")
                        mask_pair.append(f"masks/{folder}/{img_name}{mask_ext}")
                    self._items.append(img_pair)
                    self._mask_paths.append(mask_pair)
                    
        elif self.mode == 'test':
            for folder in folder_id:
                sequence_dir = os.path.join(self.imagedir, 'images', folder)
                mask_dir_path = os.path.join(self.imagedir, 'masks', folder)
                
                # Auto-detect image file extensions
                all_files = sorted(os.listdir(sequence_dir))
                img_paths = []
                for f in all_files:
                    if f.endswith(('.png', '.jpg', '.bmp', '.jpeg')):
                        img_paths.append(f)
                
                # Detect mask file extension (use the correct path)
                mask_files = sorted(os.listdir(mask_dir_path)) if os.path.exists(mask_dir_path) else []
                mask_ext = '.png'  # default
                if mask_files:
                    for mf in mask_files:
                        if mf.endswith('.jpg') or mf.endswith('.jpeg'):
                            mask_ext = '.jpg'
                            break
                        elif mf.endswith('.bmp'):
                            mask_ext = '.bmp'
                            break
                
                # Print detected formats for the first folder
                if folder == folder_id[0]:
                    print(f"  Validation file format detection ({folder}):")
                    print(f"    Images: {os.path.splitext(img_paths[0])[1] if img_paths else 'N/A'}")
                    print(f"    Masks:  {mask_ext}")
                
                file_num = len(img_paths)
                
                # In testing, process every frame
                for i in range(0, file_num):
                    img_pair, mask_pair = [], []
                    if i < channel_size - 1:
                        # For the first few frames, repeat the first frame
                        img_name = os.path.splitext(img_paths[0])[0]
                        img_pair = [f"images/{folder}/{img_paths[0]}"] * (channel_size - i - 1)
                        mask_pair = [f"masks/{folder}/{img_name}{mask_ext}"] * (channel_size - i - 1)
                        for j in range(i+1):
                            img_name = os.path.splitext(img_paths[j])[0]
                            img_pair.append(f"images/{folder}/{img_paths[j]}")
                            mask_pair.append(f"masks/{folder}/{img_name}{mask_ext}")
                    else:
                        for j in range(i - channel_size + 1, i + 1):
                            img_name = os.path.splitext(img_paths[j])[0]
                            img_pair.append(f"images/{folder}/{img_paths[j]}")
                            mask_pair.append(f"masks/{folder}/{img_name}{mask_ext}")
                    self._items.append(img_pair)
                    self._mask_paths.append(mask_pair)

    def _sync_transform(self, imgs, mask, trajectory):
        """
        Training-time transform for TSIRMT/IRDST.
        Supports both 'resize' and 'padding' modes.
        """
        num_img = len(imgs)
        
        # Random mirror
        if random.random() < 0.5:
            for i in range(num_img):
                imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            trajectory = trajectory.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.resize_mode == 'padding':
            # Padding mode: pad to a multiple of 32 (200×150 → 224×160)
            padded_imgs = []
            for i in range(num_img):
                padded_img, _, _ = pad_to_multiple_of_32(imgs[i], fill_mode='replicate')
                padded_imgs.append(padded_img)
            
            # Use constant padding for mask/trajectory (0=background)
            mask, _, _ = pad_to_multiple_of_32(mask, fill_mode='constant')
            trajectory, _, _ = pad_to_multiple_of_32(trajectory, fill_mode='constant')
            
            # final transform
            for i in range(num_img):
                padded_imgs[i] = np.array(padded_imgs[i])
            imgs, mask, trajectory = np.array(padded_imgs, dtype=np.float32), np.array(mask, dtype=np.float32), np.array(trajectory, dtype=np.float32)
        
        else:
            # Resize mode: resize to base_size×base_size, then crop to crop_size×crop_size
            # 1. Resize to base_size×base_size
            for i in range(num_img):
                imgs[i] = imgs[i].resize((self.base_size, self.base_size), Image.LANCZOS)
            # Use NEAREST for masks to preserve binariness
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
            trajectory = trajectory.resize((self.base_size, self.base_size), Image.NEAREST)
            
            # 2. Random crop if crop_size < base_size
            if self.crop_size < self.base_size:
                # Random crop
                w, h = imgs[0].size
                x1 = random.randint(0, w - self.crop_size)
                y1 = random.randint(0, h - self.crop_size)
                for i in range(num_img):
                    imgs[i] = imgs[i].crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
                mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
                trajectory = trajectory.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            
            # final transform
            for i in range(num_img):
                imgs[i] = np.array(imgs[i])
            imgs, mask, trajectory = np.array(imgs, dtype=np.float32), np.array(mask, dtype=np.float32), np.array(trajectory, dtype=np.float32)
        
        return imgs, mask, trajectory
    
    def _val_sync_transform(self, imgs, mask, trajectory):
        """
        Validation/testing transform for TSIRMT/IRDST.
        Supports both 'resize' and 'padding' modes.
        """
        num_img = len(imgs)
        
        if self.resize_mode == 'padding':
            # Padding mode: pad to a multiple of 32
            padded_imgs = []
            for i in range(num_img):
                padded_img, _, _ = pad_to_multiple_of_32(imgs[i], fill_mode='replicate')
                padded_imgs.append(padded_img)
            
            # Use constant padding for mask/trajectory (fill=0)
            mask, _, _ = pad_to_multiple_of_32(mask, fill_mode='constant')
            trajectory, _, _ = pad_to_multiple_of_32(trajectory, fill_mode='constant')
            
            # final transform
            for i in range(num_img):
                padded_imgs[i] = np.array(padded_imgs[i])
            imgs, mask, trajectory = np.array(padded_imgs, dtype=np.float32), np.array(mask, dtype=np.float32), np.array(trajectory, dtype=np.float32)
        
        else:
            # Resize mode: resize to base_size×base_size (no crop in validation)
            for i in range(num_img):
                imgs[i] = imgs[i].resize((self.base_size, self.base_size), Image.LANCZOS)
            # Use NEAREST for masks to preserve binariness
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
            trajectory = trajectory.resize((self.base_size, self.base_size), Image.NEAREST)
            
            # final transform
            for i in range(num_img):
                imgs[i] = np.array(imgs[i])
            imgs, mask, trajectory = np.array(imgs, dtype=np.float32), np.array(mask, dtype=np.float32), np.array(trajectory, dtype=np.float32)
        
        return imgs, mask, trajectory

    def __getitem__(self, idx):
        img_ids = self._items[idx]
        mask_ids = self._mask_paths[idx]

        # Load image sequence
        imgs = []
        for img_path in img_ids:
            full_path = os.path.join(self.imagedir, img_path)
            try:
                img = Image.open(full_path).convert('L')
                imgs.append(img)
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                # Fallback to a blank image if loading fails
                imgs.append(Image.new('L', (256, 256), 0))

        # Build mask and keypoints
        # Load mask for the current frame
            mask_path = os.path.join(self.imagedir, mask_ids[-1])
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    mask = np.zeros(imgs[-1].size[::-1], dtype=np.float32)
                    targets = np.array([[0, 0]])
                else:
                    # Use binary mask (normalized to 0-1)
                    mask = mask_img.astype(np.float32) / 255.0
                    
                    # Extract centroids for evaluation
                    targets = []
                    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            targets.append([cx, cy])
                    targets = np.array(targets) if targets else np.array([[0, 0]])
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                mask = np.zeros(imgs[-1].size[::-1], dtype=np.float32)
                targets = np.array([[0, 0]])

        # Background alignment
        sample_ids = img_ids

        group_homo = []
        # Do not use SIFT alignment here; return identity matrices
        for _ in range(len(img_ids)):
            group_homo.append(np.eye(3, dtype=np.float32))
        group_homo = np.array(group_homo).astype('float32')

        # Trajectory map (placeholder)
        trajectory = np.zeros(imgs[-1].size[::-1], dtype=np.float32)

        # Convert to PIL for transforms
        mask = Image.fromarray(mask)
        trajectory = Image.fromarray(trajectory)

        # Apply transforms
        if self.mode == 'train':
            imgs, mask, trajectory = self._sync_transform(imgs, mask, trajectory)
        elif self.mode == 'test' or self.mode == 'vis':
            imgs, mask, trajectory = self._val_sync_transform(imgs, mask, trajectory)

        mask = np.expand_dims(mask, axis=0).astype('float32')
        trajectory = np.expand_dims(trajectory, axis=0).astype('float32')

        sample_ids = img_ids
        
        # Single-frame mode: return only the last frame
        if self.load_single_frame:
            # Return only the last frame and its mask
            imgs_single = imgs[-1:].copy()  # [1, H, W], keep 3D shape
            mask_single = mask  # [1, H, W]
            
            if self.returnpts and self.mode == 'test':
                return torch.from_numpy(imgs_single), torch.from_numpy(mask_single).clamp(0, 1), torch.from_numpy(targets), sample_ids[-1]
            else:
                return torch.from_numpy(imgs_single), torch.from_numpy(mask_single).clamp(0, 1), sample_ids[-1]
        
        # Sequence mode
        return torch.from_numpy(imgs), torch.from_numpy(np.concatenate([mask, trajectory], axis=0)).clamp(0, 1), torch.from_numpy(group_homo), sample_ids

    def __len__(self):
        return len(self._items)

def build_dataloader(mode, args):

    assert mode in ['train', 'test']

    dataset_dir = args.root + '/' + args.dataset

    if args.dataset == 'LMIRSTD':
        # LMIRSTD dataset
        def read_split_txt(name):
            # Prefer reading from ImageSets/
            txt_path = os.path.join(dataset_dir, 'ImageSets', f"{name}.txt")
            if not os.path.isfile(txt_path):
                # Fallback to reading from the dataset root
                txt_path = os.path.join(dataset_dir, f"{name}.txt")
            assert os.path.isfile(txt_path), f"Split file {txt_path} not found"
            with open(txt_path, 'r') as f:
                seqs = [line.strip() for line in f if line.strip()]
            return seqs

        # Fast validation mode
        fast_val = getattr(args, 'fast_val', False)
        if fast_val:
            train_split = 'train_fast'  # 10 sequences
            val_split = 'val_fast'      # 5 sequences
            print(f"  Fast validation mode (LMIRSTD): train={train_split}, val={val_split}")
        else:
            train_split = 'train'
            val_split = 'test'
        
        if mode == 'train':
            folder_ids = read_split_txt(train_split)
        else:  # mode == 'test'
            folder_ids = read_split_txt(val_split)
        
        dataset = BUPTMIRSDTSet(mode=mode,
                               dataset_dir=dataset_dir,
                               folder_id=folder_ids,
                               base_size=args.base_size,
                               crop_size=args.crop_size,
                               channel_size=args.in_channels)
    
    elif args.dataset == 'TSIRMT':
        # Read TSIRMT train/test splits
        def read_split_txt(name):
            # Prefer reading from ImageSets/
            txt_path = os.path.join(dataset_dir, 'ImageSets', f"{name}.txt")
            if not os.path.isfile(txt_path):
                # Fallback to reading from the dataset root
                txt_path = os.path.join(dataset_dir, f"{name}.txt")
            assert os.path.isfile(txt_path), f"Split file {txt_path} not found"
            with open(txt_path, 'r') as f:
                seqs = [line.strip() for line in f if line.strip()]
            return seqs

        # Fast validation mode
        fast_val = getattr(args, 'fast_val', False)
        if fast_val:
            train_split = 'train_fast'  # 35 sequences
            val_split = 'val_fast'      # 15 sequences
            print(f"  Fast validation mode (TSIRMT): train={train_split}, val={val_split}")
        else:
            train_split = 'train'
            val_split = 'test'
        
        if mode == 'train':
            folder_ids = read_split_txt(train_split)
        else:  # mode == 'test'
            folder_ids = read_split_txt(val_split)
 
        # Check whether args has load_single_frame
        load_single_frame = getattr(args, 'load_single_frame', False)
        # Get resize_mode (default: 'resize')
        resize_mode = getattr(args, 'resize_mode', 'resize')
        # use_gaussian_heatmap has been removed; binary masks only
        
        # Reuse TSIRMTSet (same structure)
        dataset = TSIRMTSet(mode=mode,
                           dataset_dir=dataset_dir,
                           folder_id=folder_ids,
                           base_size=args.base_size,
                           crop_size=args.crop_size,
                           channel_size=args.in_channels,
                           load_single_frame=load_single_frame,
                           resize_mode=resize_mode)
    
    elif args.dataset == 'IRDST':
        # IRDST dataset (same structure as TSIRMT)
        def read_split_txt(name):
            # Prefer reading from ImageSets/
            txt_path = os.path.join(dataset_dir, 'ImageSets', f"{name}.txt")
            if not os.path.isfile(txt_path):
                # Fallback to reading from the dataset root
                txt_path = os.path.join(dataset_dir, f"{name}.txt")
            assert os.path.isfile(txt_path), f"Split file {txt_path} not found"
            with open(txt_path, 'r') as f:
                seqs = [line.strip() for line in f if line.strip()]
            return seqs

        if mode == 'train':
            folder_ids = read_split_txt('train')  # use train.txt
        else:  # mode == 'test'
            folder_ids = read_split_txt('test')  # use test.txt for validation
 
        # Check whether args has load_single_frame
        load_single_frame = getattr(args, 'load_single_frame', False)
        # Get resize_mode (default: 'resize')
        resize_mode = getattr(args, 'resize_mode', 'resize')
        
        # Reuse TSIRMTSet (same structure)
        dataset = TSIRMTSet(mode=mode,
                           dataset_dir=dataset_dir,
                           folder_id=folder_ids,
                           base_size=args.base_size,
                           crop_size=args.crop_size,
                           channel_size=args.in_channels,
                           load_single_frame=load_single_frame,
                           resize_mode=resize_mode)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported: IRDST, TSIRMT, LMIRSTD")
    
    batch_size = args.train_batch_size if mode == 'train' else args.test_batch_size
    if mode == 'vis':
        assert batch_size == 1
    shuffle_flag = True if mode == 'train' else False
    drop_last = True if mode == 'train' else False
    loader = DataLoader(dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle_flag, 
            num_workers=args.workers, 
            drop_last=drop_last,
            pin_memory=True)

    return loader