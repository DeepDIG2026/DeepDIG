"""
Frame Alignment Manager.

Manages frame caching, calls DBA for homography computation, 
performs warping, and builds aligned sequences.
"""

import torch
import torch.nn as nn
from collections import deque
from kornia.geometry import warp_perspective

from .DBA_module import DBAModule


class FrameCache:
    """
    Simple frame cache for storing historical frames and their descriptors.
    
    Args:
        max_size: Maximum number of frames to cache (default: 20)
    """
    
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.frames = deque(maxlen=max_size)
        self.descriptors = deque(maxlen=max_size)
        self.keypoints = deque(maxlen=max_size)
    
    def add(self, frame, descriptor, keypoints):
        """
        Add a frame to cache.
        
        Args:
            frame: [H, W] or [1, H, W] tensor
            descriptor: [1, 64, H/8, W/8] tensor
            keypoints: [N, 2] tensor
        """
        if frame.dim() == 2:
            frame = frame.unsqueeze(0)
        
        self.frames.append(frame.detach().cpu())
        self.descriptors.append(descriptor.detach().cpu())
        self.keypoints.append(keypoints.detach().cpu())
    
    def __len__(self):
        return len(self.frames)
    
    def clear(self):
        self.frames.clear()
        self.descriptors.clear()
        self.keypoints.clear()


class FrameAlignmentManager(nn.Module):
    """
    Manages frame alignment using DBA module.
    
    Handles caching, homography computation via DBA, warping, 
    and aligned sequence construction.
    
    Args:
        window_size: Number of frames in output sequence (default: 20)
        match_threshold: DBA matching threshold (default: 0.7)
        min_matches: Minimum matches for valid alignment (default: 10)
    """
    
    def __init__(self, window_size=20, match_threshold=0.7, min_matches=10):
        super().__init__()
        
        self.window_size = window_size
        
        self.dba = DBAModule(
            match_threshold=match_threshold,
            min_matches=min_matches
        )
        
        self.cache = FrameCache(max_size=window_size)
        self.use_cache = False
    
    def enable_cache(self):
        self.use_cache = True
    
    def disable_cache(self):
        self.use_cache = False
    
    def reset_cache(self):
        self.cache.clear()
    
    def _warp_frame(self, frame, H, target_size):
        """Warp frame using homography matrix."""
        device = frame.device
        H_tensor = torch.from_numpy(H).float().unsqueeze(0).to(device)
        
        if frame.dim() == 2:
            frame = frame.unsqueeze(0).unsqueeze(0)
        elif frame.dim() == 3:
            frame = frame.unsqueeze(0)
        
        warped = warp_perspective(frame, H_tensor, target_size)
        return warped.squeeze(0).squeeze(0)
    
    def align_sequence(self, frames, desc_list, kpts_list):
        """
        Align a batch sequence to its last frame.
        
        Args:
            frames: [B, T, H, W] input sequence
            desc_list: List of T descriptors per batch, each [1, 64, H/8, W/8]
            kpts_list: List of T keypoints per batch, each [N, 2]
        
        Returns:
            aligned: [B, T, H, W] aligned sequence
            stats: dict with alignment statistics
        """
        B, T, H, W = frames.shape
        device = frames.device
        
        aligned_batch = []
        total_matches = 0
        total_failed = 0
        
        for b in range(B):
            # Target is last frame
            tgt_desc = desc_list[b * T + T - 1]
            tgt_kpts = kpts_list[b * T + T - 1]
            
            aligned_frames = []
            
            for t in range(T - 1):
                idx = b * T + t
                src_desc = desc_list[idx]
                src_kpts = kpts_list[idx]
                src_frame = frames[b, t]
                
                result = self.dba.compute_homography(
                    src_desc, src_kpts, tgt_desc, tgt_kpts
                )
                
                if result['valid']:
                    try:
                        aligned = self._warp_frame(src_frame, result['H'], (H, W))
                        total_matches += result['n_matches']
                    except RuntimeError:
                        aligned = src_frame
                        total_failed += 1
                else:
                    aligned = src_frame
                    total_failed += 1
                
                aligned_frames.append(aligned)
            
            # Last frame unchanged
            aligned_frames.append(frames[b, T - 1])
            aligned_batch.append(torch.stack(aligned_frames))
        
        aligned = torch.stack(aligned_batch)
        
        stats = {
            'avg_matches': total_matches / max(1, B * (T - 1) - total_failed),
            'n_failed': total_failed
        }
        
        return aligned, stats
    
    def align_with_cache(self, current_frame, current_desc, current_kpts):
        """
        Align cached frames to current frame (inference mode).
        
        Args:
            current_frame: [H, W] current frame
            current_desc: [1, 64, H/8, W/8] current descriptors
            current_kpts: [N, 2] current keypoints
        
        Returns:
            sequence: [T, H, W] aligned sequence
            stats: dict with alignment statistics
        """
        device = current_frame.device
        
        if current_frame.dim() == 3:
            current_frame = current_frame.squeeze(0)
        H, W = current_frame.shape
        
        aligned_frames = []
        total_matches = 0
        n_aligned = 0
        
        # Align cached frames to current
        for i in range(len(self.cache)):
            hist_frame = self.cache.frames[i].to(device)
            hist_desc = self.cache.descriptors[i]
            hist_kpts = self.cache.keypoints[i]
            
            if hist_frame.dim() == 3:
                hist_frame = hist_frame.squeeze(0)
            
            result = self.dba.compute_homography(
                hist_desc, hist_kpts, current_desc, current_kpts
            )
            
            if result['valid']:
                try:
                    aligned = self._warp_frame(hist_frame, result['H'], (H, W))
                    total_matches += result['n_matches']
                    n_aligned += 1
                except RuntimeError:
                    aligned = hist_frame
            else:
                aligned = hist_frame
            
            if aligned.dim() == 3:
                aligned = aligned.squeeze(0)
            aligned_frames.append(aligned)
        
        # Add current frame
        aligned_frames.append(current_frame)
        
        # Pad if needed
        while len(aligned_frames) < self.window_size:
            aligned_frames.insert(0, aligned_frames[0].clone())
        
        # Take last window_size frames
        sequence = torch.stack(aligned_frames[-self.window_size:])
        
        # Update cache
        self.cache.add(current_frame, current_desc, current_kpts)
        
        stats = {
            'avg_matches': total_matches / max(1, n_aligned),
            'n_aligned': n_aligned
        }
        
        return sequence, stats


__all__ = ['FrameAlignmentManager', 'FrameCache']
