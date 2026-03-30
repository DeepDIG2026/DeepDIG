"""
Descriptor-Based Alignment (DBA) Module.

Computes homography matrix from descriptor matching between frame pairs.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolateSparse2d(nn.Module):
    """Interpolate feature tensor at sparse 2D positions using grid_sample."""
    
    def __init__(self, mode='bicubic', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x, pos, H, W):
        """
        Args:
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] positions (x, y)
            H, W: original resolution for normalization
        
        Returns:
            [B, N, C] sampled features
        """
        grid = 2.0 * (pos / torch.tensor([W-1, H-1], device=pos.device, dtype=pos.dtype)) - 1.0
        grid = grid.unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=self.align_corners)
        return x.permute(0, 2, 3, 1).squeeze(-2)


class DBAModule(nn.Module):
    """
    Descriptor-Based Alignment module.
    
    Computes homography matrix by matching descriptors between source 
    and target frames. Does NOT perform the actual warping.
    
    Args:
        match_threshold: Descriptor matching threshold (default: 0.7)
        min_matches: Minimum matches required for valid homography (default: 10)
        ransac_thresh: RANSAC reprojection threshold in pixels (default: 5.0)
    """
    
    def __init__(self, match_threshold=0.7, min_matches=10, ransac_thresh=5.0):
        super().__init__()
        
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        
        self.interpolator = InterpolateSparse2d('bicubic')
    
    def _sample_descriptors(self, desc, kpts):
        """Sample descriptor vectors at keypoint locations."""
        _, C, H_feat, W_feat = desc.shape
        kpts_scaled = kpts / 8.0
        feats = self.interpolator(desc, kpts_scaled.unsqueeze(0), H_feat, W_feat)
        return F.normalize(feats.squeeze(0), dim=-1)
    
    def _mutual_nn_match(self, feats_src, feats_tgt):
        """Mutual nearest neighbor matching with ratio test."""
        similarity = torch.mm(feats_src, feats_tgt.t())
        
        max1, idx1 = similarity.max(dim=1)
        max2, idx2 = similarity.max(dim=0)
        
        n_src = len(feats_src)
        indices = torch.arange(n_src, device=feats_src.device)
        mutual_mask = (idx2[idx1] == indices)
        score_mask = (max1 > self.match_threshold)
        
        valid_mask = mutual_mask & score_mask
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        
        return valid_idx, idx1[valid_idx]
    
    def _estimate_homography(self, pts_src, pts_tgt):
        """Estimate homography using RANSAC."""
        pts_src_np = pts_src.cpu().numpy()
        pts_tgt_np = pts_tgt.cpu().numpy()
        
        H, mask = cv2.findHomography(pts_src_np, pts_tgt_np, cv2.RANSAC, self.ransac_thresh)
        
        if H is None:
            return None, 0
        
        try:
            det = np.linalg.det(H[:2, :2])
            if abs(det) < 1e-6:
                return None, 0
            np.linalg.inv(H)
        except (np.linalg.LinAlgError, ValueError):
            return None, 0
        
        n_inliers = int(mask.sum()) if mask is not None else 0
        return H, n_inliers
    
    def compute_homography(self, src_desc, src_kpts, tgt_desc, tgt_kpts):
        """
        Compute homography from source to target frame.
        
        Args:
            src_desc: Source frame descriptors [1, 64, H/8, W/8]
            src_kpts: Source frame keypoints [N, 2]
            tgt_desc: Target frame descriptors [1, 64, H/8, W/8]
            tgt_kpts: Target frame keypoints [M, 2]
        
        Returns:
            dict containing:
                - H: Homography matrix [3, 3] numpy array, or None if failed
                - valid: Whether homography is valid
                - n_matches: Number of valid matches
                - n_inliers: Number of RANSAC inliers
        """
        device = tgt_desc.device
        src_desc = src_desc.to(device)
        src_kpts = src_kpts.to(device)
        
        if len(src_kpts) < self.min_matches or len(tgt_kpts) < self.min_matches:
            return {'H': None, 'valid': False, 'n_matches': 0, 'n_inliers': 0}
        
        feats_src = self._sample_descriptors(src_desc, src_kpts)
        feats_tgt = self._sample_descriptors(tgt_desc, tgt_kpts)
        
        src_idx, tgt_idx = self._mutual_nn_match(feats_src, feats_tgt)
        n_matches = len(src_idx)
        
        if n_matches < self.min_matches:
            return {'H': None, 'valid': False, 'n_matches': n_matches, 'n_inliers': 0}
        
        pts_src = src_kpts[src_idx]
        pts_tgt = tgt_kpts[tgt_idx]
        
        H, n_inliers = self._estimate_homography(pts_src, pts_tgt)
        
        return {
            'H': H,
            'valid': H is not None,
            'n_matches': n_matches,
            'n_inliers': n_inliers
        }
    
    def compute_homography_batch(self, src_desc_list, src_kpts_list, tgt_desc, tgt_kpts):
        """
        Compute homographies for multiple source frames to single target.
        
        Args:
            src_desc_list: List of source descriptors, each [1, 64, H/8, W/8]
            src_kpts_list: List of source keypoints, each [N, 2]
            tgt_desc: Target frame descriptors [1, 64, H/8, W/8]
            tgt_kpts: Target frame keypoints [M, 2]
        
        Returns:
            List of homography result dicts
        """
        results = []
        for src_desc, src_kpts in zip(src_desc_list, src_kpts_list):
            result = self.compute_homography(src_desc, src_kpts, tgt_desc, tgt_kpts)
            results.append(result)
        return results
