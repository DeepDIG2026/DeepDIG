"""
Evaluation metrics module. Configuration parameters are defined in MetricConfig (lines 9-38).
"""

import cv2
import imutils

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, auc
from skimage import measure

class MetricConfig:
    """Evaluation metric configuration (centralized thresholds)."""
    
    # Binarization thresholds
    DETECTION_THRESHOLD = 0.5    # heatmap -> binary mask (affects PD/FA/mIoU)
    IOU_THRESHOLD = 0.5          # mIoU threshold
    
    # Keypoint extraction
    MIN_CONTOUR_AREA = 1         # minimum connected-component area (pixels)
    
    # PD/FA radii (both set to 3 pixels)
    LOCLEN1 = 3                  # PD search radius (pixels) - within GT area ±3
    LOCLEN2 = 3                  # FA exclusion radius (pixels) - within GT center ±3
    
    # Distance threshold (3 pixels)
    DISTANCE_THRESHOLD = 3       # P/R/F1 distance threshold (default 3 pixels)
    
    # Note: CLI-configurable argument
    # --dthres 3 : P/R/F1 distance threshold (default 3 pixels)

def get_keypoints(featmap, min_area=None):
    """Optimized keypoint extraction.
    
    Uses skimage label + regionprops to more reliably handle tiny connected components.
    
    Args:
        featmap: feature map (H×W)
        min_area: minimum connected-component area threshold
    Returns:
        keypoints: keypoints list [[x1,y1], [x2,y2], ...]
    """
    if min_area is None:
        min_area = MetricConfig.MIN_CONTOUR_AREA
        
    fmap = featmap.copy()
    
    # Ensure the input is a 2D numpy array
    while len(fmap.shape) > 2:
        if fmap.shape[0] == 1:
            fmap = fmap.squeeze(0)
        else:
            fmap = fmap[0]
    
    # Ensure dtype
    if fmap.dtype != np.float32:
        fmap = fmap.astype(np.float32)
    
    # Binarize using configured threshold
    threshold = MetricConfig.DETECTION_THRESHOLD
    binary_map = (fmap >= threshold).astype(np.uint8)
    
    # Final check: must be 2D
    if len(binary_map.shape) != 2:
        raise ValueError(f"无法将输入转换为2D数组，当前形状: {binary_map.shape}")
    
    # Connected component analysis with skimage label (more robust)
    from skimage import measure
    labeled_map = measure.label(binary_map, connectivity=2)
    regions = measure.regionprops(labeled_map)
    
    res = []
    for region in regions:
        # Area filter
        if region.area < min_area:
            continue
        
        # Centroid: (row, col) -> (y, x)
        centroid_y, centroid_x = region.centroid
        
        # Convert to integer coordinates (x, y)
        cX = int(round(centroid_x))
        cY = int(round(centroid_y))
        
        # Clamp to image bounds
        h, w = binary_map.shape
        cX = max(0, min(w-1, cX))
        cY = max(0, min(h-1, cY))
        
        res.append([cX, cY])
    
    return res

def distance(x, y):
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2

def compute_prfa(pred, gt, th):
    """Optimized Precision/Recall/False Alarm computation.
    Args:
        pred: predicted keypoints [[x1,y1], [x2,y2], ...]
        gt: ground-truth keypoints [[x1,y1], [x2,y2], ...]
        th: distance threshold
    """
    P, G = len(pred), len(gt)

    if P == 0 and G == 0:        
        return [1.0, 1.0, 0]
    elif P == 0:        
        return [0.0, 0.0, 0]
    elif G == 0:        
        return [0.0, 0.0, P]  # fix: return actual false-alarm count
    else:
        # Use a more precise matching strategy
        matched_pred = set()
        matched_gt = set()
        
        # For each GT point, find the nearest predicted point
        for g in range(G):
            min_dist = float('inf')
            best_pred = -1
            for p in range(P):
                if p in matched_pred:
                    continue
                dist = np.sqrt(distance(pred[p], gt[g]))
                if dist <= th and dist < min_dist:
                    min_dist = dist
                    best_pred = p
            
            if best_pred != -1:
                matched_pred.add(best_pred)
                matched_gt.add(g)
        
        # Compute metrics
        tp_precision = len(matched_pred)  # correctly matched predictions
        tp_recall = len(matched_gt)       # correctly matched GT points
        
        precision = tp_precision / P if P > 0 else 0.0
        recall = tp_recall / G if G > 0 else 0.0
        falsealarm = P - tp_precision
        
        return [precision, recall, falsealarm]

def calculateF1Measure(output_image,gt_image,thre=0.5):
    # import pdb; pdb.set_trace()
    output_image = 1/(1 + np.exp(-output_image))
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return [prec, recall, F1]

def compute_bsf(data, pred):
    fmap = pred.copy()
    
    # fmap[fmap > 0] = 255
    fmap[fmap < 0] = 0
    if fmap.max() == 0:
        return 0
    else:
        normalize_pred = (fmap - fmap.min()) / (fmap.max() - fmap.min())
        return np.std(data) / np.std(normalize_pred)

def clamp(bottom, top, low, high):
    out1, out2 = bottom, top
    if bottom < low:
        out1 = low
    if top > high:
        out2 = high
    return out1, out2

def compute_cg(data, pred, gtkey):
    target_kernel = 3
    background_kernel = 23
    target_d = target_kernel // 2
    background_d = background_kernel // 2
    fmap = pred.copy()
    fmap[fmap < 0] = 0
    if fmap.max() == 0:
        return 0, 0
    else:
        normalize_pred = (fmap - fmap.min()) / (fmap.max() - fmap.min())
    cg1 = 0
    cg2 = 0
    for k in gtkey:
        ty1, ty2 = clamp(k[1]-target_d, k[1]+target_d+1, 0, data.shape[0])
        tx1, tx2 = clamp(k[0]-target_d, k[0]+target_d+1, 0, data.shape[1])
        by1, by2 = clamp(k[1]-background_d, k[1]+background_d+1, 0, data.shape[0])
        bx1, bx2 = clamp(k[0]-background_d, k[0]+background_d+1, 0, data.shape[1])
        input_target_area = data[ty1:ty2, tx1:tx2]
        input_background_area = data[by1:by2, bx1:bx2]
        output_target_area = normalize_pred[ty1:ty2, tx1:tx2]
        output_background_area = normalize_pred[by1:by2, bx1:bx2]
        cg1 += np.abs(output_target_area.max() - output_background_area.mean()) / np.abs(input_target_area.max() - input_background_area.mean())
        # cg2 += np.abs(output_target_area.mean() - output_background_area.mean()) / np.abs(input_target_area.mean() - input_background_area.mean())
        input_background_area[ty1:ty2, tx1:tx2] = 0
        output_background_area[ty1:ty2, tx1:tx2] = 0
        ibgmean = np.sum(input_background_area) / (background_kernel * background_kernel - target_kernel * target_kernel)
        obgmean = np.sum(output_background_area) / (background_kernel * background_kernel - target_kernel * target_kernel)
        # cg1 += np.abs(output_target_area.max() - obgmean) / np.abs(input_target_area.max() - ibgmean)
        cg2 += np.abs(output_target_area.mean() - obgmean) / np.abs(input_target_area.mean() - ibgmean)
    if len(gtkey) != 0:
        cg1 /= len(gtkey)
        cg2 /= len(gtkey)
    return cg1, cg2

def compute_pd_fa_with_threshold(pred, gt_mask, size, detect_th):
    """Compute PD and FA at a given threshold (without changing global config).
    Args:
        pred: predicted heatmap (numpy array)
        gt_mask: GT binary mask (numpy array)
        size: image size (H, W)
        detect_th: detection threshold
    Returns:
        pd_score: PD score
        fa_score: FA score
    """
    try:
        # Ensure numpy arrays
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
            
        # Binarize with the given threshold
        output_one = pred.copy()
        target_one = gt_mask.copy()
        
        # Prediction binarization
        output_one[np.where(output_one < detect_th)] = 0
        output_one[np.where(output_one >= detect_th)] = 1
        
        # GT binarization
        target_one[np.where(target_one < detect_th)] = 0
        target_one[np.where(target_one >= detect_th)] = 1
        
        # Ensure correct dtype
        target_one = target_one.astype(np.int32)
        output_one = output_one.astype(np.int32)
        
        # Connected component analysis
        labelimage = measure.label(target_one, connectivity=2)
        props = measure.regionprops(labelimage, intensity_image=target_one, cache=True)
        
        TgtNum = len(props)
        TrueNum = 0
        FalseNum = 0
        
        # Handle edge cases
        if TgtNum == 0:
            if np.sum(output_one) == 0:
                return 1.0, 0.0  # no GT target and no prediction
            else:
                fa_score = float(np.sum(output_one) / (size[0] * size[1]))
                return 0.0, fa_score  # no GT target but has prediction
        
        # PD computation
        LocLen1 = MetricConfig.LOCLEN1
        LocLen2 = MetricConfig.LOCLEN2
        
        for i_tgt in range(len(props)):
            True_flag = 0
            pixel_coords = props[i_tgt].coords
            
            # For each GT pixel, check whether any prediction exists within LocLen1
            for i_pixel in pixel_coords:
                r_min = max(0, i_pixel[0] - LocLen1)
                r_max = min(output_one.shape[0], i_pixel[0] + LocLen1 + 1)
                c_min = max(0, i_pixel[1] - LocLen1)
                c_max = min(output_one.shape[1], i_pixel[1] + LocLen1 + 1)
                
                Tgt_area = output_one[r_min:r_max, c_min:c_max]
                if Tgt_area.sum() >= 1:
                    True_flag = 1
                    break
            
            if True_flag == 1:
                TrueNum += 1
        
        # FA computation - exclusion region
        Box2_map = np.ones(output_one.shape)
        for i_tgt in range(len(props)):
            pixel_coords = props[i_tgt].coords
            for i_pixel in pixel_coords:
                r_min = max(0, i_pixel[0] - LocLen2)
                r_max = min(output_one.shape[0], i_pixel[0] + LocLen2 + 1)
                c_min = max(0, i_pixel[1] - LocLen2) 
                c_max = min(output_one.shape[1], i_pixel[1] + LocLen2 + 1)
                Box2_map[r_min:r_max, c_min:c_max] = 0
        
        # FA computation
        False_output_one = output_one * Box2_map
        FalseNum = np.count_nonzero(False_output_one)
        
        # Final metrics
        pd_score = TrueNum / TgtNum if TgtNum > 0 else 0.0
        fa_score = FalseNum / (size[0] * size[1])
        
        return float(pd_score), float(fa_score)
        
    except Exception as e:
        return 0.0, 0.0

def compute_auc(pred_map, gt_mask):
    """Compute PD-FA curve AUC based on metrics.py (reference-aligned implementation).
    
    Compute PD-FA AUC using ShootingRules logic:
    - PD (Probability of Detection) = TrueNum / TgtNum
    - FA (False Alarm Rate) = FalseNum / pixelsNumber
    - AUC = auc(FA_array, PD_array)
    
    Args:
        pred_map: predicted heatmap (H×W), values in [0, 1]
        gt_mask: GT binary mask (H×W), values 0/1
    Returns:
        auc_score: area under the PD-FA curve
    """
    try:
        # Ensure numpy arrays
        if isinstance(pred_map, torch.Tensor):
            pred_map = pred_map.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        # Define threshold sequence (consistent with metrics.py)
        Th_Seg = np.array([0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 
                          0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
                          0.8, 0.85, 0.9, 1])
        
        PD_array = []
        FA_array = []
        pixelsNumber = pred_map.shape[0] * pred_map.shape[1]
        
        # Compute PD/FA for each threshold (ShootingRules logic)
        for DetectTh in Th_Seg:
            FalseNum, TrueNum, TgtNum = compute_shooting_rules_single(pred_map, gt_mask, DetectTh)
            
            # Compute PD and FA
            if TgtNum > 0:
                pd_score = TrueNum / TgtNum
            else:
                pd_score = 0.0
                
            fa_score = FalseNum / pixelsNumber
            
            PD_array.append(pd_score)
            FA_array.append(fa_score)
        
        # Convert to numpy arrays
        PD_array = np.array(PD_array)
        FA_array = np.array(FA_array)
        
        # Ensure FA is increasing (required for AUC)
        sorted_indices = np.argsort(FA_array)
        FA_sorted = FA_array[sorted_indices]
        PD_sorted = PD_array[sorted_indices]
        
        # Edge-case handling (after sorting to avoid conflicts)
        unique_pd = np.unique(PD_sorted)
        unique_fa = np.unique(FA_sorted)

        # 1) PD all zeros: complete miss, AUC=0
        if len(unique_pd) == 1 and unique_pd[0] == 0:
            return 0.0

        # 2) PD all ones: perfect detection, AUC=1
        if len(unique_pd) == 1 and unique_pd[0] == 1:
            return 1.0

        # 3) FA constant or PD constant (not 0/1): degenerate curve
        if len(unique_fa) < 2 or len(unique_pd) < 2:
            # Degenerate curve: return 0.5 as a random-guess baseline.
            # This is more reasonable than 0.0 (which indicates complete failure).
            return 0.5
        
        # Compute AUC (consistent with metrics.py)
        try:
            
            # Trapezoidal integration for AUC
            auc_score = auc(FA_sorted, PD_sorted)
            
            # Clamp tiny negative values due to floating-point errors
            if auc_score < 0:
                auc_score = 0.0
            # AUC should not exceed 1
            elif auc_score > 1:
                auc_score = 1.0
                
        except Exception as e:
            print(f"AUC计算异常: {e}")
            return 0.0
        
        return float(auc_score)
        
    except Exception as e:
        print(f"PD-FA AUC计算错误: {e}")
        return 0.5

def compute_shooting_rules_single(pred_map, gt_mask, DetectTh, debug_seq_name=None):
    """Single-frame ShootingRules computation (aligned with metrics.py ShootingRules.forward).
    
    Args:
        pred_map: predicted heatmap (H×W)
        gt_mask: GT mask (H×W)
        DetectTh: detection threshold
        debug_seq_name: debug sequence name (optional)
    Returns:
        FalseNum: false alarm count
        TrueNum: true detection count
        TgtNum: target count
    """
    FalseNum = 0
    TrueNum = 0
    TgtNum = 0
    
    # Copy inputs
    output_one = pred_map.copy()
    target_one = gt_mask.copy()
    
    # Debug: basic stats of the GT mask
    if debug_seq_name and DetectTh == 0.5:  # print only at 0.5 to avoid excessive logs
        gt_unique = np.unique(target_one)
        gt_nonzero = np.count_nonzero(target_one)
        print(f"    [DEBUG {debug_seq_name}] GT shape={target_one.shape}, "
              f"unique_vals={gt_unique}, nonzero_count={gt_nonzero}, "
              f"min={target_one.min():.3f}, max={target_one.max():.3f}")
    
    # Prediction binarization (consistent with metrics.py lines 35-36)
    output_one[np.where(output_one < DetectTh)] = 0
    output_one[np.where(output_one >= DetectTh)] = 1
    
    # Key fix: binarize GT mask (this may be the root cause)
    # Ensure target_one is binary (0/1)
    if target_one.max() > 1.0:
        # If GT is in 0-255, normalize first
        target_one = target_one / 255.0
    
    # Binarize GT (values > 0.5 are treated as target)
    target_one = (target_one > 0.5).astype(np.float32)
    
    # Connected component analysis (consistent with metrics.py lines 38-39)
    labelimage = measure.label(target_one.astype(np.uint8), connectivity=2)
    props = measure.regionprops(labelimage, intensity_image=target_one, cache=True)
    
    TgtNum = len(props)
    
    if TgtNum == 0:
        # When there is no target, all predictions are false alarms
        FalseNum = np.count_nonzero(output_one)
        return FalseNum, TrueNum, TgtNum
    
    # Parameter setup (use unified config for consistency)
    LocLen1 = MetricConfig.LOCLEN1  # detection radius (from config)
    LocLen2 = MetricConfig.LOCLEN2  # exclusion radius (from config)
    
    # Initialize exclusion-region map
    Box2_map = np.ones(output_one.shape)
    
    # Process each target region (consistent with metrics.py lines 48-58)
    for i_tgt in range(len(props)):
        True_flag = 0
        
        pixel_coords = props[i_tgt].coords
        for i_pixel in pixel_coords:
            # Set exclusion region (for FA computation)
            r_min = max(0, i_pixel[0] - LocLen2)
            r_max = min(output_one.shape[0], i_pixel[0] + LocLen2 + 1)
            c_min = max(0, i_pixel[1] - LocLen2) 
            c_max = min(output_one.shape[1], i_pixel[1] + LocLen2 + 1)
            Box2_map[r_min:r_max, c_min:c_max] = 0
            
            # Detection region (for PD computation)
            r_min_detect = max(0, i_pixel[0] - LocLen1)
            r_max_detect = min(output_one.shape[0], i_pixel[0] + LocLen1 + 1)
            c_min_detect = max(0, i_pixel[1] - LocLen1)
            c_max_detect = min(output_one.shape[1], i_pixel[1] + LocLen1 + 1)
            
            Tgt_area = output_one[r_min_detect:r_max_detect, c_min_detect:c_max_detect]
            if Tgt_area.sum() >= 1:
                True_flag = 1
        
        if True_flag == 1:
            TrueNum += 1
    
    # Count false alarms (consistent with metrics.py lines 60-61)
    False_output_one = output_one * Box2_map
    FalseNum = np.count_nonzero(False_output_one)
    
    return FalseNum, TrueNum, TgtNum

def compute_roc_auc_manual(y_scores, y_true):
    """Manual ROC-AUC computation.
    
    Args:
        y_scores: prediction score array
        y_true: ground-truth label array (0/1)
    Returns:
        auc: AUC value
    """
    # Get all unique thresholds (including boundaries)
    thresholds = np.unique(y_scores)
    # Add boundary thresholds so TPR/FPR cover [0, 1]
    thresholds = np.concatenate([thresholds, [thresholds.min() - 1e-8, thresholds.max() + 1e-8]])
    thresholds = np.sort(thresholds)[::-1]  # descending
    
    # Compute TPR/FPR for each threshold
    tpr_list = []
    fpr_list = []
    
    # Total positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    for threshold in thresholds:
        # Samples predicted as positive
        y_pred = (y_scores >= threshold).astype(int)
        
        # Confusion matrix
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Convert to numpy arrays
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)
    
    # Ensure FPR is increasing (required for AUC)
    sorted_indices = np.argsort(fpr_array)
    fpr_sorted = fpr_array[sorted_indices]
    tpr_sorted = tpr_array[sorted_indices]
    
    # Trapezoidal rule for AUC
    auc_value = auc(fpr_sorted, tpr_sorted)
    
    return float(auc_value)

def compute_pd_fa_curve(pred_map, gt_mask):
    """Return (FA, PD) arrays over the full threshold sequence for visualization.

    Uses the exact same thresholds, radii, and logic as compute_auc,
    but returns curve points only (does not compute AUC).
    """
    # Ensure numpy arrays
    if isinstance(pred_map, torch.Tensor):
        pred_map = pred_map.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    Th_Seg = np.array([
        0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
        1e-1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
        0.7, 0.75, 0.8, 0.85, 0.9, 1
    ])

    PD_array = []
    FA_array = []
    pixelsNumber = pred_map.shape[0] * pred_map.shape[1]

    for DetectTh in Th_Seg:
        FalseNum, TrueNum, TgtNum = compute_shooting_rules_single(
            pred_map, gt_mask, DetectTh)
        pd_score = TrueNum / TgtNum if TgtNum > 0 else 0.0
        fa_score = FalseNum / pixelsNumber
        PD_array.append(pd_score)
        FA_array.append(fa_score)

    return np.array(FA_array), np.array(PD_array)


def save_pd_fa_curve(pred_map, gt_mask, save_path, *, figsize=(6, 5), dpi=300):
    """Save the PD-FA curve to the given path.

    Args:
        pred_map (ndarray | Tensor): predicted heatmap
        gt_mask  (ndarray | Tensor): GT mask
        save_path (str): file path including filename, e.g. '/tmp/roc.png'
    """
    import matplotlib
    matplotlib.use('Agg')  # non-GUI backend
    import matplotlib.pyplot as plt
    FA, PD = compute_pd_fa_curve(pred_map, gt_mask)

    # Sort by FA ascending (left-to-right)
    idx = np.argsort(FA)
    FA_sorted, PD_sorted = FA[idx], PD[idx]

    plt.figure(figsize=figsize)
    plt.plot(FA_sorted, PD_sorted, marker='o', linewidth=1.2)
    plt.xlabel('FA (False Alarm Rate)')
    plt.ylabel('PD (Probability of Detection)')
    plt.title('PD-FA Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    
    return save_path

def compute_metric(data, pred, predkey, gtkey, metric, dthres=3, gt_mask=None):

    if metric == 'Precision':
        pr, _, _ = compute_prfa(predkey, gtkey, dthres)
        # pr, _, _ = calculateF1Measure(pred, gt)
        return pr
    elif metric == 'Recall':
        _, re, _ = compute_prfa(predkey, gtkey, dthres)
        # _, re, _ = calculateF1Measure(pred, gt)
        return re
    elif metric == 'F1':
        pr, re, _ = compute_prfa(predkey, gtkey, dthres)
        # pr, re, _ = calculateF1Measure(pred, gt)
        if pr + re == 0:
            return 0
        else:
            return 2.0 * pr * re / (pr + re)
    elif metric == 'FalseAlarm':
        _, _, fa = compute_prfa(predkey, gtkey, dthres)
        return fa
    elif metric == "BSF":
        bsf = compute_bsf(data, pred)
        return bsf
    elif metric == 'CG1':
        cg, _ = compute_cg(data, pred, gtkey)
        return cg
    elif metric == 'CG2':
        _, cg = compute_cg(data, pred, gtkey)
        return cg
    elif metric == 'AUC':
        if gt_mask is None:
            return 0.5  # default when GT mask is missing
        return compute_auc(pred, gt_mask)
    elif metric == 'mIoU':
        if gt_mask is None:
            return 0.0  # default when GT mask is missing
        return compute_miou(pred, gt_mask)
    elif metric == 'PD':
        if gt_mask is None:
            return 0.0  # default when GT mask is missing
        size = (pred.shape[0], pred.shape[1]) if len(pred.shape) >= 2 else (256, 256)
        pd_score, _ = compute_pd_fa(pred, gt_mask, size)
        return pd_score
    elif metric == 'FA':
        if gt_mask is None:
            return 0.0  # default when GT mask is missing
        size = (pred.shape[0], pred.shape[1]) if len(pred.shape) >= 2 else (256, 256)
        _, fa_score = compute_pd_fa(pred, gt_mask, size)
        return fa_score

def compute_batch_prfa(pred, gt, th, reduction="mean"):

    batch_size = pred.shape[0]

    precision = [0] * batch_size
    recall = [0] * batch_size
    falsealarm = [0] * batch_size
    for i in range(batch_size):
        predkey = get_keypoints(pred[i])
        gtkey = get_keypoints(gt[i])
        precision[i], recall[i], falsealarm[i] = compute_prfa(predkey, gtkey, th)
    
    if reduction == "mean":
        return [np.mean(precision), np.mean(recall), np.mean(falsealarm)]
    elif reduction == "sum":
        return [np.sum(precision), np.sum(recall), np.sum(falsealarm)]
    else:
        return [precision, recall, falsealarm]

class mIoU():
    
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0
    def update(self, preds, labels, size):
        # Convert to numpy first, then process
        if isinstance(preds, torch.Tensor):
            # If logits are given, apply sigmoid before binarization
            preds_np = torch.sigmoid(preds).cpu().numpy()
        else:
            preds_np = preds
            
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels
        
        # Handle 4D tensor [B, C, H, W] -> [H, W]
        while len(preds_np.shape) > 2:
            preds_np = preds_np.squeeze(0)
        while len(labels_np.shape) > 2:
            labels_np = labels_np.squeeze(0)
        
        # Binarize (threshold 0.5)
        predits = (preds_np > 0.5).astype('int64')
        labelss = (labels_np > 0.5).astype('int64')
        
        # Ensure 2D array
        if len(predits.shape) != 2 or len(labelss.shape) != 2:
            raise ValueError(f"Expected 2D arrays, got predits shape {predits.shape}, labelss shape {labelss.shape}")

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        true_img = np.zeros(predits.shape)
        matched_image_indices = set()  # track already matched predicted regions
        
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                if m in matched_image_indices:  # skip matched regions
                    continue
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1
                    matched_image_indices.add(m)  # mark as matched
                    break

        self.dismatch_pixel += (predits - true_img).sum()
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD / self.target if self.target > 0 else 0.0
        
        # Ensure FA is float
        if hasattr(Final_FA, 'cpu'):
            Final_FA = float(Final_FA.cpu().detach().numpy())
        else:
            Final_FA = float(Final_FA)
            
        return Final_PD, Final_FA

    def reset(self):
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

def batch_pix_accuracy(output, target):
    # Ensure output is a tensor
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output).float()
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target).float()
        
    # Normalize dimensions: ensure 4D [B, C, H, W]
    if len(target.shape) == 2:  # [H, W]
        target = target.unsqueeze(0).unsqueeze(0)
    elif len(target.shape) == 3:  # [B, H, W] or [C, H, W]
        target = target.unsqueeze(1) if target.shape[0] == 1 else target.unsqueeze(0)
    
    if len(output.shape) == 2:
        output = output.unsqueeze(0).unsqueeze(0)
    elif len(output.shape) == 3:
        output = output.unsqueeze(1) if output.shape[0] == 1 else output.unsqueeze(0)
    
    target = target.float()
    output = output.float()

    assert output.shape == target.shape, f"Predict and Label Shape Don't Match: {output.shape} vs {target.shape}"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target):
    mini = 1
    maxi = 2  # fix: range should be 1..2 to count pixels with value==1 only
    nbins = 1
    
    # Ensure output is a tensor
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output).float()
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target).float()
        
    # Normalize dimensions: ensure 4D [B, C, H, W]
    if len(target.shape) == 2:  # [H, W]
        target = target.unsqueeze(0).unsqueeze(0)
    elif len(target.shape) == 3:  # [B, H, W] or [C, H, W]
        target = target.unsqueeze(1) if target.shape[0] == 1 else target.unsqueeze(0)
    
    if len(output.shape) == 2:
        output = output.unsqueeze(0).unsqueeze(0)
    elif len(output.shape) == 3:
        output = output.unsqueeze(1) if output.shape[0] == 1 else output.unsqueeze(0)
    
    target = target.float()
    output = output.float()
    
    predict = (output > 0).float()
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

def compute_miou(pred, gt_mask):
    """Compute mIoU.
    Args:
        pred: predicted heatmap (numpy array)
        gt_mask: GT binary mask (numpy array)
    Returns:
        miou_score: mIoU score
    """
    try:
        # Ensure numpy arrays
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
            
        # Binarize (configured threshold)
        pred_binary = (pred > MetricConfig.IOU_THRESHOLD).astype(np.float32)
        gt_binary = (gt_mask > MetricConfig.IOU_THRESHOLD).astype(np.float32)
        
        # Intersection and union
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
        
        # IoU
        if union == 0:
            # If both prediction and GT have no target, IoU=1
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        return float(iou)
    except Exception as e:
        print(f"mIoU计算错误: {e}")
        return 0.0

def compute_pd_fa(pred, gt_mask, size):
    """PD/FA computation aligned with the reference metrics.py implementation.
    Args:
        pred: predicted heatmap (numpy array)
        gt_mask: GT binary mask (numpy array)
        size: image size (H, W)
    Returns:
        pd_score: PD score
        fa_score: FA score
    """
    try:
        # Ensure numpy arrays and binarize
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
            
        # Binarize (configured threshold) - fix: binarize GT as well
        DetectTh = MetricConfig.DETECTION_THRESHOLD
        output_one = pred.copy()
        target_one = gt_mask.copy()
        
        # Prediction binarization
        output_one[np.where(output_one < DetectTh)] = 0
        output_one[np.where(output_one >= DetectTh)] = 1
        
        # GT binarization - key fix
        target_one[np.where(target_one < DetectTh)] = 0
        target_one[np.where(target_one >= DetectTh)] = 1
        
        # Ensure correct dtype
        target_one = target_one.astype(np.int32)
        output_one = output_one.astype(np.int32)
        
        # Connected component analysis (consistent with the reference implementation)
        labelimage = measure.label(target_one, connectivity=2)
        props = measure.regionprops(labelimage, intensity_image=target_one, cache=True)
        
        TgtNum = len(props)
        TrueNum = 0
        FalseNum = 0
        
        # Handle edge cases
        if TgtNum == 0:
            if np.sum(output_one) == 0:
                return 1.0, 0.0  # no GT target and no prediction
            else:
                fa_score = float(np.sum(output_one) / (size[0] * size[1]))
                return 0.0, fa_score  # no GT target but has prediction
        
        # Separate PD and FA computations - fix algorithmic issue
        LocLen1 = MetricConfig.LOCLEN1   # PD search radius
        LocLen2 = MetricConfig.LOCLEN2   # FA exclusion radius
        
        # Step 1: PD computation (reference implementation)
        for i_tgt in range(len(props)):
            True_flag = 0
            pixel_coords = props[i_tgt].coords
            
            # For each GT pixel, check whether any prediction exists within LocLen1
            for i_pixel in pixel_coords:
                r_min = max(0, i_pixel[0] - LocLen1)
                r_max = min(output_one.shape[0], i_pixel[0] + LocLen1 + 1)
                c_min = max(0, i_pixel[1] - LocLen1)
                c_max = min(output_one.shape[1], i_pixel[1] + LocLen1 + 1)
                
                Tgt_area = output_one[r_min:r_max, c_min:c_max]
                if Tgt_area.sum() >= 1:
                    True_flag = 1
                    break  # early exit to avoid double counting
            
            if True_flag == 1:
                TrueNum += 1
        
        # Step 2: compute FA exclusion region
        Box2_map = np.ones(output_one.shape)
        for i_tgt in range(len(props)):
            pixel_coords = props[i_tgt].coords
            # Set exclusion region for FA computation
            for i_pixel in pixel_coords:
                r_min = max(0, i_pixel[0] - LocLen2)
                r_max = min(output_one.shape[0], i_pixel[0] + LocLen2 + 1)
                c_min = max(0, i_pixel[1] - LocLen2) 
                c_max = min(output_one.shape[1], i_pixel[1] + LocLen2 + 1)
                Box2_map[r_min:r_max, c_min:c_max] = 0
        
        # FA computation (reference implementation)
        False_output_one = output_one * Box2_map
        FalseNum = np.count_nonzero(False_output_one)
        
        # Final metrics
        pd_score = TrueNum / TgtNum if TgtNum > 0 else 0.0
        fa_score = FalseNum / (size[0] * size[1])
        
        return float(pd_score), float(fa_score)
        
    except Exception as e:
        print(f"PD/FA计算错误: {e}")
        return 0.0, 0.0

# ======================== Composite evaluation wrapper ========================

def compute_sequence_auc_with_shooting_rules(seq_pred, seq_centroid):
    """Compute sequence-level AUC using the PD-FA curve (aligned with metrics.py).
    
    Args:
        seq_pred: sequence prediction (T, H, W)
        seq_centroid: sequence centroid mask (T, H, W)
        
    Returns:
        auc_score: PD-FA curve AUC value
    """
    try:
        # Define threshold sequence (consistent with metrics.py)
        Th_Seg = np.array([0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 
                          0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
                          0.8, 0.85, 0.9, 1])
        
        # Initialize accumulated statistics (consistent with metrics.py)
        FalseNumAll = np.zeros(len(Th_Seg))
        TrueNumAll = np.zeros(len(Th_Seg))
        TgtNumAll = np.zeros(len(Th_Seg))
        pixelsNumber = seq_pred.shape[0] * seq_pred.shape[1] * seq_pred.shape[2]
        
        # Accumulate per-frame statistics (using our internal implementation)
        for t in range(seq_pred.shape[0]):
            pred_frame = seq_pred[t]      # (H, W)
            gt_frame = seq_centroid[t]    # (H, W)
            
            for th_i in range(len(Th_Seg)):
                FalseNum, TrueNum, TgtNum = compute_shooting_rules_single(pred_frame, gt_frame, Th_Seg[th_i])
                FalseNumAll[th_i] += FalseNum
                TrueNumAll[th_i] += TrueNum  
                TgtNumAll[th_i] += TgtNum
        
        # Compute PD/FA (consistent with metrics.py)
        Pd_seq = TrueNumAll / (TgtNumAll + 1e-8)  # avoid division by zero
        Fa_seq = FalseNumAll / pixelsNumber
        
        # Validity check
        if len(np.unique(Pd_seq)) < 2:
            # PD is constant; cannot compute a meaningful AUC
            if Pd_seq[0] == 0:
                return 0.0  # complete miss
            elif Pd_seq[0] == 1:
                return 1.0  # perfect detection
            else:
                return 0.5  # other cases
        
        # Compute AUC (consistent with metrics.py)
        try:
            auc_score = auc(Fa_seq, Pd_seq)
            
            # Boundary protection
            if auc_score < 0:
                auc_score = 0.0
            elif auc_score > 1:
                auc_score = 1.0
                
            return float(auc_score)
        except Exception as e:
            print(f"序列AUC计算异常: {e}")
            return 0.5
        
    except Exception as e:
        print(f"序列PD-FA AUC计算错误: {e}")
        return 0.5

def evaluate_comprehensive_metrics(seq_predictions, seq_targets, seq_centroids,
                                      dataset_info=None, *, roc_save_dir=None):
    """Unified interface for comprehensive metric evaluation (hybrid scheme).
    
    Args:
        seq_predictions: list of sequence predictions [seq1(T,H,W), seq2(T,H,W), ...]
        seq_targets: list of sequence target masks [seq1(T,H,W), seq2(T,H,W), ...]
        seq_centroids: list of sequence centroid masks [seq1(T,H,W), seq2(T,H,W), ...]
        dataset_info: dataset info dict {'seq_names': [...], 'dataset_name': 'xxx'}
        
    Returns:
        results: dict of metric results
    """
    results = {}
    seq_names = dataset_info.get('seq_names', [f'Seq{i}' for i in range(len(seq_predictions))])
    dataset_name = dataset_info.get('dataset_name', 'Unknown')
    
    # Compute metrics per sequence and optionally save ROC curves
    for seq_idx, seq_name in enumerate(seq_names):
        if seq_idx >= len(seq_predictions):
            continue
            
        seq_pred = seq_predictions[seq_idx]   # (T, H, W)
        seq_target = seq_targets[seq_idx]     # (T, H, W)  
        seq_centroid = seq_centroids[seq_idx] # (T, H, W)
        
        # Accumulated frame-level metrics
        frame_metrics = {
            'precision': [], 'recall': [], 'f1': [],
            'pd': [], 'fa': [], 'miou': []
        }
        
        # Compute per-frame metrics
        for t in range(seq_pred.shape[0]):
            pred_frame = seq_pred[t]      # (H, W)
            target_frame = seq_target[t]  # (H, W)
            centroid_frame = seq_centroid[t]  # (H, W)
            
            # Compute metrics using helpers in metric.py
            # Get keypoints
            pred_keypoints = get_keypoints(pred_frame)
            gt_keypoints = get_keypoints(centroid_frame)
            
            # Keypoint-based metrics (Precision/Recall/F1)
            precision = compute_metric(None, pred_frame, pred_keypoints, gt_keypoints, 
                                     'Precision', MetricConfig.DISTANCE_THRESHOLD)
            recall = compute_metric(None, pred_frame, pred_keypoints, gt_keypoints, 
                                  'Recall', MetricConfig.DISTANCE_THRESHOLD)
            f1 = compute_metric(None, pred_frame, pred_keypoints, gt_keypoints, 
                               'F1', MetricConfig.DISTANCE_THRESHOLD)
            
            # Other metrics via metric.py helpers
            miou_score = compute_miou(pred_frame, target_frame)
            
            # PD/FA via metric.py method
            size = pred_frame.shape
            pd_score, fa_score = compute_pd_fa(pred_frame, centroid_frame, size)
            
            # Collect metrics
            frame_metrics['precision'].append(precision)
            frame_metrics['recall'].append(recall)
            frame_metrics['f1'].append(f1)
            frame_metrics['pd'].append(pd_score)
            frame_metrics['fa'].append(fa_score)
            frame_metrics['miou'].append(miou_score)
        
        # Sequence-level AUC using the reference method (ShootingRules + PD/FA-ROC)
        auc_score = compute_sequence_auc_with_shooting_rules(seq_pred, seq_centroid)

        # Optionally save ROC / PD-FA curves
        if roc_save_dir is not None:
            try:
                save_sequence_pd_fa_curve(seq_pred, seq_centroid,
                                          save_dir=roc_save_dir,
                                          seq_name=seq_name)
            except Exception as _e:
                # Print one warning and continue
                print(f"[warn] 保存 {seq_name} ROC 失败: {_e}")
        
        # Sequence averages
        results[seq_name] = {
            'precision': np.mean(frame_metrics['precision']) if frame_metrics['precision'] else 0.0,
            'recall': np.mean(frame_metrics['recall']) if frame_metrics['recall'] else 0.0,
            'f1': np.mean(frame_metrics['f1']) if frame_metrics['f1'] else 0.0,
            'auc': auc_score,
            'pd': np.mean(frame_metrics['pd']) if frame_metrics['pd'] else 0.0,
            'fa': np.mean(frame_metrics['fa']) if frame_metrics['fa'] else 0.0,
            'miou': np.mean(frame_metrics['miou']) if frame_metrics['miou'] else 0.0,
            'num': seq_pred.shape[0]
        }
    
    # Overall statistics (for NUDT-MIRSDT, also compute low/high SNR stats)
    if 'NUDT-MIRSDT' in dataset_name:
        # Low/high SNR sequence indices for NUDT-MIRSDT
        if 'Noise' in dataset_name:
            # Sequence split for the noise variant
            all_aucs = [results[seq_name]['auc'] for seq_name in results.keys() 
                       if seq_name.startswith('Sequence')]
            results['low_snr_auc'] = np.mean(all_aucs) if all_aucs else 0.0
            results['high_snr_auc'] = np.mean(all_aucs) if all_aucs else 0.0
            results['overall_auc'] = np.mean(all_aucs) if all_aucs else 0.0
        else:
            # Sequence split for the original NUDT-MIRSDT
            low_snr_indices = [7, 13, 14, 15, 16, 17, 18, 19]
            high_snr_indices = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
            
            low_snr_aucs = []
            high_snr_aucs = []
            all_aucs = []
            
            for seq_idx, seq_name in enumerate(seq_names):
                if seq_name in results:
                    auc_val = results[seq_name]['auc']
                    all_aucs.append(auc_val)
                    if seq_idx in low_snr_indices:
                        low_snr_aucs.append(auc_val)
                    elif seq_idx in high_snr_indices:
                        high_snr_aucs.append(auc_val)
            
            results['low_snr_auc'] = np.mean(low_snr_aucs) if low_snr_aucs else 0.0
            results['high_snr_auc'] = np.mean(high_snr_aucs) if high_snr_aucs else 0.0
            results['overall_auc'] = np.mean(all_aucs) if all_aucs else 0.0
        
        # Average mIoU
        all_mious = [results[seq_name]['miou'] for seq_name in results.keys() 
                    if seq_name.startswith('Sequence')]
        results['avg_miou'] = np.mean(all_mious) if all_mious else 0.0
    
    return results

def print_evaluation_table(results, dataset_name):
    """Print a formatted metrics table.
    
    Args:
        results: output of evaluate_comprehensive_metrics
        dataset_name: dataset name
    """
    print("\n" + "="*120)
    print(f"Results for {dataset_name}")
    print("="*120)
    
    # Header
    print(f"+{'-'*12}+{'-'*6}+{'-'*12}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+")
    print(f"|{'Seq':<12}|{'Num':<6}|{'Precision':<12}|{'Recall':<8}|{'F1':<8}|{'AUC':<8}|{'PD':<8}|{'FA':<8}|{'mIoU':<8}|")
    print(f"+{'-'*12}+{'-'*6}+{'-'*12}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+")
    
    # Per-sequence results
    overall_metrics = {
        'precision': [], 'recall': [], 'f1': [], 'auc': [], 
        'pd': [], 'fa': [], 'miou': [], 'num': []
    }
    
    # Show all sequences (exclude aggregate stats)
    for seq_name, metrics in results.items():
        # Filter out aggregate stat keys (e.g., low_snr_auc, high_snr_auc, overall_auc, avg_miou)
        if seq_name in ['low_snr_auc', 'high_snr_auc', 'overall_auc', 'avg_miou']:
            continue
            
        # Ensure metrics is a dict and contains required keys
        if not isinstance(metrics, dict) or 'precision' not in metrics:
            continue
            
        # Supported sequence name patterns:
        # - Sequence*, Seq* (e.g., NUDT-MIRSDT)
        # - TSIRMT, DSAT, SIATD (single-sequence datasets)
        # - 00001, 00002, ... (numeric sequences, e.g., frame-wise TSIRMT processing)
        if not (seq_name[0].isalpha() or seq_name[0].isdigit()):
            continue
            
        print(f"|{seq_name:<12}|{metrics['num']:>6}|"
              f"{metrics['precision']:>12.4f}|{metrics['recall']:>8.4f}|"
              f"{metrics['f1']:>8.4f}|{metrics['auc']:>8.4f}|"
              f"{metrics['pd']:>8.4f}|{metrics['fa']:>8.6f}|"
              f"{metrics['miou']:>8.4f}|")
        
        # Collect overall stats
        for key in ['precision', 'recall', 'f1', 'auc', 'pd', 'fa', 'miou', 'num']:
            overall_metrics[key].append(metrics[key])
    
    print(f"+{'-'*12}+{'-'*6}+{'-'*12}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+")
    
    # Overall results
    if overall_metrics['precision']:
        overall_prec = np.mean(overall_metrics['precision'])
        overall_recall = np.mean(overall_metrics['recall'])
        overall_f1 = np.mean(overall_metrics['f1'])
        overall_auc = np.mean(overall_metrics['auc'])
        overall_pd = np.mean(overall_metrics['pd'])
        overall_fa = np.mean(overall_metrics['fa'])
        overall_miou = np.mean(overall_metrics['miou'])
        overall_num = sum(overall_metrics['num'])
        
        print(f"|{'Overall':<12}|{overall_num:>6}|"
              f"{overall_prec:>12.4f}|{overall_recall:>8.4f}|"
              f"{overall_f1:>8.4f}|{overall_auc:>8.4f}|"
              f"{overall_pd:>8.4f}|{overall_fa:>8.6f}|"
              f"{overall_miou:>8.4f}|")
    
    print(f"+{'-'*12}+{'-'*6}+{'-'*12}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+")
    
    # If grouped results exist, show low/high SNR stats
    if 'low_snr_auc' in results:
        print(f"\nLow SNR Sequences AUC: {results['low_snr_auc']:.5f}")
        print(f"High SNR Sequences AUC: {results['high_snr_auc']:.5f}")
        print(f"Overall AUC: {results['overall_auc']:.5f}")
        print(f"Average Class IoU: {results.get('avg_miou', 0.0):.5f}")
    
    print("="*120)

# === Sequence ROC curve saving utility ===

def save_sequence_pd_fa_curve(seq_pred, seq_gt, save_dir, seq_name,
                              *, figsize=(6,5), dpi=300):
    """Save a PD-FA (ROC) curve PNG for an entire sequence (aligned with metrics.py).
    
    Correct approach: accumulate counts (TrueNum, FalseNum, TgtNum) first, then compute PD/FA.
    Reference: metrics.py lines 83-85.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Define threshold sequence (aligned with AUC computation)
    Th_Seg = np.array([0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 
                      0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
                      0.8, 0.85, 0.9, 1])
    
    # Initialize accumulated-stat arrays (consistent with metrics.py)
    FalseNumAll = np.zeros(len(Th_Seg))
    TrueNumAll = np.zeros(len(Th_Seg))
    TgtNumAll = np.zeros(len(Th_Seg))
    pixelsNumber = seq_pred.shape[0] * seq_pred.shape[1] * seq_pred.shape[2]
    
    # Accumulate per-frame stats (consistent with compute_sequence_auc_with_shooting_rules)
    for t in range(seq_pred.shape[0]):
        pred_frame = seq_pred[t]
        gt_frame = seq_gt[t]
        
        for th_i in range(len(Th_Seg)):
            FalseNum, TrueNum, TgtNum = compute_shooting_rules_single(
                pred_frame, gt_frame, Th_Seg[th_i], debug_seq_name=seq_name)
            FalseNumAll[th_i] += FalseNum
            TrueNumAll[th_i] += TrueNum
            TgtNumAll[th_i] += TgtNum
    
    # Compute PD/FA (consistent with metrics.py lines 83-85)
    # Sum counts first, then divide (do not average per-frame PD/FA)
    PD_array = TrueNumAll / (TgtNumAll + 1e-8)  # avoid division by zero
    FA_array = FalseNumAll / pixelsNumber
    
    # Plot curves
    import matplotlib
    matplotlib.use('Agg')  # non-GUI backend
    import matplotlib.pyplot as plt
    idx = np.argsort(FA_array)
    FA_sorted = FA_array[idx]
    PD_sorted = PD_array[idx]
    
    plt.figure(figsize=figsize)
    plt.plot(FA_sorted, PD_sorted, marker='o', linewidth=1.2, markersize=4)
    plt.xlabel('FA (False Alarm Rate)', fontsize=12)
    plt.ylabel('PD (Probability of Detection)', fontsize=12)
    plt.title(f'PD-FA Curve: {seq_name}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{seq_name}_pd_fa.png")
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    
    # Debug prints
    auc_val = np.trapz(PD_sorted, FA_sorted) if len(np.unique(PD_sorted)) > 1 else 0.0
    print(f"  [{seq_name}] TgtNum={TgtNumAll.sum():.0f}, "
          f"PD_range=[{PD_array.min():.3f}, {PD_array.max():.3f}], "
          f"FA_range=[{FA_array.min():.6f}, {FA_array.max():.6f}], "
          f"AUC={auc_val:.4f}")
    
    return save_path


# ======================== Unified two-level evaluator ========================

class TwoLevelMetrics:
    """Two-level evaluator: accumulated level and sequence level.
    
    Usage:
        metric = TwoLevelMetrics()
        
        # For each sequence
        for seq in sequences:
            metric.start_sequence()
            
            # For each frame in the sequence
            for frame in seq:
                metric.update(pred, gt)
            
            metric.end_sequence()
        
        # Get results
        results = metric.get_results()
    """
    
    def __init__(self):
        """Initialize the two-level evaluator."""
        # Accumulated level - global accumulation
        self.accumulated_miou = mIoU()
        self.accumulated_pdfa = PD_FA()
        
        # Sequence level - average per sequence, then average across sequences
        self.seq_metrics_list = []  # per-sequence average metrics
        self.current_seq_metrics = None  # current sequence per-frame metrics
        
        # Counters
        self.total_frames = 0
        self.total_sequences = 0
        
    def start_sequence(self):
        """Start evaluation for a new sequence."""
        self.current_seq_metrics = {
            'ious': [],
            'pds': [],
            'fas': [],
            'precisions': [],
            'recalls': [],
            'f1s': []
        }
        
    def update(self, pred_tensor, gt_tensor, size=(256, 256)):
        """Update metrics (process one frame).
        
        Args:
            pred_tensor: prediction tensor [1, 1, H, W], in logits format
            gt_tensor: GT tensor [1, 1, H, W], values in [0, 1]
            size: image size used by PD_FA
        """
        # === 1. Accumulated-level update ===
        self.accumulated_miou.update(pred_tensor, gt_tensor)
        self.accumulated_pdfa.update(pred_tensor, gt_tensor, size)
        
        # === 2. Sequence-level computation (compute per-frame, then average) ===
        # Convert to numpy for computation
        pred_np = torch.sigmoid(pred_tensor).cpu().numpy().squeeze()  # apply sigmoid then convert to probability
        gt_np = gt_tensor.cpu().numpy().squeeze()
        
        # IoU computation
        pred_binary = (pred_np > 0.5).astype(np.float32)
        gt_binary = (gt_np > 0.5).astype(np.float32)
        
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        frame_iou = intersection / (union + 1e-8)
        
        # PD/FA computation (via compute_pd_fa)
        frame_pd, frame_fa = compute_pd_fa(pred_np, gt_np, size)
        
        # Precision/Recall/F1 (keypoint-based)
        pred_keypoints = get_keypoints(pred_np)
        gt_keypoints = get_keypoints(gt_np)
        precision, recall, _ = compute_prfa(pred_keypoints, gt_keypoints, MetricConfig.DISTANCE_THRESHOLD)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        
        # Store to current sequence (if within a sequence)
        if self.current_seq_metrics is not None:
            self.current_seq_metrics['ious'].append(frame_iou)
            self.current_seq_metrics['pds'].append(frame_pd)
            self.current_seq_metrics['fas'].append(frame_fa)
            self.current_seq_metrics['precisions'].append(precision)
            self.current_seq_metrics['recalls'].append(recall)
            self.current_seq_metrics['f1s'].append(f1)
        
        self.total_frames += 1
        
    def end_sequence(self):
        """End evaluation for the current sequence."""
        if self.current_seq_metrics is not None:
            # Compute current sequence averages
            seq_avg_metrics = {
                'miou': np.mean(self.current_seq_metrics['ious']) if self.current_seq_metrics['ious'] else 0.0,
                'pd': np.mean(self.current_seq_metrics['pds']) if self.current_seq_metrics['pds'] else 0.0,
                'fa': np.mean(self.current_seq_metrics['fas']) if self.current_seq_metrics['fas'] else 0.0,
                'precision': np.mean(self.current_seq_metrics['precisions']) if self.current_seq_metrics['precisions'] else 0.0,
                'recall': np.mean(self.current_seq_metrics['recalls']) if self.current_seq_metrics['recalls'] else 0.0,
                'f1': np.mean(self.current_seq_metrics['f1s']) if self.current_seq_metrics['f1s'] else 0.0,
                'num_frames': len(self.current_seq_metrics['ious'])
            }
            self.seq_metrics_list.append(seq_avg_metrics)
            self.current_seq_metrics = None
            self.total_sequences += 1
    
    def get_results(self):
        """Get evaluation results at both levels.
        
        Returns:
            dict: metric results for both levels
        """
        # 1. Accumulated level
        acc_pixacc, acc_miou = self.accumulated_miou.get()
        acc_pd, acc_fa = self.accumulated_pdfa.get()
        
        # 2. Sequence level
        if self.seq_metrics_list:
            seq_mious = [s['miou'] for s in self.seq_metrics_list]
            seq_pds = [s['pd'] for s in self.seq_metrics_list]
            seq_fas = [s['fa'] for s in self.seq_metrics_list]
            seq_precisions = [s['precision'] for s in self.seq_metrics_list]
            seq_recalls = [s['recall'] for s in self.seq_metrics_list]
            seq_f1s = [s['f1'] for s in self.seq_metrics_list]
            
            seq_miou = np.mean(seq_mious)
            seq_pd = np.mean(seq_pds)
            seq_fa = np.mean(seq_fas)
            seq_precision = np.mean(seq_precisions)
            seq_recall = np.mean(seq_recalls)
            seq_f1 = np.mean(seq_f1s)
        else:
            seq_miou = 0.0
            seq_pd = 0.0
            seq_fa = 0.0
            seq_precision = 0.0
            seq_recall = 0.0
            seq_f1 = 0.0
        
        return {
            'accumulated': {
                'miou': float(acc_miou),
                'pd': float(acc_pd),
                'fa': float(acc_fa),
                'pixacc': float(acc_pixacc)
            },
            'sequence_level': {
                'miou': float(seq_miou),
                'pd': float(seq_pd),
                'fa': float(seq_fa),
                'precision': float(seq_precision),
                'recall': float(seq_recall),
                'f1': float(seq_f1)
            },
            'statistics': {
                'total_frames': self.total_frames,
                'total_sequences': self.total_sequences
            }
        }
    
    def reset(self):
        """Reset all statistics."""
        self.accumulated_miou.reset()
        self.accumulated_pdfa.reset()
        self.seq_metrics_list = []
        self.current_seq_metrics = None
        self.total_frames = 0
        self.total_sequences = 0


def print_two_level_results(results, dataset_name='Unknown', pred_dir=None):
    """Print a detailed table for two-level evaluation results.
    
    Args:
        results: output of TwoLevelMetrics.get_results()
        dataset_name: dataset name
        pred_dir: prediction directory (for display)
    """
    acc = results['accumulated']
    seq = results['sequence_level']
    stats = results['statistics']
    
    print("\n" + "=" * 120)
    print(f"两级别评估结果 - {dataset_name}")
    if pred_dir:
        print(f"预测目录: {pred_dir}")
    print("=" * 120)
    
    print(f"\n统计信息:")
    print(f"  总帧数: {stats['total_frames']}")
    print(f"  序列数: {stats['total_sequences']}")
    
    # Table header
    print("\n" + "-" * 120)
    print(f"{'评估级别':<20} | {'mIoU':<10} | {'PD':<10} | {'FA(×10⁻⁶)':<12} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'说明':<20}")
    print("-" * 120)
    
    # Accumulated level
    print(f"{'累积级别':<20} | {acc['miou']:<10.4f} | {acc['pd']:<10.4f} | "
          f"{acc['fa']*1e6:<12.2f} | {'-':<10} | {'-':<10} | {'-':<10} | {'累积I∩U计算':<20}")
    
    # Sequence level
    print(f"{'序列级别':<20} | {seq['miou']:<10.4f} | {seq['pd']:<10.4f} | "
          f"{seq['fa']*1e6:<12.2f} | {seq['precision']:<10.4f} | {seq['recall']:<10.4f} | {seq['f1']:<10.4f} | {'序列平均后再平均':<20}")
    
    print("-" * 120)
    
    # Detailed explanation
    print("\n" + "=" * 120)
    print("评估级别详解")
    print("=" * 120)
    
    print("\n1. 累积级别 (Accumulated-level)")
    print("   计算方式: 累积所有帧的 intersection 和 union，最后一起计算")
    print("   公式: mIoU = Σ(intersection) / Σ(union)")
    print("   特点: 与训练时的累积计算完全一致，是论文中通常报告的指标")
    print(f"   结果: mIoU={acc['miou']:.4f}, PD={acc['pd']:.4f}, FA={acc['fa']*1e6:.2f}×10⁻⁶")
    print(f"   说明: 累积级别不计算Precision/Recall/F1（需要逐帧关键点匹配）")
    
    print("\n2. 序列级别 (Sequence-level)")
    print("   计算方式: 先对每个序列求平均，再对所有序列求平均")
    print("   公式: mIoU = mean([seq₁_mean_iou, seq₂_mean_iou, ...])")
    print("   特点: 每个序列权重相同，避免长序列主导结果")
    print(f"   结果: mIoU={seq['miou']:.4f}, PD={seq['pd']:.4f}, FA={seq['fa']*1e6:.2f}×10⁻⁶")
    print(f"          Precision={seq['precision']:.4f}, Recall={seq['recall']:.4f}, F1={seq['f1']:.4f}")
    
    print("\n" + "=" * 120)
    print("建议")
    print("=" * 120)
    print("  - 论文报告: 使用 累积级别的mIoU/PD/FA (与主流方法对比)")
    print("  - 性能分析: 使用 序列级别 (包含完整指标：mIoU/PD/FA/Precision/Recall/F1)")
    print("  - 公平对比: 使用 序列级别 (避免数据不平衡，每个序列权重相同)")
    print("=" * 120 + "\n")