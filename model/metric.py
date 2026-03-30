"""
评估指标模块 - 配置参数在 MetricConfig 类（第9-38行）
"""

import cv2
import imutils

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, auc
from skimage import measure

class MetricConfig:
    """评估指标配置 - 所有阈值参数集中管理"""
    
    # 二值化阈值
    DETECTION_THRESHOLD = 0.5    # 热图转二值mask（影响PD/FA/mIoU）
    IOU_THRESHOLD = 0.5          # mIoU计算阈值
    
    # 关键点提取
    MIN_CONTOUR_AREA = 1         # 最小连通域面积（像素）
    
    # PD/FA阈值（统一设置为3像素）
    LOCLEN1 = 3                  # PD搜索半径（像素）- GT区域±3范围
    LOCLEN2 = 3                  # FA排除半径（像素）- GT中心±3范围（统一为3）
    
    # 距离阈值（统一为3像素）
    DISTANCE_THRESHOLD = 3       # P/R/F1距离阈值（默认3像素）
    
    # 注：命令行可配置参数
    # --dthres 3         : P/R/F1距离阈值（统一默认值3像素）

def get_keypoints(featmap, min_area=None):
    """优化的关键点提取函数
    
    使用skimage的label+regionprops，更可靠地处理极小连通域
    
    Args:
        featmap: 特征图 (H×W)
        min_area: 最小连通域面积阈值
    Returns:
        keypoints: 关键点列表 [[x1,y1], [x2,y2], ...]
    """
    if min_area is None:
        min_area = MetricConfig.MIN_CONTOUR_AREA
        
    fmap = featmap.copy()
    
    # 确保输入是2D numpy数组
    while len(fmap.shape) > 2:
        if fmap.shape[0] == 1:
            fmap = fmap.squeeze(0)
        else:
            fmap = fmap[0]
    
    # 确保数据类型正确
    if fmap.dtype != np.float32:
        fmap = fmap.astype(np.float32)
    
    # 使用配置的阈值进行二值化
    threshold = MetricConfig.DETECTION_THRESHOLD
    binary_map = (fmap >= threshold).astype(np.uint8)
    
    # 最后检查：确保是2D数组
    if len(binary_map.shape) != 2:
        raise ValueError(f"无法将输入转换为2D数组，当前形状: {binary_map.shape}")
    
    # ⭐ 使用skimage的label进行连通域分析（更可靠）
    from skimage import measure
    labeled_map = measure.label(binary_map, connectivity=2)
    regions = measure.regionprops(labeled_map)
    
    res = []
    for region in regions:
        # 面积过滤
        if region.area < min_area:
            continue
        
        # 获取质心 (row, col) -> (y, x)
        centroid_y, centroid_x = region.centroid
        
        # 转换为整数坐标 (x, y)
        cX = int(round(centroid_x))
        cY = int(round(centroid_y))
        
        # 确保坐标在图像范围内
        h, w = binary_map.shape
        cX = max(0, min(w-1, cX))
        cY = max(0, min(h-1, cY))
        
        res.append([cX, cY])
    
    return res

def distance(x, y):
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2

def compute_prfa(pred, gt, th):
    """优化的Precision/Recall/FA计算
    Args:
        pred: 预测关键点列表 [[x1,y1], [x2,y2], ...]
        gt: 真实关键点列表 [[x1,y1], [x2,y2], ...]
        th: 距离阈值
    """
    P, G = len(pred), len(gt)

    if P == 0 and G == 0:        
        return [1.0, 1.0, 0]
    elif P == 0:        
        return [0.0, 0.0, 0]
    elif G == 0:        
        return [0.0, 0.0, P]  # 修复：返回实际虚警数量
    else:
        # 使用更精确的匹配算法
        matched_pred = set()
        matched_gt = set()
        
        # 为每个GT点找到最近的预测点
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
        
        # 计算指标
        tp_precision = len(matched_pred)  # 被正确匹配的预测数
        tp_recall = len(matched_gt)       # 被正确匹配的GT数
        
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
    """使用指定阈值计算PD和FA（不修改全局配置）
    Args:
        pred: 预测热图 (numpy array)
        gt_mask: GT二值掩码 (numpy array) 
        size: 图像尺寸 (H, W)
        detect_th: 检测阈值
    Returns:
        pd_score: PD值
        fa_score: FA值
    """
    try:
        # 确保数据是numpy数组
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
            
        # 使用指定阈值进行二值化
        output_one = pred.copy()
        target_one = gt_mask.copy()
        
        # 预测二值化
        output_one[np.where(output_one < detect_th)] = 0
        output_one[np.where(output_one >= detect_th)] = 1
        
        # GT二值化
        target_one[np.where(target_one < detect_th)] = 0
        target_one[np.where(target_one >= detect_th)] = 1
        
        # 确保数据类型正确
        target_one = target_one.astype(np.int32)
        output_one = output_one.astype(np.int32)
        
        # 连通域分析
        labelimage = measure.label(target_one, connectivity=2)
        props = measure.regionprops(labelimage, intensity_image=target_one, cache=True)
        
        TgtNum = len(props)
        TrueNum = 0
        FalseNum = 0
        
        # 边界情况处理
        if TgtNum == 0:
            if np.sum(output_one) == 0:
                return 1.0, 0.0  # 无目标无预测
            else:
                fa_score = float(np.sum(output_one) / (size[0] * size[1]))
                return 0.0, fa_score  # 无目标有预测
        
        # PD计算
        LocLen1 = MetricConfig.LOCLEN1
        LocLen2 = MetricConfig.LOCLEN2
        
        for i_tgt in range(len(props)):
            True_flag = 0
            pixel_coords = props[i_tgt].coords
            
            # 对目标的每个像素检查LocLen1范围内是否有预测
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
        
        # FA计算 - 排除区域
        Box2_map = np.ones(output_one.shape)
        for i_tgt in range(len(props)):
            pixel_coords = props[i_tgt].coords
            for i_pixel in pixel_coords:
                r_min = max(0, i_pixel[0] - LocLen2)
                r_max = min(output_one.shape[0], i_pixel[0] + LocLen2 + 1)
                c_min = max(0, i_pixel[1] - LocLen2) 
                c_max = min(output_one.shape[1], i_pixel[1] + LocLen2 + 1)
                Box2_map[r_min:r_max, c_min:c_max] = 0
        
        # FA计算
        False_output_one = output_one * Box2_map
        FalseNum = np.count_nonzero(False_output_one)
        
        # 计算最终指标
        pd_score = TrueNum / TgtNum if TgtNum > 0 else 0.0
        fa_score = FalseNum / (size[0] * size[1])
        
        return float(pd_score), float(fa_score)
        
    except Exception as e:
        return 0.0, 0.0

def compute_auc(pred_map, gt_mask):
    """基于metrics.py的PD-FA曲线AUC计算（完全参照原始实现）
    
    使用ShootingRules逻辑计算PD-FA曲线AUC：
    - PD (Probability of Detection) = TrueNum / TgtNum
    - FA (False Alarm Rate) = FalseNum / pixelsNumber
    - AUC = auc(FA_array, PD_array)
    
    Args:
        pred_map: 预测热图 (H×W), 值在[0,1]
        gt_mask: GT二值掩码 (H×W), 值为0或1
    Returns:
        auc_score: PD-FA曲线下面积
    """
    try:
        # 确保数据是numpy数组
        if isinstance(pred_map, torch.Tensor):
            pred_map = pred_map.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        # 定义阈值序列（完全参照metrics.py）
        Th_Seg = np.array([0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 
                          0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
                          0.8, 0.85, 0.9, 1])
        
        PD_array = []
        FA_array = []
        pixelsNumber = pred_map.shape[0] * pred_map.shape[1]
        
        # 对每个阈值计算PD和FA（使用ShootingRules逻辑）
        for DetectTh in Th_Seg:
            FalseNum, TrueNum, TgtNum = compute_shooting_rules_single(pred_map, gt_mask, DetectTh)
            
            # 计算PD和FA
            if TgtNum > 0:
                pd_score = TrueNum / TgtNum
            else:
                pd_score = 0.0
                
            fa_score = FalseNum / pixelsNumber
            
            PD_array.append(pd_score)
            FA_array.append(fa_score)
        
        # 转换为numpy数组
        PD_array = np.array(PD_array)
        FA_array = np.array(FA_array)
        
        # 确保FA是递增的（AUC计算要求）
        sorted_indices = np.argsort(FA_array)
        FA_sorted = FA_array[sorted_indices]
        PD_sorted = PD_array[sorted_indices]
        
        # 边界情况处理（在排序后进行，避免逻辑冲突）
        unique_pd = np.unique(PD_sorted)
        unique_fa = np.unique(FA_sorted)

        # 1) PD全为0: 完全漏检，AUC=0
        if len(unique_pd) == 1 and unique_pd[0] == 0:
            return 0.0

        # 2) PD全为1: 完美检测，AUC=1
        if len(unique_pd) == 1 and unique_pd[0] == 1:
            return 1.0

        # 3) FA全相同或PD全相同且不为0/1: 曲线退化
        if len(unique_fa) < 2 or len(unique_pd) < 2:
            # 曲线退化，无法计算有效AUC，返回0.5（随机猜测基线）
            # 这比返回0.0更合理，因为0.0意味着"完全失败"，而曲线退化只是"无法评估"
            return 0.5
        
        # 计算AUC（完全参照metrics.py）
        try:
            
            # 使用梯形积分计算AUC
            auc_score = auc(FA_sorted, PD_sorted)
            
            # 浮点舍入误差可能产生极小负数；截断为0
            if auc_score < 0:
                auc_score = 0.0
            # AUC不应大于1
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
    """单帧ShootingRules计算（完全参照metrics.py的ShootingRules.forward）
    
    Args:
        pred_map: 预测热图 (H×W)
        gt_mask: GT掩码 (H×W) 
        DetectTh: 检测阈值
        debug_seq_name: 调试序列名（可选）
    Returns:
        FalseNum: 虚警数量
        TrueNum: 检测数量  
        TgtNum: 目标数量
    """
    FalseNum = 0
    TrueNum = 0
    TgtNum = 0
    
    # 复制输入数据
    output_one = pred_map.copy()
    target_one = gt_mask.copy()
    
    # 调试：检查GT mask的基本信息
    if debug_seq_name and DetectTh == 0.5:  # 只在阈值0.5时打印，避免过多输出
        gt_unique = np.unique(target_one)
        gt_nonzero = np.count_nonzero(target_one)
        print(f"    [DEBUG {debug_seq_name}] GT shape={target_one.shape}, "
              f"unique_vals={gt_unique}, nonzero_count={gt_nonzero}, "
              f"min={target_one.min():.3f}, max={target_one.max():.3f}")
    
    # 预测二值化（完全参照metrics.py第35-36行）
    output_one[np.where(output_one < DetectTh)] = 0
    output_one[np.where(output_one >= DetectTh)] = 1
    
    # 🔥 关键修复：GT mask二值化（这可能是问题所在）
    # 确保target_one是二值的（0或1）
    if target_one.max() > 1.0:
        # 如果GT是0-255格式，先归一化
        target_one = target_one / 255.0
    
    # 二值化GT（任何>0.5的值都认为是目标）
    target_one = (target_one > 0.5).astype(np.float32)
    
    # 连通域分析（完全参照metrics.py第38-39行）
    labelimage = measure.label(target_one.astype(np.uint8), connectivity=2)
    props = measure.regionprops(labelimage, intensity_image=target_one, cache=True)
    
    TgtNum = len(props)
    
    if TgtNum == 0:
        # 无目标时，所有预测都是虚警
        FalseNum = np.count_nonzero(output_one)
        return FalseNum, TrueNum, TgtNum
    
    # 参数设置（使用统一配置，确保与其他PD/FA计算一致）
    LocLen1 = MetricConfig.LOCLEN1  # 检测半径（从配置读取）
    LocLen2 = MetricConfig.LOCLEN2  # 排除半径（从配置读取）
    
    # 初始化排除区域地图
    Box2_map = np.ones(output_one.shape)
    
    # 对每个目标区域处理（完全参照metrics.py第48-58行）
    for i_tgt in range(len(props)):
        True_flag = 0
        
        pixel_coords = props[i_tgt].coords
        for i_pixel in pixel_coords:
            # 设置排除区域（FA计算用）
            r_min = max(0, i_pixel[0] - LocLen2)
            r_max = min(output_one.shape[0], i_pixel[0] + LocLen2 + 1)
            c_min = max(0, i_pixel[1] - LocLen2) 
            c_max = min(output_one.shape[1], i_pixel[1] + LocLen2 + 1)
            Box2_map[r_min:r_max, c_min:c_max] = 0
            
            # 检测区域（PD计算用）
            r_min_detect = max(0, i_pixel[0] - LocLen1)
            r_max_detect = min(output_one.shape[0], i_pixel[0] + LocLen1 + 1)
            c_min_detect = max(0, i_pixel[1] - LocLen1)
            c_max_detect = min(output_one.shape[1], i_pixel[1] + LocLen1 + 1)
            
            Tgt_area = output_one[r_min_detect:r_max_detect, c_min_detect:c_max_detect]
            if Tgt_area.sum() >= 1:
                True_flag = 1
        
        if True_flag == 1:
            TrueNum += 1
    
    # 计算虚警（完全参照metrics.py第60-61行）
    False_output_one = output_one * Box2_map
    FalseNum = np.count_nonzero(False_output_one)
    
    return FalseNum, TrueNum, TgtNum

def compute_roc_auc_manual(y_scores, y_true):
    """手动实现ROC-AUC计算
    
    Args:
        y_scores: 预测分数数组
        y_true: 真实标签数组 (0/1)
    Returns:
        auc: AUC值
    """
    # 获取所有unique的阈值（包括边界值）
    thresholds = np.unique(y_scores)
    # 添加边界阈值确保TPR和FPR覆盖[0,1]
    thresholds = np.concatenate([thresholds, [thresholds.min() - 1e-8, thresholds.max() + 1e-8]])
    thresholds = np.sort(thresholds)[::-1]  # 降序排列
    
    # 计算每个阈值下的TPR和FPR
    tpr_list = []
    fpr_list = []
    
    # 总的正例和负例数
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    for threshold in thresholds:
        # 预测为正例的样本
        y_pred = (y_scores >= threshold).astype(int)
        
        # 计算混淆矩阵
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # 计算TPR和FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # 转换为numpy数组
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)
    
    # 确保FPR是递增的（AUC计算要求）
    sorted_indices = np.argsort(fpr_array)
    fpr_sorted = fpr_array[sorted_indices]
    tpr_sorted = tpr_array[sorted_indices]
    
    # 使用梯形法则计算AUC
    auc_value = auc(fpr_sorted, tpr_sorted)
    
    return float(auc_value)

def compute_pd_fa_curve(pred_map, gt_mask):
    """返回整个阈值序列上的 (FA, PD) 数组，便于可视化。

    与 compute_auc 使用完全相同的阈值、半径与逻辑，
    但仅返回曲线数据，不计算 AUC。
    """
    # 确保 numpy
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
    """保存 PD-FA 曲线到指定路径。

    Args:
        pred_map (ndarray | Tensor): 预测热图
        gt_mask  (ndarray | Tensor): GT mask
        save_path (str): 路径包含文件名，如 '/tmp/roc.png'
    """
    import matplotlib
    matplotlib.use('Agg')  # 非GUI后端
    import matplotlib.pyplot as plt
    FA, PD = compute_pd_fa_curve(pred_map, gt_mask)

    # 按 FA 升序排序（确保从左到右）
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
            return 0.5  # 缺少GT mask时返回默认值
        return compute_auc(pred, gt_mask)
    elif metric == 'mIoU':
        if gt_mask is None:
            return 0.0  # 缺少GT mask时返回默认值
        return compute_miou(pred, gt_mask)
    elif metric == 'PD':
        if gt_mask is None:
            return 0.0  # 缺少GT mask时返回默认值
        size = (pred.shape[0], pred.shape[1]) if len(pred.shape) >= 2 else (256, 256)
        pd_score, _ = compute_pd_fa(pred, gt_mask, size)
        return pd_score
    elif metric == 'FA':
        if gt_mask is None:
            return 0.0  # 缺少GT mask时返回默认值
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
        # 先转换为numpy，然后处理
        if isinstance(preds, torch.Tensor):
            # 如果是logits，先sigmoid再二值化
            preds_np = torch.sigmoid(preds).cpu().numpy()
        else:
            preds_np = preds
            
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels
        
        # 处理4D tensor [B, C, H, W] -> [H, W]
        while len(preds_np.shape) > 2:
            preds_np = preds_np.squeeze(0)
        while len(labels_np.shape) > 2:
            labels_np = labels_np.squeeze(0)
        
        # 二值化（使用阈值0.5）
        predits = (preds_np > 0.5).astype('int64')
        labelss = (labels_np > 0.5).astype('int64')
        
        # 确保是2D数组
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
        matched_image_indices = set()  # 记录已匹配的预测区域
        
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                if m in matched_image_indices:  # 跳过已匹配的区域
                    continue
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1
                    matched_image_indices.add(m)  # 标记为已匹配
                    break

        self.dismatch_pixel += (predits - true_img).sum()
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD / self.target if self.target > 0 else 0.0
        
        # 确保FA是float类型
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
    # 确保输出是tensor
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output).float()
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target).float()
        
    # 统一处理维度：确保是4D [B, C, H, W]
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
    maxi = 2  # 修正：范围应该是1到2，这样只统计值为1的像素
    nbins = 1
    
    # 确保输出是tensor
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output).float()
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target).float()
        
    # 统一处理维度：确保是4D [B, C, H, W]
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
    """计算mIoU
    Args:
        pred: 预测热图 (numpy array)
        gt_mask: GT二值掩码 (numpy array)
    Returns:
        miou_score: mIoU值
    """
    try:
        # 确保数据是numpy数组
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
            
        # 二值化 (使用配置参数)
        pred_binary = (pred > MetricConfig.IOU_THRESHOLD).astype(np.float32)
        gt_binary = (gt_mask > MetricConfig.IOU_THRESHOLD).astype(np.float32)
        
        # 计算交集和并集
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
        
        # 计算IoU
        if union == 0:
            # 如果预测和GT都没有目标，IoU为1
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        return float(iou)
    except Exception as e:
        print(f"mIoU计算错误: {e}")
        return 0.0

def compute_pd_fa(pred, gt_mask, size):
    """基于权威metrics.py的PD和FA计算方法
    Args:
        pred: 预测热图 (numpy array)
        gt_mask: GT二值掩码 (numpy array) 
        size: 图像尺寸 (H, W)
    Returns:
        pd_score: PD值
        fa_score: FA值
    """
    try:
        # 确保数据是numpy数组并二值化
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
            
        # 二值化 (使用配置参数) - 修复：对GT也进行二值化
        DetectTh = MetricConfig.DETECTION_THRESHOLD
        output_one = pred.copy()
        target_one = gt_mask.copy()
        
        # 预测二值化
        output_one[np.where(output_one < DetectTh)] = 0
        output_one[np.where(output_one >= DetectTh)] = 1
        
        # GT二值化 - 关键修复！
        target_one[np.where(target_one < DetectTh)] = 0
        target_one[np.where(target_one >= DetectTh)] = 1
        
        # 确保数据类型正确
        target_one = target_one.astype(np.int32)
        output_one = output_one.astype(np.int32)
        
        # 连通域分析 (与权威方法一致)
        labelimage = measure.label(target_one, connectivity=2)
        props = measure.regionprops(labelimage, intensity_image=target_one, cache=True)
        
        TgtNum = len(props)
        TrueNum = 0
        FalseNum = 0
        
        # 边界情况处理
        if TgtNum == 0:
            if np.sum(output_one) == 0:
                return 1.0, 0.0  # 无目标无预测
            else:
                fa_score = float(np.sum(output_one) / (size[0] * size[1]))
                return 0.0, fa_score  # 无目标有预测
        
        # 分离PD和FA计算 - 修复算法错误  
        LocLen1 = MetricConfig.LOCLEN1   # 调整：PD搜索半径，接近P/R的10像素阈值
        LocLen2 = MetricConfig.LOCLEN2  # FA排除半径
        
        # 第1步：PD计算 - 严格按照权威方法
        for i_tgt in range(len(props)):
            True_flag = 0
            pixel_coords = props[i_tgt].coords
            
            # 对目标的每个像素检查LocLen1范围内是否有预测
            for i_pixel in pixel_coords:
                r_min = max(0, i_pixel[0] - LocLen1)
                r_max = min(output_one.shape[0], i_pixel[0] + LocLen1 + 1)
                c_min = max(0, i_pixel[1] - LocLen1)
                c_max = min(output_one.shape[1], i_pixel[1] + LocLen1 + 1)
                
                Tgt_area = output_one[r_min:r_max, c_min:c_max]
                if Tgt_area.sum() >= 1:
                    True_flag = 1
                    break  # 找到就退出，避免重复计算
            
            if True_flag == 1:
                TrueNum += 1
        
        # 第2步：单独计算FA排除区域
        Box2_map = np.ones(output_one.shape)
        for i_tgt in range(len(props)):
            pixel_coords = props[i_tgt].coords
            # 为FA计算设置排除区域
            for i_pixel in pixel_coords:
                r_min = max(0, i_pixel[0] - LocLen2)
                r_max = min(output_one.shape[0], i_pixel[0] + LocLen2 + 1)
                c_min = max(0, i_pixel[1] - LocLen2) 
                c_max = min(output_one.shape[1], i_pixel[1] + LocLen2 + 1)
                Box2_map[r_min:r_max, c_min:c_max] = 0
        
        # FA计算 - 权威方法
        False_output_one = output_one * Box2_map
        FalseNum = np.count_nonzero(False_output_one)
        
        # 计算最终指标
        pd_score = TrueNum / TgtNum if TgtNum > 0 else 0.0
        fa_score = FalseNum / (size[0] * size[1])
        
        return float(pd_score), float(fa_score)
        
    except Exception as e:
        print(f"PD/FA计算错误: {e}")
        return 0.0, 0.0

# ======================== 综合指标评估封装 ========================

def compute_sequence_auc_with_shooting_rules(seq_pred, seq_centroid):
    """使用PD-FA曲线计算序列级别的AUC（完全参照metrics.py）
    
    Args:
        seq_pred: 序列预测结果 (T, H, W) 
        seq_centroid: 序列centroid掩码 (T, H, W)
        
    Returns:
        auc_score: PD-FA曲线AUC值
    """
    try:
        # 定义阈值序列（完全参照metrics.py）
        Th_Seg = np.array([0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 
                          0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
                          0.8, 0.85, 0.9, 1])
        
        # 初始化累积统计（完全参照metrics.py）
        FalseNumAll = np.zeros(len(Th_Seg))
        TrueNumAll = np.zeros(len(Th_Seg))
        TgtNumAll = np.zeros(len(Th_Seg))
        pixelsNumber = seq_pred.shape[0] * seq_pred.shape[1] * seq_pred.shape[2]
        
        # 对每一帧累积统计（使用我们的内部实现）
        for t in range(seq_pred.shape[0]):
            pred_frame = seq_pred[t]      # (H, W)
            gt_frame = seq_centroid[t]    # (H, W)
            
            for th_i in range(len(Th_Seg)):
                FalseNum, TrueNum, TgtNum = compute_shooting_rules_single(pred_frame, gt_frame, Th_Seg[th_i])
                FalseNumAll[th_i] += FalseNum
                TrueNumAll[th_i] += TrueNum  
                TgtNumAll[th_i] += TgtNum
        
        # 计算PD/FA（完全参照metrics.py）
        Pd_seq = TrueNumAll / (TgtNumAll + 1e-8)  # 避免除零
        Fa_seq = FalseNumAll / pixelsNumber
        
        # 检查有效性
        if len(np.unique(Pd_seq)) < 2:
            # PD全相同，无法计算有效AUC
            if Pd_seq[0] == 0:
                return 0.0  # 完全漏检
            elif Pd_seq[0] == 1:
                return 1.0  # 完美检测
            else:
                return 0.5  # 其他情况
        
        # 计算AUC（完全参照metrics.py）
        try:
            auc_score = auc(Fa_seq, Pd_seq)
            
            # 边界保护
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
    """综合指标评估的统一接口 - 混合方案
    
    Args:
        seq_predictions: 序列预测结果列表 [seq1(T,H,W), seq2(T,H,W), ...]
        seq_targets: 序列目标掩码列表 [seq1(T,H,W), seq2(T,H,W), ...]  
        seq_centroids: 序列centroid掩码列表 [seq1(T,H,W), seq2(T,H,W), ...]
        dataset_info: 数据集信息字典 {'seq_names': [...], 'dataset_name': 'xxx'}
        
    Returns:
        results: 指标结果字典
    """
    results = {}
    seq_names = dataset_info.get('seq_names', [f'Seq{i}' for i in range(len(seq_predictions))])
    dataset_name = dataset_info.get('dataset_name', 'Unknown')
    
    # 为每个序列计算指标，并可选保存 ROC 曲线
    for seq_idx, seq_name in enumerate(seq_names):
        if seq_idx >= len(seq_predictions):
            continue
            
        seq_pred = seq_predictions[seq_idx]   # (T, H, W)
        seq_target = seq_targets[seq_idx]     # (T, H, W)  
        seq_centroid = seq_centroids[seq_idx] # (T, H, W)
        
        # 累积帧级别指标
        frame_metrics = {
            'precision': [], 'recall': [], 'f1': [],
            'pd': [], 'fa': [], 'miou': []
        }
        
        # 对每一帧计算指标
        for t in range(seq_pred.shape[0]):
            pred_frame = seq_pred[t]      # (H, W)
            target_frame = seq_target[t]  # (H, W)
            centroid_frame = seq_centroid[t]  # (H, W)
            
            # 使用metric.py中封装的方法计算各项指标
            # 获取关键点
            pred_keypoints = get_keypoints(pred_frame)
            gt_keypoints = get_keypoints(centroid_frame)
            
            # 计算基于关键点的指标（Precision/Recall/F1）
            precision = compute_metric(None, pred_frame, pred_keypoints, gt_keypoints, 
                                     'Precision', MetricConfig.DISTANCE_THRESHOLD)
            recall = compute_metric(None, pred_frame, pred_keypoints, gt_keypoints, 
                                  'Recall', MetricConfig.DISTANCE_THRESHOLD)
            f1 = compute_metric(None, pred_frame, pred_keypoints, gt_keypoints, 
                               'F1', MetricConfig.DISTANCE_THRESHOLD)
            
            # 使用metric.py中的封装方法计算其他指标
            miou_score = compute_miou(pred_frame, target_frame)
            
            # PD/FA使用metric.py的方法
            size = pred_frame.shape
            pd_score, fa_score = compute_pd_fa(pred_frame, centroid_frame, size)
            
            # 收集指标
            frame_metrics['precision'].append(precision)
            frame_metrics['recall'].append(recall)
            frame_metrics['f1'].append(f1)
            frame_metrics['pd'].append(pd_score)
            frame_metrics['fa'].append(fa_score)
            frame_metrics['miou'].append(miou_score)
        
        # 使用原始方法计算序列级别的AUC（基于Shooting Rules + PD/FA-ROC）
        auc_score = compute_sequence_auc_with_shooting_rules(seq_pred, seq_centroid)

        # 可选保存 ROC/PD-FA 曲线
        if roc_save_dir is not None:
            try:
                save_sequence_pd_fa_curve(seq_pred, seq_centroid,
                                          save_dir=roc_save_dir,
                                          seq_name=seq_name)
            except Exception as _e:
                # 打印一次警告，继续流程
                print(f"[warn] 保存 {seq_name} ROC 失败: {_e}")
        
        # 计算序列平均值
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
    
    # 计算整体统计（如果是NUDT-MIRSDT数据集，计算低/高SNR统计）
    if 'NUDT-MIRSDT' in dataset_name:
        # NUDT-MIRSDT的低/高SNR序列索引
        if 'Noise' in dataset_name:
            # Noise版本的序列分类
            all_aucs = [results[seq_name]['auc'] for seq_name in results.keys() 
                       if seq_name.startswith('Sequence')]
            results['low_snr_auc'] = np.mean(all_aucs) if all_aucs else 0.0
            results['high_snr_auc'] = np.mean(all_aucs) if all_aucs else 0.0
            results['overall_auc'] = np.mean(all_aucs) if all_aucs else 0.0
        else:
            # 原始NUDT-MIRSDT的分类
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
        
        # 计算平均mIoU
        all_mious = [results[seq_name]['miou'] for seq_name in results.keys() 
                    if seq_name.startswith('Sequence')]
        results['avg_miou'] = np.mean(all_mious) if all_mious else 0.0
    
    return results

def print_evaluation_table(results, dataset_name):
    """打印漂亮的指标表格
    
    Args:
        results: evaluate_comprehensive_metrics的返回结果
        dataset_name: 数据集名称
    """
    print("\n" + "="*120)
    print(f"Results for {dataset_name}")
    print("="*120)
    
    # 表头
    print(f"+{'-'*12}+{'-'*6}+{'-'*12}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+")
    print(f"|{'Seq':<12}|{'Num':<6}|{'Precision':<12}|{'Recall':<8}|{'F1':<8}|{'AUC':<8}|{'PD':<8}|{'FA':<8}|{'mIoU':<8}|")
    print(f"+{'-'*12}+{'-'*6}+{'-'*12}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+")
    
    # 序列结果
    overall_metrics = {
        'precision': [], 'recall': [], 'f1': [], 'auc': [], 
        'pd': [], 'fa': [], 'miou': [], 'num': []
    }
    
    # 显示所有序列结果（排除汇总统计）
    for seq_name, metrics in results.items():
        # 过滤掉汇总统计项（如 low_snr_auc, high_snr_auc, overall_auc, avg_miou）
        if seq_name in ['low_snr_auc', 'high_snr_auc', 'overall_auc', 'avg_miou']:
            continue
            
        # 确保 metrics 是字典且包含必要的键
        if not isinstance(metrics, dict) or 'precision' not in metrics:
            continue
            
        # 支持所有数据集的序列名称：
        # - Sequence*, Seq* (NUDT-MIRSDT等)
        # - TSIRMT, DSAT, SIATD (单序列数据集)
        # - 00001, 00002, ... (数字序号序列，如TSIRMT的帧级别处理)
        if not (seq_name[0].isalpha() or seq_name[0].isdigit()):
            continue
            
        print(f"|{seq_name:<12}|{metrics['num']:>6}|"
              f"{metrics['precision']:>12.4f}|{metrics['recall']:>8.4f}|"
              f"{metrics['f1']:>8.4f}|{metrics['auc']:>8.4f}|"
              f"{metrics['pd']:>8.4f}|{metrics['fa']:>8.6f}|"
              f"{metrics['miou']:>8.4f}|")
        
        # 收集总体统计
        for key in ['precision', 'recall', 'f1', 'auc', 'pd', 'fa', 'miou', 'num']:
            overall_metrics[key].append(metrics[key])
    
    print(f"+{'-'*12}+{'-'*6}+{'-'*12}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+")
    
    # 总体结果
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
    
    # 如果有分类结果，显示低/高SNR统计
    if 'low_snr_auc' in results:
        print(f"\nLow SNR Sequences AUC: {results['low_snr_auc']:.5f}")
        print(f"High SNR Sequences AUC: {results['high_snr_auc']:.5f}")
        print(f"Overall AUC: {results['overall_auc']:.5f}")
        print(f"Average Class IoU: {results.get('avg_miou', 0.0):.5f}")
    
    print("="*120)

# === 新增: 序列ROC曲线保存工具 ===

def save_sequence_pd_fa_curve(seq_pred, seq_gt, save_dir, seq_name,
                              *, figsize=(6,5), dpi=300):
    """为整个序列保存一张 PD-FA (ROC) 曲线 PNG（完全参照metrics.py逻辑）
    
    正确方式：先累积统计量(TrueNum, FalseNum, TgtNum)，再计算PD/FA
    参考：metrics.py 第83-85行
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义阈值序列（与AUC计算完全一致）
    Th_Seg = np.array([0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 
                      0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
                      0.8, 0.85, 0.9, 1])
    
    # 初始化累积统计数组（完全参照metrics.py）
    FalseNumAll = np.zeros(len(Th_Seg))
    TrueNumAll = np.zeros(len(Th_Seg))
    TgtNumAll = np.zeros(len(Th_Seg))
    pixelsNumber = seq_pred.shape[0] * seq_pred.shape[1] * seq_pred.shape[2]
    
    # 逐帧累积统计（与compute_sequence_auc_with_shooting_rules一致）
    for t in range(seq_pred.shape[0]):
        pred_frame = seq_pred[t]
        gt_frame = seq_gt[t]
        
        for th_i in range(len(Th_Seg)):
            FalseNum, TrueNum, TgtNum = compute_shooting_rules_single(
                pred_frame, gt_frame, Th_Seg[th_i], debug_seq_name=seq_name)
            FalseNumAll[th_i] += FalseNum
            TrueNumAll[th_i] += TrueNum
            TgtNumAll[th_i] += TgtNum
    
    # 计算PD/FA（完全参照metrics.py第83-85行）
    # 先累加统计量，再相除（而不是平均每帧的PD/FA）
    PD_array = TrueNumAll / (TgtNumAll + 1e-8)  # 避免除零
    FA_array = FalseNumAll / pixelsNumber
    
    # 绘制曲线
    import matplotlib
    matplotlib.use('Agg')  # 非GUI后端
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
    
    # 打印调试信息
    auc_val = np.trapz(PD_sorted, FA_sorted) if len(np.unique(PD_sorted)) > 1 else 0.0
    print(f"  [{seq_name}] TgtNum={TgtNumAll.sum():.0f}, "
          f"PD_range=[{PD_array.min():.3f}, {PD_array.max():.3f}], "
          f"FA_range=[{FA_array.min():.6f}, {FA_array.max():.6f}], "
          f"AUC={auc_val:.4f}")
    
    return save_path


# ======================== 统一的两级别评估类 ========================

class TwoLevelMetrics:
    """两级别评估类：累积级别、序列级别
    
    使用方式：
        metric = TwoLevelMetrics()
        
        # 对每个序列
        for seq in sequences:
            metric.start_sequence()
            
            # 对序列中的每一帧
            for frame in seq:
                metric.update(pred, gt)
            
            metric.end_sequence()
        
        # 获取结果
        results = metric.get_results()
    """
    
    def __init__(self):
        """初始化两级别评估器"""
        # 累积级别（Accumulated）- 全局累积
        self.accumulated_miou = mIoU()
        self.accumulated_pdfa = PD_FA()
        
        # 序列级别（Sequence）- 每序列平均后再求平均
        self.seq_metrics_list = []  # 存储每个序列的平均指标
        self.current_seq_metrics = None  # 当前序列的帧指标
        
        # 计数器
        self.total_frames = 0
        self.total_sequences = 0
        
    def start_sequence(self):
        """开始一个新序列的评估"""
        self.current_seq_metrics = {
            'ious': [],
            'pds': [],
            'fas': [],
            'precisions': [],
            'recalls': [],
            'f1s': []
        }
        
    def update(self, pred_tensor, gt_tensor, size=(256, 256)):
        """更新指标（处理一帧）
        
        Args:
            pred_tensor: 预测tensor [1, 1, H, W]，已经是logits格式
            gt_tensor: GT tensor [1, 1, H, W]，值为0-1
            size: 图像尺寸，用于PD_FA计算
        """
        # === 1. 累积级别更新 ===
        self.accumulated_miou.update(pred_tensor, gt_tensor)
        self.accumulated_pdfa.update(pred_tensor, gt_tensor, size)
        
        # === 2. 序列级别计算（单独计算每帧，用于序列平均） ===
        # 转换为numpy进行计算
        pred_np = torch.sigmoid(pred_tensor).cpu().numpy().squeeze()  # sigmoid后转为概率
        gt_np = gt_tensor.cpu().numpy().squeeze()
        
        # IoU计算
        pred_binary = (pred_np > 0.5).astype(np.float32)
        gt_binary = (gt_np > 0.5).astype(np.float32)
        
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        frame_iou = intersection / (union + 1e-8)
        
        # PD/FA计算（使用metric.py的compute_pd_fa）
        frame_pd, frame_fa = compute_pd_fa(pred_np, gt_np, size)
        
        # Precision/Recall/F1计算（基于关键点）
        pred_keypoints = get_keypoints(pred_np)
        gt_keypoints = get_keypoints(gt_np)
        precision, recall, _ = compute_prfa(pred_keypoints, gt_keypoints, MetricConfig.DISTANCE_THRESHOLD)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        
        # 存储到当前序列（如果正在处理序列）
        if self.current_seq_metrics is not None:
            self.current_seq_metrics['ious'].append(frame_iou)
            self.current_seq_metrics['pds'].append(frame_pd)
            self.current_seq_metrics['fas'].append(frame_fa)
            self.current_seq_metrics['precisions'].append(precision)
            self.current_seq_metrics['recalls'].append(recall)
            self.current_seq_metrics['f1s'].append(f1)
        
        self.total_frames += 1
        
    def end_sequence(self):
        """结束当前序列的评估"""
        if self.current_seq_metrics is not None:
            # 计算当前序列的平均指标
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
        """获取两级别的评估结果
        
        Returns:
            dict: 包含两个级别的指标结果
        """
        # 1. 累积级别
        acc_pixacc, acc_miou = self.accumulated_miou.get()
        acc_pd, acc_fa = self.accumulated_pdfa.get()
        
        # 2. 序列级别
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
        """重置所有统计"""
        self.accumulated_miou.reset()
        self.accumulated_pdfa.reset()
        self.seq_metrics_list = []
        self.current_seq_metrics = None
        self.total_frames = 0
        self.total_sequences = 0


def print_two_level_results(results, dataset_name='Unknown', pred_dir=None):
    """打印两级别评估结果的详细表格
    
    Args:
        results: TwoLevelMetrics.get_results()的返回值
        dataset_name: 数据集名称
        pred_dir: 预测目录（用于显示）
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
    
    # 表格表头
    print("\n" + "-" * 120)
    print(f"{'评估级别':<20} | {'mIoU':<10} | {'PD':<10} | {'FA(×10⁻⁶)':<12} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'说明':<20}")
    print("-" * 120)
    
    # 累积级别
    print(f"{'累积级别':<20} | {acc['miou']:<10.4f} | {acc['pd']:<10.4f} | "
          f"{acc['fa']*1e6:<12.2f} | {'-':<10} | {'-':<10} | {'-':<10} | {'累积I∩U计算':<20}")
    
    # 序列级别
    print(f"{'序列级别':<20} | {seq['miou']:<10.4f} | {seq['pd']:<10.4f} | "
          f"{seq['fa']*1e6:<12.2f} | {seq['precision']:<10.4f} | {seq['recall']:<10.4f} | {seq['f1']:<10.4f} | {'序列平均后再平均':<20}")
    
    print("-" * 120)