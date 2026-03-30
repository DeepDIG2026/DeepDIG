#!/usr/bin/env python3
"""
从已保存的预测结果计算指标（两级别评估）
不需要重新运行推理，直接读取logits/bin_mask文件
完全使用model/metric.py的计算方式

评估级别：
1. 累积级别（Accumulated-level）：累积所有帧的I∩U后计算
   - 指标：mIoU, PD, FA, PixAcc
   - 特点：与训练时的累积计算完全一致
   
2. 序列级别（Sequence-level）：每序列平均后再求平均
   - 指标：mIoU, PD, FA, Precision, Recall, F1, AUC
   - 特点：每个序列权重相同，避免长序列主导结果

支持数据集：
- TSIRMT
- IRDST  
- IRSTD-1k
- DSAT
- SIATD
- BUPT-MIRSDT
- NUDT-MIRSDT
- NUDT-MIRSDT-Noise

使用示例：
    # TSIRMT数据集（自动路径）
    python cal.py --dataset TSIRMT
    
    # IRDST数据集（自动路径）
    python cal.py --dataset IRDST
    
    # BUPT-MIRSDT数据集（自动路径）
    python cal.py --dataset BUPT-MIRSDT
    
    # 自定义路径
    python cal.py \
        --dataset TSIRMT \
        --pred_dir ./predict/TSIRMT \
        --dataset_root /path/to/dataset/TSIRMT
    
    # 调整阈值参数（默认已统一为3像素）
    python cal.py \
        --dataset TSIRMT \
        --detection_threshold 0.5 \
        --distance_threshold 3 \
        --pd_search_radius 3 \
        --fa_exclude_radius 3
"""
import os
import sys
import cv2
import numpy as np
import glob
from tqdm import tqdm
import csv
import scipy.io as scio
from PIL import Image
import torch

# 导入model/metric.py中的所有评估函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从model/metric.py导入
from model.metric import (
    MetricConfig,
    get_keypoints,
    compute_prfa,
    compute_pd_fa,
    compute_miou,
    compute_auc,
    compute_sequence_auc_with_shooting_rules,
    mIoU,
    PD_FA
)

def calculate_metrics_from_saved_results(
    pred_dir, 
    dataset_root, 
    use_logits=True, 
    dataset_name='TSIRMT',
    detection_threshold=0.5,
    distance_threshold=3,
    iou_threshold=0.5,
    pd_search_radius=3,
    fa_exclude_radius=3
):
    """
    从保存的预测结果计算指标（完全使用model/metric.py的方法）
    
    Args:
        pred_dir: 预测结果目录（如 ./predict/TSIRMT）
        dataset_root: 数据集根目录（如 ./Datasets/TSIRMT）
        use_logits: True使用logits（热图），False使用bin_mask（二值图）
        dataset_name: 数据集名称
        detection_threshold: 二值化阈值 (默认0.5)
        distance_threshold: P/R/F1的距离匹配阈值 (默认3像素，统一)
        iou_threshold: IoU计算的二值化阈值 (默认0.5)
        pd_search_radius: PD搜索半径 (默认3像素，统一)
        fa_exclude_radius: FA排除半径 (默认3像素，统一)
    """
    print("=" * 80)
    print("从保存的预测结果计算指标（使用model/metric.py方法）")
    print("=" * 80)
    print(f"预测目录: {pred_dir}")
    print(f"数据集目录: {dataset_root}")
    print(f"使用: {'logits (热图)' if use_logits else 'bin_mask (二值图)'}")
    print(f"数据集: {dataset_name}")
    print(f"\n📐 阈值配置:")
    print(f"  - 检测阈值(detection): {detection_threshold}")
    print(f"  - IoU阈值(iou): {iou_threshold}")
    print(f"  - 距离阈值(P/R/F1): {distance_threshold}像素")
    print(f"  - PD搜索半径: {pd_search_radius}像素")
    print(f"  - FA排除半径: {fa_exclude_radius}像素")
    print("=" * 80)
    
    # 临时设置MetricConfig的值
    original_config = {
        'DETECTION_THRESHOLD': MetricConfig.DETECTION_THRESHOLD,
        'IOU_THRESHOLD': MetricConfig.IOU_THRESHOLD,
        'DISTANCE_THRESHOLD': MetricConfig.DISTANCE_THRESHOLD,
        'LOCLEN1': MetricConfig.LOCLEN1,
        'LOCLEN2': MetricConfig.LOCLEN2,
    }
    
    MetricConfig.DETECTION_THRESHOLD = detection_threshold
    MetricConfig.IOU_THRESHOLD = iou_threshold
    MetricConfig.DISTANCE_THRESHOLD = distance_threshold
    MetricConfig.LOCLEN1 = pd_search_radius
    MetricConfig.LOCLEN2 = fa_exclude_radius
    
    # 路径设置
    pred_path = pred_dir
    gt_path = os.path.join(dataset_root, 'masks')
    
    # 获取所有序列
    sequences = sorted([d for d in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path, d))])
    print(f"\n找到 {len(sequences)} 个序列")
    
    # 存储所有序列的预测和GT（用于计算序列级别的AUC）
    all_seq_predictions = []
    all_seq_gts = []
    all_seq_names = []
    
    # 存储每个序列的指标
    vid_metrics_dict = {}
    
    # 1. 累积级别（Accumulated）- mIoU计算
    acc_total_inter = 0
    acc_total_union = 0
    acc_total_correct = 0
    acc_total_label = 0
    
    # 2. 帧级别（Frame-level）- 直接累积每帧的指标值
    frame_ious = []  # 每帧IoU
    frame_precisions = []  # 每帧Precision
    frame_recalls = []  # 每帧Recall
    frame_pds = []  # 每帧PD
    frame_fas = []  # 每帧FA
    frame_aucs = []  # 每个序列的AUC（仍需按序列计算）
    
    # PD/FA累积统计（用于累积级别）
    acc_pd_target = 0
    acc_pd_detected = 0
    acc_fa_false_pixels = 0
    acc_fa_total_pixels = 0
    
    # 逐序列处理
    for seq_name in tqdm(sequences, desc="计算指标"):
        seq_pred_dir = os.path.join(pred_path, seq_name)
        seq_gt_dir = os.path.join(gt_path, seq_name)
        
        if not os.path.exists(seq_gt_dir):
            print(f"⚠️  跳过 {seq_name}：GT目录不存在")
            continue
        
        # 获取所有预测文件（兼容多种扩展名）
        pred_files = []
        for pattern in ('*.bmp', '*.png', '*.jpg'):
            pred_files.extend(glob.glob(os.path.join(seq_pred_dir, pattern)))
        pred_files = sorted(pred_files)
        
        if len(pred_files) == 0:
            print(f"⚠️  跳过 {seq_name}：没有预测文件")
            continue
        
        # 序列级别指标存储
        seq_frame_metrics = {
            'ious': [], 'precisions': [], 'recalls': [], 'f1s': [],
            'pds': [], 'fas': []
        }
        seq_preds = []
        seq_gts = []
        
        # 逐帧处理
        for pred_file in pred_files:
            frame_name = os.path.splitext(os.path.basename(pred_file))[0]
            # 尝试匹配不同扩展名的GT
            gt_file = None
            for ext in ('.bmp', '.png', '.jpg', '.jpeg'):
                candidate = os.path.join(seq_gt_dir, frame_name + ext)
                if os.path.exists(candidate):
                    gt_file = candidate
                    break
            if gt_file is None:
                continue
            
            # 读取预测和GT
            pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
            gt_cv = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            if pred is None or gt_cv is None:
                continue

            # ⭐ 统一在原始尺寸上计算（不resize）
            # 预测：binary文件归一化到[0,1]
            pred_norm = np.clip(pred.astype(np.float32) / 255.0, 0, 1)
            
            # GT：归一化并使用>=0.5阈值二值化
            gt_norm = (gt_cv.astype(np.float32) / 255.0 >= 0.5).astype(np.float32)
            
            # 收集序列数据（用于AUC计算）   
            seq_preds.append(pred_norm)
            seq_gts.append(gt_norm)
            
            # ⭐ 累积级别更新（手动累积，直接使用概率值）
            # 二值化（使用0.5阈值，与序列级别一致）
            pred_binary = (pred_norm > 0.5).astype(np.float32)
            gt_binary = (gt_norm > 0.5).astype(np.float32)
            
            # IoU累积
            intersection = np.sum(pred_binary * gt_binary)
            union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
            acc_total_inter += intersection
            acc_total_union += union
            
            # PixAcc累积
            correct = np.sum((pred_binary == gt_binary) * (gt_binary > 0))
            labeled = np.sum(gt_binary > 0)
            acc_total_correct += correct
            acc_total_label += labeled
            
            # PD/FA累积（使用compute_pd_fa，与序列级别一致）
            size = pred_norm.shape
            pd, fa = compute_pd_fa(pred_norm, gt_norm, size)
            
            # 累积PD/FA的统计量（需要原始计数）
            # 使用连通域分析获取目标数和检测数
            gt_binary_int = gt_binary.astype(np.int32)
            pred_binary_int = pred_binary.astype(np.int32)
            
            from skimage import measure
            labelimage = measure.label(gt_binary_int, connectivity=2)
            props = measure.regionprops(labelimage)
            num_targets = len(props)
            
            acc_pd_target += num_targets
            acc_pd_detected += int(pd * num_targets)  # PD比例 × 目标数 = 检测数
            acc_fa_false_pixels += int(fa * size[0] * size[1])  # FA比例 × 总像素 = 虚警像素
            acc_fa_total_pixels += size[0] * size[1]
            
            # 使用metric.py计算所有指标，直接累积到帧级别
            # 1. IoU
            iou = compute_miou(pred_norm, gt_norm)
            if iou is not None:
                seq_frame_metrics['ious'].append(iou)
                frame_ious.append(iou)  # ⭐ 帧级别累积
            
            # 2. PD/FA
            size = pred_norm.shape
            pd, fa = compute_pd_fa(pred_norm, gt_norm, size)
            seq_frame_metrics['pds'].append(pd)
            seq_frame_metrics['fas'].append(fa)
            frame_pds.append(pd)  # ⭐ 帧级别累积
            frame_fas.append(fa)  # ⭐ 帧级别累积
            
            # 3. Precision/Recall
            pred_keypoints = get_keypoints(pred_norm)
            gt_keypoints = get_keypoints(gt_norm)
            precision, recall, falsealarm = compute_prfa(pred_keypoints, gt_keypoints, distance_threshold)
            
            if precision + recall > 0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            seq_frame_metrics['precisions'].append(precision)
            seq_frame_metrics['recalls'].append(recall)
            seq_frame_metrics['f1s'].append(f1)
            frame_precisions.append(precision)  # ⭐ 帧级别累积
            frame_recalls.append(recall)  # ⭐ 帧级别累积
        
        # 计算序列级别的AUC（AUC必须按序列计算）
        if len(seq_preds) > 0:
            seq_pred_array = np.array(seq_preds)
            seq_gt_array = np.array(seq_gts)
            auc_score = compute_sequence_auc_with_shooting_rules(seq_pred_array, seq_gt_array)
            frame_aucs.append(auc_score)  # ⭐ 收集每个序列的AUC
            all_seq_predictions.append(seq_pred_array)
            all_seq_gts.append(seq_gt_array)
            all_seq_names.append(seq_name)
        else:
            auc_score = 0.0
        
        # 保存序列指标
        vid_metrics_dict[seq_name] = {
            'frames': len(pred_files),
            'valid_frames': len(seq_frame_metrics['ious']),
            'miou': np.mean(seq_frame_metrics['ious']) if len(seq_frame_metrics['ious']) > 0 else 0.0,
            'precision': np.mean(seq_frame_metrics['precisions']) if len(seq_frame_metrics['precisions']) > 0 else 0.0,
            'recall': np.mean(seq_frame_metrics['recalls']) if len(seq_frame_metrics['recalls']) > 0 else 0.0,
            'f1': np.mean(seq_frame_metrics['f1s']) if len(seq_frame_metrics['f1s']) > 0 else 0.0,
            'pd': np.mean(seq_frame_metrics['pds']) if len(seq_frame_metrics['pds']) > 0 else 0.0,
            'fa': np.mean(seq_frame_metrics['fas']) if len(seq_frame_metrics['fas']) > 0 else 0.0,
            'auc': auc_score
        }
        
        # 打印序列指标
        m = vid_metrics_dict[seq_name]
        print(f"video:{seq_name} | IoU:{m['miou']:.5f} ({m['valid_frames']}/{m['frames']}帧) | "
              f"P:{m['precision']:.4f} R:{m['recall']:.4f} F1:{m['f1']:.4f} | "
              f"PD:{m['pd']:.4f} FA:{m['fa']:.6f} | AUC:{m['auc']:.4f}")
    
    # ========== 1. 累积级别（Accumulated）- mIoU ==========
    acc_miou = acc_total_inter / (acc_total_union + 1e-8)
    acc_pixacc = acc_total_correct / (acc_total_label + 1e-8)
    acc_pd = acc_pd_detected / (acc_pd_target + 1e-8)
    acc_fa = acc_fa_false_pixels / (acc_fa_total_pixels + 1e-8)
    
    # ========== 2. 帧级别平均（Frame-level Average）==========
    # nIoU: 每帧IoU的平均
    niou = np.mean(frame_ious) if len(frame_ious) > 0 else 0.0
    
    # Precision/Recall: 所有帧的平均
    frame_avg_precision = np.mean(frame_precisions) if len(frame_precisions) > 0 else 0.0
    frame_avg_recall = np.mean(frame_recalls) if len(frame_recalls) > 0 else 0.0
    
    # F1: 从帧平均的Precision和Recall计算
    if frame_avg_precision + frame_avg_recall > 0:
        frame_avg_f1 = 2.0 * frame_avg_precision * frame_avg_recall / (frame_avg_precision + frame_avg_recall)
    else:
        frame_avg_f1 = 0.0
    
    # PD/FA: 所有帧的平均
    frame_avg_pd = np.mean(frame_pds) if len(frame_pds) > 0 else 0.0
    frame_avg_fa = np.mean(frame_fas) if len(frame_fas) > 0 else 0.0
    
    # AUC: 所有序列的平均（AUC必须按序列计算）
    frame_avg_auc = np.mean(frame_aucs) if len(frame_aucs) > 0 else 0.0
    
    # 打印汇总（累积级别 + 帧级别平均）
    print("\n" + "=" * 120)
    print("两种计算方式评估结果")
    print("=" * 120)
    
    # 表头
    print(f"{'计算方式':<20} | {'mIoU':<10} | {'nIoU':<10} | {'PD':<10} | {'FA(×10⁻⁶)':<12} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'AUC':<10} | {'说明':<20}")
    print("-" * 120)
    
    # 1. 累积级别（mIoU）
    print(f"{'累积 (mIoU)':<20} | {acc_miou:<10.4f} | {'-':<10} | {acc_pd:<10.4f} | "
          f"{acc_fa*1e6:<12.2f} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<10} | {'累积Σ(I∩U)/Σ(U)':<20}")
    
    # 2. 帧平均（nIoU）
    print(f"{'帧平均 (nIoU)':<20} | {'-':<10} | {niou:<10.4f} | {frame_avg_pd:<10.4f} | "
          f"{frame_avg_fa*1e6:<12.2f} | {frame_avg_precision:<10.4f} | {frame_avg_recall:<10.4f} | {frame_avg_f1:<10.4f} | {frame_avg_auc:<10.4f} | {'所有帧直接平均':<20}")
    
    print("-" * 120)
    
    print("\n" + "=" * 120)
    print("📊 计算方式详解")
    print("=" * 120)
    print("\n1️⃣  累积计算 - mIoU (Accumulated IoU)")
    print("   计算方式: 累积所有帧的 intersection 和 union，最后一起计算")
    print("   公式: mIoU = Σ(intersection) / Σ(union)")
    print("   特点: 与训练时的累积计算完全一致，是论文中通常报告的主指标")
    print(f"   结果: mIoU={acc_miou:.4f}, PD={acc_pd:.4f}, FA={acc_fa*1e6:.2f}×10⁻⁶")
    print(f"   说明: 累积计算不计算Precision/Recall/F1/AUC（需要逐帧关键点匹配）")
    
    print("\n2️⃣  帧平均计算 - nIoU (Frame Average)")
    print("   计算方式: 对每帧计算指标，然后对所有帧求平均")
    print("   公式: nIoU = mean([iou₁, iou₂, ..., iouₙ])")
    print(f"         P_avg = mean([p₁, p₂, ..., pₙ])")
    print(f"         R_avg = mean([r₁, r₂, ..., rₙ])")
    print(f"         F1 = 2×P_avg×R_avg/(P_avg+R_avg)")
    print("   特点: 每帧权重相同，更关注单帧检测质量")
    print(f"   结果: nIoU={niou:.4f}, PD={frame_avg_pd:.4f}, FA={frame_avg_fa*1e6:.2f}×10⁻⁶")
    print(f"          Precision={frame_avg_precision:.4f}, Recall={frame_avg_recall:.4f}")
    print(f"          F1={frame_avg_f1:.4f} (从帧平均P/R计算), AUC={frame_avg_auc:.4f}")
    print(f"\n   统计信息: 共 {len(frame_ious)} 帧参与计算")
    
    # 添加详细序列结果表格
    print("\n" + "=" * 120)
    print("详细序列结果")
    print("=" * 120)
    
    # 表格标题行
    seq_header = "+---------+-----+-----------+--------+--------+--------+----------+--------+--------+"
    print(seq_header)
    print(f"|   Seq   | Num | Precision | Recall |   F1   |   PD   |    FA    |  mIoU  |  AUC   |")
    print(seq_header)
    
    # 按序列名排序显示
    for seq_name in sorted(vid_metrics_dict.keys()):
        m = vid_metrics_dict[seq_name]
        print(f"| {seq_name:>7} | {m['frames']:>3} | {m['precision']:>9.4f} | {m['recall']:>6.4f} | {m['f1']:>6.4f} | {m['pd']:>6.4f} | {m['fa']:>8.6f} | {m['miou']:>6.4f} | {m['auc']:>6.4f} |")
    
    # 整体统计行
    overall_precision = np.mean([m['precision'] for m in vid_metrics_dict.values()]) if vid_metrics_dict else 0.0
    overall_recall = np.mean([m['recall'] for m in vid_metrics_dict.values()]) if vid_metrics_dict else 0.0
    overall_f1 = np.mean([m['f1'] for m in vid_metrics_dict.values()]) if vid_metrics_dict else 0.0
    overall_pd = np.mean([m['pd'] for m in vid_metrics_dict.values()]) if vid_metrics_dict else 0.0
    overall_fa = np.mean([m['fa'] for m in vid_metrics_dict.values()]) if vid_metrics_dict else 0.0
    overall_miou = np.mean([m['miou'] for m in vid_metrics_dict.values()]) if vid_metrics_dict else 0.0
    overall_auc = np.mean([m['auc'] for m in vid_metrics_dict.values()]) if vid_metrics_dict else 0.0
    total_frames = sum([m['frames'] for m in vid_metrics_dict.values()])
    
    print(f"| Overall | {total_frames:>3} | {overall_precision:>9.4f} | {overall_recall:>6.4f} | {overall_f1:>6.4f} | {overall_pd:>6.4f} | {overall_fa:>8.6f} | {overall_miou:>6.4f} | {overall_auc:>6.4f} |")
    print(seq_header)
    print("=" * 120)
    
    # 保存.mat文件（包含累积和帧平均两种计算方式）
    mat_path = os.path.join(pred_dir, f'{dataset_name}_metrics.mat')
    scio.savemat(mat_path, {
        # 累积计算 - mIoU
        'mIoU': float(acc_miou),
        'accumulated_PD': float(acc_pd),
        'accumulated_FA': float(acc_fa),
        'accumulated_PixAcc': float(acc_pixacc),
        # 帧平均计算 - nIoU
        'nIoU': niou,
        'frame_avg_PD': frame_avg_pd,
        'frame_avg_FA': frame_avg_fa,
        'frame_avg_Precision': frame_avg_precision,
        'frame_avg_Recall': frame_avg_recall,
        'frame_avg_F1': frame_avg_f1,
        'frame_avg_AUC': frame_avg_auc,
        # 统计信息
        'total_frames': len(frame_ious)
    })
    print(f"\n✅ .mat文件已保存: {mat_path}")
    
    # 保存CSV（详细结果）
    csv_path = os.path.join(pred_dir, 'video_metrics_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 表头
        writer.writerow(['Video', 'Frames', 'Valid_Frames', 'IoU', 'Precision', 'Recall', 'F1', 'PD', 'FA', 'AUC'])
        
        # 各序列数据
        for seq_name in sorted(vid_metrics_dict.keys()):
            m = vid_metrics_dict[seq_name]
            writer.writerow([
                seq_name,
                m['frames'],
                m['valid_frames'],
                f"{m['miou']:.5f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['pd']:.5f}",
                f"{m['fa']:.6f}",
                f"{m['auc']:.5f}"
            ])
        
        # 整体统计（序列级别）
        total_frames = sum([m['frames'] for m in vid_metrics_dict.values()])
        total_valid = sum([m['valid_frames'] for m in vid_metrics_dict.values()])
        writer.writerow([
            'Overall (Frame-Avg)',
            total_frames,
            total_valid,
            f"{niou:.5f}",
            f"{frame_avg_precision:.4f}",
            f"{frame_avg_recall:.4f}",
            f"{frame_avg_f1:.4f}",
            f"{frame_avg_pd:.5f}",
            f"{frame_avg_fa:.6f}",
            f"{frame_avg_auc:.5f}"
        ])
        
        # 累积计算统计
        writer.writerow(['', '', '', '', '', '', '', '', '', ''])
        writer.writerow(['累积计算 (mIoU)', '', '', '', '', '', '', '', '', ''])
        writer.writerow(['mIoU', f"{acc_miou:.5f}", '', '', '', '', '', '', '', ''])
        writer.writerow(['PD', f"{acc_pd:.5f}", '', '', '', '', '', '', '', ''])
        writer.writerow(['FA', f"{acc_fa:.6f}", '', '', '', '', '', '', '', ''])
        writer.writerow(['PixAcc', f"{acc_pixacc:.5f}", '', '', '', '', '', '', '', ''])
        
        # 配置参数
        writer.writerow(['', '', '', '', '', '', '', '', '', ''])
        writer.writerow(['配置参数', '值', '', '', '', '', '', '', '', ''])
        writer.writerow(['detection_threshold', detection_threshold, '', '', '', '', '', '', '', ''])
        writer.writerow(['iou_threshold', iou_threshold, '', '', '', '', '', '', '', ''])
        writer.writerow(['distance_threshold', distance_threshold, '', '', '', '', '', '', '', ''])
        writer.writerow(['pd_search_radius', pd_search_radius, '', '', '', '', '', '', '', ''])
        writer.writerow(['fa_exclude_radius', fa_exclude_radius, '', '', '', '', '', '', '', ''])
    
    print(f"✅ CSV结果已保存: {csv_path}")
    
    # 恢复原始MetricConfig设置
    for key, value in original_config.items():
        setattr(MetricConfig, key, value)
    
    print("=" * 80)
    
    # 返回结果（包含累积级别和序列级别）
    total_frames = sum([m['frames'] for m in vid_metrics_dict.values()])
    total_valid = sum([m['valid_frames'] for m in vid_metrics_dict.values()])
    
    return {
        # 累积计算 - mIoU
        'accumulated': {
            'miou': float(acc_miou),
            'pd': float(acc_pd),
            'fa': float(acc_fa),
            'pixacc': float(acc_pixacc)
        },
        # 帧平均计算 - nIoU
        'frame_average': {
            'niou': niou,
            'precision': frame_avg_precision,
            'recall': frame_avg_recall,
            'f1': frame_avg_f1,
            'pd': frame_avg_pd,
            'fa': frame_avg_fa,
            'auc': frame_avg_auc
        },
        'statistics': {
            'total_frames': len(frame_ious),
            'total_valid_frames': total_valid,
            'total_sequences': len(vid_metrics_dict)
        }
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('从保存结果计算指标（完全使用model/metric.py方法）')
    parser.add_argument('--pred_dir', type=str, default='/mnt/c/Users/admin/Desktop/DeepDIG_v3/result/TSIRMT/test_0330-1330/predictions',
                       help='预测结果目录（留空则自动设置）')
    parser.add_argument('--dataset_root', type=str, default='/mnt/c/Users/admin/Desktop/STDMANet-bupt/dataset/TSIRMT',
                       help='数据集根目录（留空则自动设置）')
    parser.add_argument('--use_logits', action='store_true', default=False,
                       help='使用logits（热图）而不是bin_mask')
    parser.add_argument('--dataset', type=str, default='TSIRMT',
                       choices=['TSIRMT', 'IRDST', 'IRSTD-1k', 'DSAT', 'SIATD', 'BUPT-MIRSDT', 'NUDT-MIRSDT', 'NUDT-MIRSDT-Noise'],
                       help='数据集名称')
    
    # 阈值参数（统一设置为3像素）
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                       help='检测二值化阈值 (默认0.5)')
    parser.add_argument('--distance_threshold', type=float, default=3,
                       help='P/R/F1的距离匹配阈值(像素) (默认3，统一)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU计算的二值化阈值 (默认0.5)')
    parser.add_argument('--pd_search_radius', type=int, default=3,
                       help='PD检测搜索半径(像素) (默认3，统一)')
    parser.add_argument('--fa_exclude_radius', type=int, default=3,
                       help='FA计算排除半径(像素) (默认3，统一)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("从保存的预测结果计算指标")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    
    # 自动设置路径（如果未指定）
    if not args.pred_dir:
        args.pred_dir = f'./predict/{args.dataset}'
        print(f"预测目录（自动）: {args.pred_dir}")
    else:
        print(f"预测目录（手动）: {args.pred_dir}")
    
    if not args.dataset_root:
        args.dataset_root = f'/mnt/c/Users/admin/Desktop/STDMANet/dataset/{args.dataset}'
        print(f"数据集根目录（自动）: {args.dataset_root}")
    else:
        print(f"数据集根目录（手动）: {args.dataset_root}")
    
    print(f"使用预测类型: {'logits（热图）' if args.use_logits else 'bin_mask（二值图）'}")
    print(f"\n阈值配置:")
    print(f"  - 检测阈值: {args.detection_threshold}")
    print(f"  - 距离阈值（P/R/F1）: {args.distance_threshold} 像素")
    print(f"  - IoU阈值: {args.iou_threshold}")
    print(f"  - PD搜索半径: {args.pd_search_radius} 像素")
    print(f"  - FA排除半径: {args.fa_exclude_radius} 像素")
    print("=" * 80)
    
    # 检查路径是否存在
    if not os.path.exists(args.pred_dir):
        print(f"\n❌ 错误: 预测目录不存在 - {args.pred_dir}")
        print(f"\n💡 提示:")
        print(f"  1. 确保已运行推理并保存了预测结果")
        print(f"  2. 或使用 --pred_dir 参数手动指定预测目录")
        print(f"\n示例:")
        print(f"  python cal.py --dataset {args.dataset} --pred_dir /path/to/predictions")
        sys.exit(1)
    
    if not os.path.exists(args.dataset_root):
        print(f"\n❌ 错误: 数据集目录不存在 - {args.dataset_root}")
        print(f"\n💡 提示:")
        print(f"  使用 --dataset_root 参数指定数据集根目录")
        print(f"\n示例:")
        print(f"  python cal.py --dataset {args.dataset} --dataset_root /path/to/dataset/{args.dataset}")
        sys.exit(1)
    
    results = calculate_metrics_from_saved_results(
        pred_dir=args.pred_dir,
        dataset_root=args.dataset_root,
        use_logits=args.use_logits,
        dataset_name=args.dataset,
        detection_threshold=args.detection_threshold,
        distance_threshold=args.distance_threshold,
        iou_threshold=args.iou_threshold,
        pd_search_radius=args.pd_search_radius,
        fa_exclude_radius=args.fa_exclude_radius
    )
    
    print("\n" + "=" * 80)
    print("✅ 指标计算完成！")