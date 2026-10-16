"""
DeepDIG Testing Script.

Test DeepDIG model for infrared small target detection.

Usage examples:
    python test_deepdig.py \
        --ckpt ./weights/YOUR_CHECKPOINT.pth \
        --root ./dataset \
        --dataset TSIRMT \
        --save_pred


    python test_deepdig.py \
            --ckpt ./weights/YOUR_CHECKPOINT.pth \
            --root ./dataset \
            --dataset IRDST \
            --save_pred




    python test_deepdig.py \
        --ckpt ./weights/YOUR_CHECKPOINT.pth \
        --root ./dataset \
        --dataset LMIRSTD \
        --save_pred
"""

import os
import sys
import time
import argparse
import hashlib
import json
import platform
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime

# Get device_num from command line and set environment variable
device_num = '0'
if len(sys.argv) > 1:
    for i, arg in enumerate(sys.argv):
        if arg == '--device_num' and i + 1 < len(sys.argv):
            device_num = sys.argv[i + 1]
            break
os.environ['CUDA_VISIBLE_DEVICES'] = device_num

import torch

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# Import local modules
from model.deep_dig import build_deep_dig
from utils.metric import compute_metric, get_keypoints
from utils.loss import AverageMeter
from datautils.dataloader import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description='DeepDIG testing script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--in_channels', type=int, default=20, help='Input sequence length')
    parser.add_argument('--detection_channels', type=int, default=64, help='Detection feature channels')
    parser.add_argument('--deep_supervision', action='store_true', default=True, help='Whether to use deep supervision')

    # Data configuration
    parser.add_argument('--dataset', type=str, default='IRDST',
                        choices=['TSIRMT', 'IRDST', 'LMIRSTD'], help='Dataset name')
    parser.add_argument('--root', type=str, default='dataset', help='Dataset root directory')
    parser.add_argument('--split_file', type=str, default='test.txt', help='Test set split file')
    parser.add_argument('--workers', type=int, default=8, help='Data loading threads')
    parser.add_argument('--test_batch_size', type=int, default=2, help='Test batch size; motion normalization depends on batch composition')

    # Preprocessing configuration
    parser.add_argument('--base_size', type=int, default=256, help='Input image base size')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size')
    parser.add_argument('--resize_mode', type=str, default='resize',
                        choices=['resize', 'padding'], help='Data preprocessing mode')
    parser.add_argument('--dthres', type=int, default=3, help='Detection matching threshold (pixels)')
    parser.add_argument('--descriptor_threshold', type=float, default=0.25,
                        help='DBA keypoint detection threshold')
    parser.add_argument('--seed', type=int, default=0, help='PyTorch, NumPy and OpenCV seed')
    parser.add_argument('--show_model', action='store_true', help='Print the loaded network and exit without loading data')
    parser.add_argument('--profile', action='store_true', help='Measure pre-aligned detector MACs with optional thop')
    parser.add_argument('--warmup_batches', type=int, default=3, help='Synthetic warm-up passes before timing')
    parser.add_argument('--max_samples', type=int, default=0, help='Limit test frames for a smoke check; 0 evaluates the full split')

    # Output configuration
    parser.add_argument('--save_pred', action='store_true', help='Save binarized prediction results')
    parser.add_argument('--save_heatmap', action='store_true', help='Save pseudo-color heatmap')
    parser.add_argument('--save_logits', action='store_true',
                        help='Save original grayscale logits map from sigmoid output (for ROC curve), '
                             'organized by {seq}/{frame}.png, output to '
                             'compared_methods/DeepDIG/DeepDIG_{dataset}/')
    parser.add_argument('--logits_dir', type=str, default='',
                        help='Logits map save root directory (empty for auto set at '
                             'compared_methods/DeepDIG/DeepDIG_{dataset})')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    parser.add_argument('--device_num', type=str, default='0', help='CUDA device number')

    return parser.parse_args()


# Original dataset sizes
DATASET_ORIGINAL_SIZES = {
    'IRDST': (150, 200),
    'TSIRMT': (150, 200),
    'LMIRSTD': (512, 640),
}


def crop_to_original_size(tensor, dataset_name, resize_mode='resize'):
    """Adjust prediction back to original size"""
    if dataset_name not in DATASET_ORIGINAL_SIZES:
        return tensor

    orig_h, orig_w = DATASET_ORIGINAL_SIZES[dataset_name]

    if tensor.dim() == 4:
        current_h, current_w = tensor.shape[2], tensor.shape[3]
    elif tensor.dim() == 3:
        current_h, current_w = tensor.shape[1], tensor.shape[2]
    else:
        return tensor

    if current_h == orig_h and current_w == orig_w:
        return tensor

    if resize_mode == 'padding':
        if tensor.dim() == 4:
            return tensor[:, :, :orig_h, :orig_w]
        else:
            return tensor[:, :orig_h, :orig_w]
    else:
        import torch.nn.functional as F
        original_dtype = tensor.dtype
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
            resized = F.interpolate(tensor, size=(orig_h, orig_w), mode='area')
            resized = resized.squeeze(1)
        else:
            resized = F.interpolate(tensor, size=(orig_h, orig_w), mode='area')

        if original_dtype == torch.uint8:
            resized = resized.clamp(0, 255).byte()

        return resized


def test():
    args = parse_args()
    if args.in_channels < 2 or args.test_batch_size < 1 or args.warmup_batches < 0 or args.max_samples < 0:
        raise ValueError('Invalid sequence length, batch size, warm-up count or sample limit')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setRNGSeed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set output directory
    if not args.output_dir:
        timestamp = datetime.now().strftime('%m%d-%H%M')
        args.output_dir = f'result/{args.dataset}/test_{timestamp}'

    if args.save_pred or args.save_heatmap:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)
        if args.save_heatmap:
            os.makedirs(os.path.join(args.output_dir, 'heatmaps'), exist_ok=True)

    # Set logits save directory
    if args.save_logits:
        if not args.logits_dir:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            args.logits_dir = os.path.join(
                script_dir, 'compared_methods', 'DeepDIG',
                f'DeepDIG_{args.dataset}'
            )
        os.makedirs(args.logits_dir, exist_ok=True)
        print(f"  Logits save directory: {args.logits_dir}")

    print("=" * 80)
    print("DeepDIG Testing")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Build model
    from model.deep_dig import build_deep_dig

    model = build_deep_dig(
        input_channels=args.in_channels,
        deep_supervision=args.deep_supervision,
        detection_channels=args.detection_channels,
        with_cache=True,
        window_size=args.in_channels,
        descriptor_threshold=args.descriptor_threshold,
    ).to(device)

    # Load weights
    print(f"\nLoad model: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get('pipeline_ablation', 'full') != 'full':
        raise ValueError('This entry point requires a complete DBA + RADC + MAG checkpoint')
    state_dict = ckpt.get('state_dict', ckpt)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    print("  Model loaded successfully")

    # Model parameters count
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    if args.show_model:
        print(model)
        return

    # Calculate GFLOPs
    gflops = None
    gflops_str = 'N/A'
    if args.profile and not THOP_AVAILABLE:
        raise RuntimeError('--profile requires thop; install it or omit this optional flag')
    if args.profile:
        try:
            dummy = torch.randn(
                1, args.in_channels, args.base_size, args.base_size,
                device=device
            )
            with torch.no_grad():
                macs, _ = profile(model, inputs=(dummy,), verbose=False)
            gflops = 2 * macs / 1e9
            macs_fmt, _ = clever_format([macs, total_params], "%.3f")
            gflops_str = f"{gflops:.3f}"
            print(f"\nPre-aligned detector cost (DBA excluded; 2 FLOPs per MAC; input [1,{args.in_channels},{args.base_size},{args.base_size}]):")
            print(f"  MACs:   {macs_fmt}")
            print(f"  GFLOPs: {gflops_str}")
        except Exception as e:
            print(f"  Warning: GFLOPs calculation failed: {e}")

    # Build data loader
    args.mode = 'test'
    args.load_single_frame = False
    args.train_batch_size = args.test_batch_size
    args.channel_size = 'four'
    args.backbone = 'resnet_10'
    args.fast_val = False

    test_loader = build_dataloader('test', args)
    if not len(test_loader.dataset):
        raise ValueError('The test split contains no valid frames')
    if args.max_samples:
        from torch.utils.data import DataLoader, Subset
        test_loader = DataLoader(
            Subset(test_loader.dataset, range(min(args.max_samples, len(test_loader.dataset)))),
            batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
        )
    print(f"  Test samples: {len(test_loader.dataset)}")

    # GPU warm-up
    WARMUP_BATCHES = args.warmup_batches
    print(f"\nWarming up GPU ({WARMUP_BATCHES} batches)...")
    with torch.no_grad():
        dummy_w = torch.randn(
            args.test_batch_size, args.in_channels,
            args.base_size, args.base_size, device=device
        )
        for _ in range(WARMUP_BATCHES):
            _ = model(dummy_w, extract_xfeat=False, align_mode='descriptor')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print("  Warm-up completed")

    # Test
    test_metrics = ['Precision', 'Recall', 'PD', 'FA', 'mIoU']
    metric_values = {m: AverageMeter() for m in test_metrics}
    match_counts  = AverageMeter()
    infer_time_meter = AverageMeter()   # Pure inference time (sec/sample)
    total_intersection = 0
    total_union = 0
    frame_ious = []
    empty_union_frames = 0
    total_targets = 0
    detected_targets = 0
    false_pixels = 0
    total_pixels = 0
    input_shape = None

    tbar = tqdm(test_loader, ncols=120, desc='Testing')

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tbar):
            data = data_batch[0].to(device)
            labels_full = data_batch[1].to(device)
            paths = data_batch[-1]

            if labels_full.dim() == 3:
                labels = labels_full.unsqueeze(1)
            elif labels_full.shape[1] == 2:
                labels = labels_full[:, :1]
            else:
                labels = labels_full

            # Normalize
            data = data / 255.0
            input_shape = list(data.shape[1:])
            if labels.max() > 1.0:
                labels = labels / 255.0

            # Measure inference time
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            result = model(data, extract_xfeat=False, align_mode='descriptor')

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_size = data.shape[0]
            infer_time_meter.update((t1 - t0) / batch_size, batch_size)

            if isinstance(result, tuple):
                preds, align_stats = result
                match_counts.update(align_stats['avg_matches'], data.shape[0])
            else:
                preds = result

            if args.deep_supervision:
                pred = preds[-1]
            else:
                pred = preds

            # Adjust to original size
            pred_for_metric = crop_to_original_size(pred, args.dataset, args.resize_mode)

            B = pred_for_metric.shape[0]
            for b in range(B):
                pred_np = pred_for_metric[b].squeeze(0).cpu().numpy()
                preditem = (pred_np > 0.5).astype(np.float32)

                # Read original GT
                sample_path = paths[-1][b]
                gt_path = sample_path.replace('images', 'masks')
                full_gt_path = os.path.join(args.root, args.dataset, gt_path)

                if not os.path.exists(full_gt_path):
                    base_path = os.path.splitext(gt_path)[0]
                    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                        candidate = os.path.join(args.root, args.dataset, base_path + ext)
                        if os.path.exists(candidate):
                            full_gt_path = candidate
                            break

                original_gt = cv2.imread(full_gt_path, cv2.IMREAD_GRAYSCALE)
                if original_gt is not None:
                    gt_mask = (original_gt.astype(np.float32) / 255.0 >= 0.5).astype(np.float32)
                else:
                    raise FileNotFoundError(f'Missing or unreadable ground truth: {full_gt_path}')
                if preditem.shape != gt_mask.shape:
                    raise ValueError(f'Prediction and GT shapes differ: {sample_path}')
                intersection = int(np.logical_and(preditem > 0.5, gt_mask > 0.5).sum())
                union = int(np.logical_or(preditem > 0.5, gt_mask > 0.5).sum())
                total_intersection += intersection
                total_union += union
                frame_ious.append(intersection / union if union else 0.0)
                empty_union_frames += int(union == 0)

                predkey = get_keypoints(preditem)
                gtkey = get_keypoints(gt_mask)

                frame_padded = data[b, -1].cpu().numpy()
                frame_tensor = torch.from_numpy(frame_padded).unsqueeze(0).unsqueeze(0)
                frame_for_metric = crop_to_original_size(frame_tensor, args.dataset, args.resize_mode)[0, 0].numpy()

                frame_metrics = {}
                for metric in test_metrics:
                    value = compute_metric(
                        frame_for_metric, preditem, predkey, gtkey,
                        metric, args.dthres, gt_mask
                    )
                    if value is not None:
                        metric_values[metric].update(value)
                        frame_metrics[metric] = value
                num_targets = cv2.connectedComponents((gt_mask > 0.5).astype(np.uint8), connectivity=8)[0] - 1
                total_targets += num_targets
                detected_targets += int(round(frame_metrics['PD'] * num_targets))
                false_pixels += int(round(frame_metrics['FA'] * gt_mask.size))
                total_pixels += int(gt_mask.size)

                # Save binarized prediction results
                if args.save_pred:
                    pred_save = (preditem * 255).astype(np.uint8)
                    # Keep a one-to-one naming/path mapping with dataset masks.
                    dataset_root = os.path.join(args.root, args.dataset)
                    masks_root = os.path.join(dataset_root, 'masks')

                    normalized_gt = os.path.normpath(full_gt_path)
                    normalized_masks_root = os.path.normpath(masks_root)

                    if normalized_gt.startswith(normalized_masks_root + os.sep):
                        rel_mask_path = os.path.relpath(normalized_gt, normalized_masks_root)
                        pred_save_path = os.path.join(args.output_dir, 'predictions', rel_mask_path)
                    else:
                        parts = sample_path.replace('\\', '/').split('/')
                        seq_id = parts[-2] if len(parts) >= 2 else ''
                        frame_name = os.path.splitext(parts[-1])[0] + '.png'
                        pred_save_path = os.path.join(args.output_dir, 'predictions', seq_id, frame_name)

                    os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                    cv2.imwrite(pred_save_path, pred_save)

                # Save pseudo-color heatmap
                if args.save_heatmap:
                    heatmap = (pred_np * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    save_name = sample_path.replace('/', '_').replace('\\', '_')
                    save_name = os.path.splitext(save_name)[0] + '_heatmap.png'
                    cv2.imwrite(os.path.join(args.output_dir, 'heatmaps', save_name), heatmap_color)

                # Save original grayscale logits map (for ROC curve)
                if args.save_logits:
                    # Parse sequence ID and frame name from sample_path
                    # sample_path Format: images/{seq_id}/{frame}.ext
                    parts = sample_path.replace('\\', '/').split('/')
                    seq_id   = parts[-2]                          # e.g. '13' / '00002'
                    frame_fn = os.path.splitext(parts[-1])[0] + '.png'  # e.g. '0001.png'

                    seq_dir = os.path.join(args.logits_dir, seq_id)
                    os.makedirs(seq_dir, exist_ok=True)

                    # Linearly map sigmoid output [0,1] to uint8 [0,255] and save as grayscale
                    logits_gray = np.clip(pred_np * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(seq_dir, frame_fn), logits_gray)

    # Compute summary metrics
    precision = metric_values['Precision'].avg
    recall    = metric_values['Recall'].avg
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fps = 1.0 / infer_time_meter.avg if infer_time_meter.avg > 0 else 0.0
    iou = total_intersection / total_union if total_union else 0.0
    # Match the historical metric.py / test protocol for the primary scores.
    # compute_miou already assigns 1 when both prediction and GT are empty.
    niou = metric_values['mIoU'].avg
    pd_frame_mean = metric_values['PD'].avg
    fa_frame_mean = metric_values['FA'].avg
    # Retain the previous aggregation policies under explicit diagnostic names.
    niou_empty_zero = float(np.mean(frame_ious)) if frame_ious else 0.0
    pd_accumulated = detected_targets / total_targets if total_targets else 0.0
    fa_accumulated = false_pixels / total_pixels if total_pixels else 0.0

    # Print results
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)
    print("  Metric policy: legacy_frame_mean_v1 (empty-union nIoU = 1)")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1:         {f1:.4f}")
    print(f"  PD:         {pd_frame_mean:.4f}")
    print(f"  FA:         {fa_frame_mean:.8f}")
    print(f"  FA_x1e6:    {fa_frame_mean * 1e6:.4f}")
    print(f"  IoU:        {iou:.4f}")
    print(f"  nIoU:       {niou:.4f}")
    print(f"  mIoU:       {metric_values['mIoU'].avg:.4f}")
    print(f"  Avg Matches:{match_counts.avg:.1f}")
    print(f"\nPerformance stats (pure model inference only, excluding dataloading/post-processing):")
    print(f"  FPS:        {fps:.2f}  (= 1 / {infer_time_meter.avg*1000:.2f} ms)")
    print(f"  Total params:   {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  GFLOPs:     {gflops_str}")
    print("=" * 80)
    if args.max_samples:
        print('PARTIAL TEST: these metrics cover only the requested frame limit.')

    # Save results to file
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, 'results.txt')
        with open(result_path, 'w') as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Checkpoint: {args.ckpt}\n")
            f.write("MetricPolicy: legacy_frame_mean_v1\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1: {f1:.4f}\n")
            f.write(f"PD: {pd_frame_mean:.8f}\n")
            f.write(f"FA: {fa_frame_mean:.12f}\n")
            f.write(f"FA_x1e6: {fa_frame_mean * 1e6:.8f}\n")
            f.write(f"IoU: {iou:.8f}\n")
            f.write(f"nIoU: {niou:.8f}\n")
            f.write(f"PD_frame_mean: {metric_values['PD'].avg:.8f}\n")
            f.write(f"FA_frame_mean: {metric_values['FA'].avg:.12f}\n")
            f.write(f"PD_target_weighted: {pd_accumulated:.8f}\n")
            f.write(f"FA_pixel_weighted: {fa_accumulated:.12f}\n")
            f.write(f"nIoU_empty_zero: {niou_empty_zero:.8f}\n")
            f.write("nIoU_empty_union_policy: 1\n")
            f.write(f"Frames: {len(frame_ious)}\n")
            f.write(f"EmptyUnionFrames: {empty_union_frames}\n")
            f.write(f"PartialTest: {bool(args.max_samples)}\n")
            f.write(f"mIoU: {metric_values['mIoU'].avg:.4f}\n")
            f.write(f"\n# Performance stats\n")
            f.write(f"FPS: {fps:.2f}\n")
            f.write(f"Infer_ms_per_sample: {infer_time_meter.avg * 1000:.3f}\n")
            f.write(f"Total_Params: {total_params}\n")
            f.write(f"Trainable_Params: {trainable_params}\n")
            f.write(f"GFLOPs: {gflops_str}\n")
        print(f"\nResults saved to: {result_path}")
        source_files = [__file__] + [os.path.join(current_dir, 'model', name)
                                    for name in sorted(os.listdir(os.path.join(current_dir, 'model')))
                                    if name.endswith('.py')]
        source_files += [os.path.join(current_dir, 'datautils', 'dataloader.py'),
                         os.path.join(current_dir, 'utils', 'metric.py')]
        def sha256(path):
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            return digest.hexdigest()
        split_name = os.path.splitext(args.split_file)[0]
        split_candidates = [os.path.join(args.root, args.dataset, 'ImageSets', split_name + '.txt'),
                            os.path.join(args.root, args.dataset, split_name + '.txt')]
        split_path = next(path for path in split_candidates if os.path.isfile(path))
        report = {
            'arguments': vars(args), 'checkpoint_sha256': sha256(args.ckpt),
            'split_file': split_path, 'split_sha256': sha256(split_path),
            'frames': len(frame_ious), 'partial_test': bool(args.max_samples),
            'input_shape_per_sample': input_shape,
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'numpy': np.__version__, 'opencv': cv2.__version__,
                            'device': str(device)},
            'source_sha256': {os.path.relpath(path, current_dir): sha256(path) for path in source_files},
            'metrics': {'Precision': precision, 'Recall': recall, 'F1': f1,
                        'PD': pd_frame_mean, 'FA': fa_frame_mean, 'FA_x1e6': fa_frame_mean * 1e6,
                        'IoU': iou, 'nIoU': niou, 'mIoU_legacy': metric_values['mIoU'].avg,
                        'PD_frame_mean': pd_frame_mean, 'FA_frame_mean': fa_frame_mean,
                        'PD_target_weighted': pd_accumulated, 'FA_pixel_weighted': fa_accumulated,
                        'nIoU_empty_zero': niou_empty_zero},
            'metric_policy': {'name': 'legacy_frame_mean_v1',
                              'prediction_threshold': 0.5, 'evaluation_resolution': 'original',
                              'IoU': 'sum(intersection)/sum(union)',
                              'nIoU': 'frame mean; empty union=1 (same as historical mIoU)',
                              'mIoU_legacy': 'frame mean; empty union=1',
                              'F1': 'harmonic mean of frame-averaged precision and recall',
                              'PD_FA': 'frame mean; historical metric.py definitions',
                              'PD_target_weighted': 'sum(detected GT components)/sum(GT components)',
                              'FA_pixel_weighted': 'sum(false pixels)/sum(image pixels)',
                              'nIoU_empty_zero': 'diagnostic frame mean; empty union=0'},
            'performance': {'FPS': fps, 'infer_ms_per_sample': infer_time_meter.avg * 1000,
                            'total_params': total_params, 'detector_GFLOPs': gflops,
                            'timing_scope': 'DBA alignment and detection; excludes loading and postprocessing',
                            'profile_scope': 'pre-aligned detector only; 2 FLOPs per MAC'},
        }
        with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)


if __name__ == '__main__':
    test()
