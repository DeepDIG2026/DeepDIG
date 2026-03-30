import math

import numpy as np

def put_heatmap(heatmap, center, sigma):
 
    center_x, center_y = center
    height, width = heatmap.shape
 
    th = 4.6052
    delta = math.sqrt(th * 2)
 
    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))
 
    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))
 
    exp_factor = 1 / 2.0 / sigma / sigma
 
    ## fast - vectorize
    arr_heatmap = heatmap[y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y)**2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    return heatmap

def bluring(img):
    h, w = img.shape[:2]
    
    h1 = np.random.randint(0, h - 1)
    h2 = np.random.randint(0, h - 1)
    while h2 == h1:
       h2 = np.random.randint(0, h - 1) 
    w1 = np.random.randint(0, w - 1)
    w2 = np.random.randint(0, w - 1)
    while w2 == w1:
        w2 = np.random.randint(0, w - 1)
    
    hmin = min(h1, h2)
    hmax = max(h1, h2)
    wmin = min(w1, w2)
    wmax = max(w1, w2)

    img[hmin:hmax, wmin:wmax] = 0

    return img

def mask_to_keypoints(mask, threshold=127):
    """
    Convert a binary mask to a list of keypoints.
    
    Args:
        mask: binary mask image (numpy array)
        threshold: binarization threshold, default 127
    
    Returns:
        keypoints: keypoints list [(x1, y1), (x2, y2), ...]
    """
    import cv2
    
    # Ensure binary image
    if mask.max() > 1:
        binary_mask = (mask > threshold).astype(np.uint8) * 255
    else:
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    for contour in contours:
        # Compute contour centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            keypoints.append([cx, cy])
    
    return keypoints

def keypoints_to_heatmap(keypoints, image_size, sigma=3):
    """
    Convert a keypoints list to a Gaussian heatmap.
    
    Args:
        keypoints: keypoints list [(x1, y1), (x2, y2), ...]
        image_size: image size (height, width)
        sigma: standard deviation of the Gaussian kernel
    
    Returns:
        heatmap: Gaussian heatmap (numpy array)
    """
    height, width = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    for kp in keypoints:
        x, y = kp
        heatmap = put_heatmap(heatmap, [x, y], sigma)
    
    return heatmap