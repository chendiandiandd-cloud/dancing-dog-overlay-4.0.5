"""Conservative runtime green-fringe cleanup for straight-alpha dog frames.

Original NPZ assets stay byte-identical. Only transition-alpha green excess and
nearly invisible alpha noise are changed; opaque character colors stay intact.
"""
import numpy as np
import cv2


def cleanup_edge_rgb(straight_rgb, alpha):
    """Return cleaned float32 RGB and alpha, preserving HW/HW1 alpha shape.

    Input RGB is straight, not premultiplied. Call once on prepared/cached
    frames before premultiplication, not after compositing over the photograph.
    """
    rgb = np.asarray(straight_rgb, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('straight_rgb must have shape (H, W, 3)')
    if a.shape not in (rgb.shape[:2], rgb.shape[:2] + (1,)):
        raise ValueError('alpha must have shape (H, W) or (H, W, 1)')
    was_2d = a.ndim == 2
    if was_2d:
        a = a[..., None]
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    a = np.clip(np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    clean_alpha = np.where(a < 0.008, np.float32(0), a)
    # Chroma-key compression can leave an opaque green line next to a cutout.
    # Include only a narrow three-pixel geometric band, never opaque interiors.
    near_clear = cv2.dilate((a[..., 0] < 0.10).astype(np.uint8),
                            np.ones((7, 7), np.uint8)) > 0
    fringe = (clean_alpha > 0) & ((clean_alpha < 0.98) | near_clear[..., None])
    rb_max = np.maximum(rgb[..., 0:1], rgb[..., 2:3])
    limit = rb_max + np.float32(0.02)
    clean_rgb = rgb.copy()
    clean_rgb[..., 1:2] = np.where(fringe & (rgb[..., 1:2] > limit), limit, rgb[..., 1:2])
    return clean_rgb, clean_alpha[..., 0] if was_2d else clean_alpha
