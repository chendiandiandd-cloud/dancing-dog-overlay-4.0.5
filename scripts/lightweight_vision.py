"""Offline face, text, person and surface cues; no external model download."""
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
import math


@lru_cache(maxsize=1)
def cascades():
    root = Path(getattr(getattr(cv2, 'data', None), 'haarcascades', ''))
    models = []
    for name in ('haarcascade_frontalface_default.xml', 'haarcascade_eye.xml'):
        path = root / name
        model = cv2.CascadeClassifier(str(path)) if path.is_file() else None
        models.append(model if model is not None and not model.empty() else None)
    return models


def face_boxes(rgb):
    model, eye = cascades()
    if model is None:
        return [], False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    result = []
    for x, y, w, h in model.detectMultiScale(gray, 1.1, 5, minSize=(20, 20)):
        area = w * h / (W * H)
        if area > .30 or w > W * .65 or h > H * .60:
            continue
        if area > .075:
            eyes = eye.detectMultiScale(gray[y:y + int(h * .65), x:x + w], 1.1, 3,
                                       minSize=(max(5, w // 10), max(5, h // 10))) if eye is not None else []
            if len(eyes) < 1:
                continue
        result.append([int(x), int(y), int(x + w), int(y + h)])
    return result, True


def text_regions(rgb):
    """Small high-contrast strokes grouped into horizontal text-like runs."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    response = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    binary = (response > 55).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    keep = np.zeros(n, np.uint8)
    for label, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        letter = 2 <= bh <= max(8, h * .045) and 1 <= bw <= max(12, w * .055)
        joined = 3 <= bh <= max(8, h * .04) and bw / bh > 5 and w * .08 < bw < w * .80
        if (letter or joined) and area >= 3:
            keep[label] = 255
    strokes = keep[labels]
    grouped = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 50), 3)))
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        density = float(np.mean(strokes[y:y + bh, x:x + bw] > 0))
        # Text usually doesn't occupy the very bottom edge where dogs walk
        if y + bh > h * 0.95: continue
        if bw >= max(14, w * .045) and 3 <= bh <= h * .07 and bw / bh > 2.2 and .10 < density < .7:
            result.append([max(0, x - 3), max(0, y - 3), min(w, x + bw + 3), min(h, y + bh + 3)])
    return result


def head_regions(rgb):
    """Bounded head-like cues for hats/shades missed by Haar."""
    H, W = rgb.shape[:2]
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    r, g, b = [rgb[..., k].astype(np.int16) for k in range(3)]
    skin = ((ycc[..., 0] > 45) & (ycc[..., 1] > 130) & (ycc[..., 1] < 185) &
            (ycc[..., 2] > 75) & (ycc[..., 2] < 140) & (r - g > 5) & (r - b > 5) & (hsv[..., 1] < 170)).astype(np.uint8)
    smooth = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    integral = {k: cv2.integral(v) for k, v in
                [('skin', skin), ('dark', (gray < 100).astype(np.uint8)),
                 ('neutral', ((hsv[..., 1] < 65) & (gray > 80)).astype(np.uint8))]}

    def mean(k, rect):
        x, y, x2, y2 = [int(v) for v in rect]
        x, y, x2, y2 = max(0, x), max(0, y), min(W, x2), min(H, y2)
        if x2 <= x or y2 <= y:
            return 0.
        a = integral[k]
        return float((a[y2, x2] - a[y, x2] - a[y2, x] + a[y, x]) / ((x2 - x) * (y2 - y)))

    proposals = []
    _, _, stats, _ = cv2.connectedComponentsWithStats(smooth)
    for x, y, w, h, area in stats[1:]:
        if not (.02 * H <= h <= .16 * H and .025 * W <= w <= .23 * W and .6 < w / h < 2.1 and .3 < area / (w * h) < .96):
            continue
        if not (.15 < (x + w / 2) / W < .85 and .08 < y / H < .65):
            continue
        cloth = mean('neutral', [x, y + h, x + w, y + 2 * h])
        brim = mean('dark', [x, y + h * .25, x + w, y + h * .65])
        if cloth > .30 or brim > .15:
            proposals.append((float(.6 + cloth * .2 + brim * .2),
                              [int(x - w * .15), int(y - h * .1), int(x + w * 1.15), int(y + h * 1.25)],
                              'compact_hat_head'))
    for h in (H * .08, H * .12, H * .18, H * .25):
        w = h * .8
        for cx in np.linspace(.22 * W, .78 * W, 11):
            for cy in np.linspace(.15 * H, .55 * H, 11):
                x, y = cx - w / 2, cy - h / 2
                if y < H * .05 or y > H * .32:
                    continue
                cheek = mean('skin', [x + w * .12, y + h * .45, x + w * .88, y + h * .95])
                eye = mean('dark', [x, y + h * .05, x + w, y + h * .45])
                side = max(mean('skin', [x - w * .4, y + h * .45, x, y + h * .95]),
                           mean('skin', [x + w, y + h * .45, x + w * 1.4, y + h * .95]))
                bottom = mean('skin', [x, y + h, x + w, y + h * 1.45])
                if cheek < .50 or eye < .16 or cheek - side < .10:
                    continue
                # Large masonry/reflection patches can have an eye-like dark
                # stripe but carry the same skin-like color far below the face.
                if h >= H * .16 and cheek - bottom < .15:
                    continue
                score = cheek * .6 + eye * .2 + (cheek - side) * .3 + max(0, cheek - bottom) * .2
                proposals.append((score, [round(x), round(y), round(x + w), round(y + h)], 'dark_band_cheek_head'))
    proposals.sort(reverse=True)
    picked = []
    for score, box, kind in proposals:
        if score < .6:
            continue
        x, y, x2, y2 = box
        duplicate = False
        for found in picked:
            a, b, c, d = found['box']
            intersection = max(0, min(x2, c) - max(x, a)) * max(0, min(y2, d) - max(y, b))
            if intersection / max(1, min((x2 - x) * (y2 - y), (c - a) * (d - b))) > .3:
                duplicate = True
                break
        if not duplicate:
            picked.append(dict(confidence=min(1., float(score)),
                               box=[max(0, x), max(0, y), min(W, x2), min(H, y2)], cue=kind))
        if len(picked) == 4:
            break
    return picked


def subject_proxy_zones(heads, H, W):
    """Extend head cues downward to estimate person bounds."""
    proxies = []
    for h in heads:
        x1, y1, x2, y2 = h['box']
        hw = x2 - x1
        hh = y2 - y1
        px1 = max(0, x1 - 0.2 * hw)
        px2 = min(W, x2 + 0.2 * hw)
        py1 = y1
        py2 = min(H, y2 + 6.2 * hh)
        proxies.append([px1, py1, px2, py2])
    return proxies


@lru_cache(maxsize=1)
def people_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def person_boxes(rgb):
    """Built-in upright-person HOG, bounded CPU; no external model download."""
    return [list(box) for box in _person_boxes(rgb.tobytes(),*rgb.shape[:2])]


@lru_cache(maxsize=8)
def _person_boxes(pixels, H, W):
    rgb=np.frombuffer(pixels,np.uint8).reshape(H,W,3)
    factor = min(1.,640/max(H,W))
    frame = cv2.resize(rgb,(round(W*factor),round(H*factor)))
    if frame.shape[0]<128 or frame.shape[1]<64:
        return []
    boxes,weights = people_detector().detectMultiScale(frame,winStride=(8,8),
                                                       padding=(8,8),scale=1.08)
    result = []
    for (x,y,w,h),score in sorted(zip(boxes,weights),key=lambda v:float(v[1]),reverse=True):
        if float(score)<.7:
            continue
        box=np.array([x,y,x+w,y+h],float)/factor
        duplicate=False
        for a,b,c,d in result:
            overlap=max(0,min(c,box[2])-max(a,box[0]))*max(0,min(d,box[3])-max(b,box[1]))
            if overlap/min((c-a)*(d-b),w*h/factor**2)>.5:
                duplicate=True
                break
        if not duplicate:
            result.append(box.tolist())
    return tuple(tuple(box) for box in result[:12])


def subject_saliency_mask(rgb):
    """Lightweight subject saliency using contrast, color and central bias."""
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    grad = cv2.GaussianBlur(grad, (7, 7), 0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_mean, a_mean, b_mean = np.mean(lab, axis=(0, 1))
    dist = np.sqrt((lab[..., 0] - l_mean)**2 + (lab[..., 1] - a_mean)**2 + (lab[..., 2] - b_mean)**2)
    dist = cv2.GaussianBlur(dist, (15, 15), 0)
    yy, xx = np.mgrid[:H, :W]
    bias = np.exp(-(((xx - W * 0.5) / (W * 0.4))**2 + ((yy - H * 0.55) / (H * 0.45))**2))
    combined = grad * 0.3 + dist * 0.5 + bias * 50
    thresh = np.percentile(combined, 88)
    mask = (combined > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    return mask


def _surface_evidence(rgb):
    """Bottom contact is necessary, not sufficient: inspect local 2-D texture.

    Color alone cannot separate shaded stone from water, or green reflections
    from grass. Reject distributed horizontal ripples; sharp neutral stair
    risers remain eligible. All measurements stay on the bounded input frame.
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    color = rgb.astype(np.float32)
    lab = cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    near = np.zeros((h, w), np.uint8)
    near[round(h * .72):] = 1
    blue = (hsv[..., 0] > 70) & (hsv[..., 0] < 140)
    warm = (hsv[..., 0] < 30) & (hsv[..., 1] < 170)
    mineral = ((hsv[..., 1] < 80) | (blue & (hsv[..., 1] < 210)) | warm) & (hsv[..., 2] > 30)
    grass = ((hsv[..., 0] >= 30) & (hsv[..., 0] <= 88) &
             (hsv[..., 1] > 28) & (hsv[..., 2] > 35) &
             (color[..., 1] > color[..., 0] * 1.12) &
             (color[..., 1] > color[..., 2] * 1.12))
    components, palettes = [], {}
    for kind, mask in (('paving', mineral), ('grass', grass)):
        surface = cv2.morphologyEx(mask.astype(np.uint8) * near, cv2.MORPH_CLOSE,
                                  np.ones((5, 5), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(surface)
        for label in np.unique(labels[max(0, h - 8):]):
            if label and stats[label, cv2.CC_STAT_AREA] >= w * h * .012:
                component = labels == label
                ex, ey = float(gx[component].mean()), float(gy[component].mean())
                cool = float(blue[component].mean())
                if (kind == 'grass' or cool > .25) and ey / max(1., ex) > 1.8:
                    continue
                components.append((kind, component))
                bottom_colors=lab[round(h*.95):][component[round(h*.95):]]
                quantized=(bottom_colors[:,1:]/24).astype(np.int16)
                bins,counts=np.unique(quantized,axis=0,return_counts=True)
                chosen=bins[np.argsort(-counts)[:4]]
                representatives=[]
                for key in chosen:
                    colors=bottom_colors[np.all(quantized==key,axis=1)]
                    for percentile in (25,75):
                        light=np.percentile(colors[:,0],percentile)
                        representatives.append(colors[np.argmin(abs(colors[:,0]-light))])
                palettes[id(component)]=np.array(representatives)

    def supported(component, kind, px, py):
        dx, dy = max(3, round(w * .035)), max(2, round(h * .012))
        xa, xb = max(0, px - dx), min(w, px + dx)
        ya, yb = max(0, py - dy), min(h, py + 2 * dy)
        strip = component[ya:yb, xa:xb]
        if not strip.size or float(strip.mean()) < .90:
            return False
        # A near-ground foot strip must have a broad downward path, not merely
        # share a coarse color class with clothing, a wall or a railing.
        corridor=component[py:,xa:xb]
        if not corridor.size or float(np.mean(corridor.mean(axis=1)>.70))<.85:
            return False
        contact=lab[max(0,py-dy):min(h,py+dy),xa:xb]
        palette=palettes[id(component)]
        if contact.size and palette.size:
            distance=np.linalg.norm(palette-np.median(contact.reshape(-1,3),axis=0),axis=1).min()
            if distance>38:
                return False
            above=lab[max(0,py-dy):py,xa:xb]
            if above.size and np.linalg.norm(palette-np.median(above.reshape(-1,3),axis=0),axis=1).min()>45:
                return False
        xa, xb = max(0, px - round(w * .08)), min(w, px + round(w * .08))
        ya, yb = max(round(h * .72), py - round(h * .08)), min(h, py + round(h * .03))
        x, y = gx[ya:yb, xa:xb], gy[ya:yb, xa:xb]
        if not x.size:
            return False
        ex, ey = float(x.mean()), float(y.mean())
        if min(ex, ey) < 2.0:
            return False
        direction = ey / max(1., ex)
        row = y.mean(axis=1)
        peaks = float(np.percentile(row, 90) / max(1., np.median(row)))
        patch = hsv[ya:yb, xa:xb]
        cool = float(((patch[..., 0] > 70) & (patch[..., 0] < 140) &
                      (patch[..., 1] > 50)).mean())
        stairs = kind == 'paving' and cool < .25 and float(patch[..., 1].mean()) < 80 and peaks > 2.0
        # Low vertical energy is also a warning for railing/clothing stripes.
        return .65 <= direction <= 1.8 or (direction > 1.8 and stairs)

    return components, supported


def detect_open_grass(rgb):
    """Broad textured lawn patches can end at foreground, above the bottom edge."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    color = rgb.astype(np.float32)
    mask = ((hsv[..., 0] >= 30) & (hsv[..., 0] <= 70) &
            (hsv[..., 1] > 28) & (hsv[..., 2] > 35) &
            (color[..., 1] >= color[..., 0]) &
            (color[..., 1] > color[..., 2] * 1.12)).astype(np.uint8)
    mask[:round(h*.40)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    candidates = []
    for label, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        if area < w*h*.04 or bw < w*.20 or bh < h*.15:
            continue
        component = labels == label
        for fy in (.48, .56, .64, .72, .80, .88):
            py = round(h*fy)
            for fx in np.linspace(.09, .91, 13):
                px = round(w*float(fx))
                dx = max(3, round(w*.055))
                xa, xb = max(0, px-dx), min(w, px+dx)
                ya, yb = max(0, py-round(h*.035)), min(h, py+round(h*.12))
                patch = component[ya:yb, xa:xb]
                if not patch.size or patch.mean() < .92:
                    continue
                ex, ey = float(gx[ya:yb, xa:xb].mean()), float(gy[ya:yb, xa:xb].mean())
                if min(ex, ey) < 2 or not .65 <= ey/max(1., ex) <= 1.8:
                    continue
                # Keep tracking inside this local lawn, away from bottom bags/cloth.
                candidates.append(dict(anchor=(px, py), score=.90,
                                       surface_kind='open_grass',
                                       support_region=[xa/w, ya/h, xb/w, yb/h]))
    return candidates


def detect_ground_regions(rgb):
    """Bottom-connected near-ground evidence, not arbitrary empty-space anchors.

    Grass needs a separate color/texture route: it is not low-saturation paving.
    Candidate foot strips must belong to a supported component, not just have
    their center pixel in it. Directional water texture remains rejected.
    """
    sh, sw = rgb.shape[:2]
    components, supported = _surface_evidence(rgb)
    candidates = []
    for kind, component in components:
        for fy in (.84, .90, .955, .98):
            py = min(sh - 1, round(fy * sh))
            for fx in np.linspace(.09, .91, 13):
                px = min(sw - 1, round(float(fx) * sw))
                if supported(component, kind, px, py):
                    candidates.append(dict(anchor=(px, py), score=.82,
                                           surface_kind=kind))
    return candidates + detect_open_grass(rgb)


def detect_horizontal_supports(rgb):
    """Line-based supports with stricter water/sky rejection."""
    sh, sw = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 130)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                            minLineLength=max(8, int(sw * 0.06)), maxLineGap=20)
    line_rows = [] if lines is None else np.asarray(lines).reshape(-1, 4)[:150].tolist()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)) / 1024
    components, supported = _surface_evidence(rgb)
    supports = []
    for line in line_rows:
        x1, y1, x2, y2 = map(float, line)
        if min(y1, y2) / sh < 0.72: continue
        if abs(y2 - y1) > max(1, abs(x2 - x1)) * 0.15: continue
        ly = max(0, min(sh - 1, round((y1 + y2) / 2)))
        xa, xb = max(0, int(min(x1, x2))), min(sw, int(max(x1, x2)) + 1)
        below_rect = (min(sh, ly + 2), min(sh, ly + max(5, round(sh * 0.05))))
        def strip_mean(arr, bounds):
            region = arr[bounds[0]:bounds[1], xa:xb]
            return float(np.mean(region)) if region.size else 0.
        tex_below = strip_mean(grad, below_rect)
        patch_hsv = hsv[below_rect[0]:below_rect[1], xa:xb]

        # Water/Sky rejection: Green-Blue hue (70-140) and moderate/high saturation
        # Or very smooth/bright (Sky/Fog)
        hue = float(np.mean(patch_hsv[..., 0])) if patch_hsv.size else 0.
        sat = float(np.mean(patch_hsv[..., 1])) if patch_hsv.size else 0.
        val = float(np.mean(patch_hsv[..., 2])) if patch_hsv.size else 0.

        if val > 220 and sat < 30:
            continue
        if tex_below > 0.28 or tex_below < 0.005: continue

        # A Hough line may cross water, clothes or a wall: only expose verified
        # foot points, not its unchecked full extent to placement's interpolation.
        for k in (.2, .5, .8):
            px, py = round(x1 + k * (x2 - x1)), round(y1 + k * (y2 - y1))
            ground = any(supported(c, kind, px, py) for kind, c in components)
            # A textured vertical wall, railing or clothing can satisfy the
            # ledge gradient test. Require bottom-connected surface evidence;
            # elevated carriers need an explicit scene-plan region.
            if ground and py >= sh*.84:
                supports.append(dict(line=[float(px), float(py), float(px), float(py)],
                                     score=.86))
    return sorted(supports, key=lambda s: s['score'], reverse=True)
