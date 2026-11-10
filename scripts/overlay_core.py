"""Deterministic offline overlay primitives; bounded analysis and one shader."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import subprocess
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from shot_detection import ShotDetector
from lightweight_vision import face_boxes, text_regions, head_regions, subject_proxy_zones, detect_horizontal_supports, subject_saliency_mask, detect_ground_regions, person_boxes

ENGINE_VERSION = 'overlay-4.0.5'
HEIGHT_RATIO_MIN = .24
HEIGHT_RATIO_MAX = .35


def validate_height_ratio(ratio):
    if ratio is not None and (not math.isfinite(ratio) or not .04 <= ratio <= .60):
        raise ValueError('--height-ratio必须为[.04,.60]内的有限数')
    return ratio
RNG_SALT = 'overlay-1.3.5'  # Stable input salt; dog stream is isolated from placement.
cv2.setNumThreads(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp', delete=False) as f:
        temp = Path(f.name)
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def run(args, timeout=120):
    result = subprocess.run([str(x) for x in args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exit {result.returncode}: ' +
                           result.stderr.decode('utf-8', 'replace')[-6000:])
    return result.stdout


class PipeProcess:
    """Deadline kills child even when pipe read/write blocks."""
    def __init__(self, args, timeout=600, reader=False, writer=False):
        self.err = tempfile.TemporaryFile()
        self.expired = False
        try:
            self.p = subprocess.Popen([str(x) for x in args],
                                      stdin=subprocess.PIPE if writer else subprocess.DEVNULL,
                                      stdout=subprocess.PIPE if reader else subprocess.DEVNULL,
                                      stderr=self.err, bufsize=0)
        except Exception:
            self.err.close()
            raise
        self.timer = threading.Timer(timeout, self._expire)
        self.timer.daemon = True
        self.timer.start()

    def _expire(self):
        self.expired = True
        if self.p.poll() is None:
            self.p.kill()

    def finish(self):
        if self.p.stdin and not self.p.stdin.closed:
            self.p.stdin.close()
        self.p.wait(timeout=15)
        self.timer.cancel()
        self.err.seek(0)
        error = self.err.read().decode('utf-8', 'replace')[-6000:]
        if self.expired or self.p.returncode:
            raise RuntimeError(f'FFmpeg {"timeout" if self.expired else "exit " + str(self.p.returncode)}: {error}')

    def close(self):
        self.timer.cancel()
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait(timeout=15)
        for stream in (self.p.stdin, self.p.stdout):
            if stream and not stream.closed:
                stream.close()
        self.err.close()


def probe(path):
    return json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                           '-of', 'json', path], timeout=60))


def rate(value, fallback=30.):
    try:
        a, *b = str(value).split('/')
        r = float(a) / (float(b[0]) if b else 1)
        return r if math.isfinite(r) and r > 0 else fallback
    except (ValueError, ZeroDivisionError):
        return fallback


def image_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return np.asarray(bg.convert('RGB')).copy()


def media_info(path, image_duration, quality):
    try:
        with Image.open(path) as im:
            original_format = im.format
            animated = bool(getattr(im, 'is_animated', False))
        rgb = image_rgb(path)
        h, w = rgb.shape[:2]
        info = dict(kind='image', width=w, height=h, duration=image_duration,
                    fps=30., audio=None, rotation=0, image_format=original_format,
                    animated_first_frame_only=animated, exif_transposed=True)
    except (OSError, ValueError, Image.DecompressionBombError):
        data = probe(path)
        stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'
                       and not s.get('disposition', {}).get('attached_pic')), None)
        if stream is None:
            raise ValueError('输入不是可解码的图片或视频')
        w, h = int(stream['width']), int(stream['height'])
        rotation = float(stream.get('tags', {}).get('rotate', 0) or 0)
        for side in stream.get('side_data_list', []):
            if 'rotation' in side:
                rotation = float(side['rotation'])
        if 45 < abs(rotation) % 180 < 135:
            w, h = h, w
        d = stream.get('duration') or data.get('format', {}).get('duration')
        duration = float(d) if d not in (None, 'N/A') else 0
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('无法确定有限的视频时长')
        source_fps = rate(stream.get('avg_frame_rate'), rate(stream.get('r_frame_rate')))
        fps = min(60., max(1., source_fps))
        audio = next((s for s in data['streams'] if s.get('codec_type') == 'audio'), None)
        info = dict(kind='video', width=w, height=h, duration=duration, fps=fps,
                    source_fps=source_fps, audio=audio, rotation=rotation,
                    video_stream_index=stream['index'],
                    timing_policy='FFmpeg fps filter to CFR; VFR timestamps approximated, not preserved',
                    rotation_policy='FFmpeg default autorotate before identical analysis/render scaling')
    limit = {'preview': 1280, 'hd': 1920, 'source': max(w, h)}[quality]
    factor = min(1., limit / max(w, h))
    dw, dh = max(1, round(w * factor)), max(1, round(h * factor))
    info.update(display_width=dw, display_height=dh, output_width=dw + dw % 2,
                output_height=dh + dh % 2, quality=quality, even_padding=[dw % 2, dh % 2])
    return info


def video_frames(path, info, width, height, fps, timeout=600, pad=False):
    vf = f'fps={fps:.10f},scale={width}:{height}:flags=bilinear'
    ow, oh = width, height
    if pad:
        ow, oh = width + width % 2, height + height % 2
        vf += f',pad={ow}:{oh}:0:0:color=black'
    proc = PipeProcess(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
                        '-filter_threads', '1', '-i', path, '-map',
                        f'0:{info["video_stream_index"]}', '-an', '-sn', '-dn',
                        '-vf', vf, '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-threads', '1', 'pipe:1'],
                       timeout=timeout, reader=True)
    size = ow * oh * 3
    try:
        while True:
            chunks, remaining = [], size
            while remaining:
                chunk = proc.p.stdout.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining == size:
                break
            if remaining:
                raise RuntimeError('FFmpeg解码返回不完整帧')
            yield np.frombuffer(b''.join(chunks), np.uint8).reshape(oh, ow, 3).copy()
        proc.finish()
    finally:
        proc.close()


def small_size(info):
    f = min(1., 480 / max(info['display_width'], info['display_height']))
    return max(1, round(info['display_width'] * f)), max(1, round(info['display_height'] * f))


def encode_image(image):
    ok, arr = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('缓存图像编码失败')
    return base64.b64encode(arr).decode('ascii')


def decode_image(text):
    arr = np.frombuffer(base64.b64decode(text), np.uint8)
    return cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def difference(a, b):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    diff = max(float(np.mean(cv2.absdiff(ga, gb))), float(np.mean(cv2.absdiff(a, b)))) / 255
    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(ha, ha, 1, 0, cv2.NORM_L1)
    cv2.normalize(hb, hb, 1, 0, cv2.NORM_L1)
    return diff, float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


def normalize_cuts(cuts, duration):
    result = []
    for t in sorted(set(float(c) for c in cuts)):
        if not math.isfinite(t) or not 0 < t < duration:
            raise ValueError('cuts必须为时长范围内的有限正数')
        if t - (result[-1] if result else 0) < .25 - 1e-6 or duration - t < .25 - 1e-6:
            raise ValueError('人工cuts产生小于0.25秒的镜头，请调整cuts')
        result.append(t)
    return result


def analyze(path, info, input_hash, cache_dir, manual_cuts=None, timeout=600):
    """One source decode; retain at most three frames/shot from that same pass.

    Representative is the exact detected cut frame. An isolated deterministic
    reservoir retains one later sample, and the last observed frame is retained.
    No frame is PNG-encoded until its shot is finalized; no per-frame body scan.
    """
    signature = dict(version=ENGINE_VERSION, opencv=cv2.__version__, detector='spatial-edge-context-3',
                     sampling='representative-reservoir-last-1', hash=input_hash, kind=info['kind'],
                     width=info['display_width'], height=info['display_height'],
                     duration=info['duration'], source_fps=info['fps'], cuts=manual_cuts)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_file = Path(cache_dir) / (key + '.json')
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding='utf-8'))
            if cached['signature'] == signature and cached['shots']:
                for shot in cached['shots']:
                    decode_image(shot['representative_png'])
                    for sample in shot['temporal_samples']:
                        decode_image(sample['png'])
                return cached, True
        except (ValueError, KeyError, cv2.error, TypeError, OSError):
            pass
    sw, sh = small_size(info)
    sampling_fps = info['fps'] if info['kind'] == 'video' else 1.
    detector = ShotDetector(sampling_fps, info['duration'])
    cuts = (normalize_cuts(manual_cuts, info['duration'])
            if manual_cuts is not None and info['kind'] == 'video' else None)
    shots, cut_events, state = [], [], None
    previous, count, manual_index = None, 0, 0
    import random
    sample_rng = random.Random(0)

    def start(t, frame):
        return dict(start=t, representative=frame.copy(), sample=None, last=None, seen=0)

    def finish(s, end):
        samples = {}
        for item in (s['sample'], s['last']):
            if item is not None and s['start'] < item[0] < end:
                samples[item[0]] = item[1]
        shots.append(dict(start=s['start'], end=end, representative_png=encode_image(s['representative']),
                          temporal_samples=[dict(time=t, png=encode_image(f)) for t, f in sorted(samples.items())]))

    frames = (video_frames(path, info, sw, sh, sampling_fps, timeout) if info['kind'] == 'video'
              else iter([cv2.resize(image_rgb(path), (sw, sh))]))
    try:
        for i, frame in enumerate(frames):
            count += 1
            t = i / sampling_fps
            if state is None:
                state = start(0., frame)
            if cuts is not None:
                events = []
                while manual_index < len(cuts) and t >= cuts[manual_index]:
                    events.append(dict(time=cuts[manual_index], frame=frame))
                    manual_index += 1
            else:
                events = detector.observe(frame, t, difference(previous, frame) if previous is not None else (0., 0.))
            for event in events:
                finish(state, event['time'])
                state = start(event['time'], event['frame'])
                if 'metrics' in event:
                    cut_events.append(event['metrics'])
            state['seen'] += 1
            if t > state['start']:
                state['last'] = (t, frame.copy())
                if sample_rng.randrange(state['seen']) == 0:
                    state['sample'] = (t, frame.copy())
            previous = frame
    finally:
        if hasattr(frames, 'close'):
            frames.close()
    if state is None:
        raise ValueError('分析未解码到任何帧')
    if cuts is None:
        for event in detector.observe(previous, min(info['duration'], count / sampling_fps), (0., 0.)):
            finish(state, event['time'])
            state = start(event['time'], event['frame'])
            cut_events.append(event['metrics'])
    else:
        while manual_index < len(cuts):
            finish(state, cuts[manual_index])
            state = start(cuts[manual_index], previous)
            manual_index += 1
    finish(state, info['duration'])
    result = dict(signature=signature, shots=shots, sampling_fps=sampling_fps, sampled_frames=count,
                  analysis_source_decode_passes=1 if info['kind']=='video' else 0,
                  protection_sample_counts=[1+len(s['temporal_samples']) for s in shots],
                  protection_sampling_policy='same-pass representative + reservoir + last; max 3 per shot',
                  cut_events=cut_events, cut_method='manual override' if cuts is not None else
                  'CFR all-frame RGB + HSV + spatial luminance + adaptive edge + persistence/return',
                  detector_stats=dict(rejected_returns=detector.rejected_returns, illumination_frames=detector.illumination_frames),
                  cut_time_resolution_seconds=1 / sampling_fps,
                  limitations=[f'最长边480、{sampling_fps:g}fps CFR多特征分镜；VFR逐帧PTS不保留。',
                               '0.25s持续/返回确认；极亮/暗恢复段、快运动、渐变仍可能误/漏切。',
                               '每镜头最多3帧人脸/文字启发式；没有全片主体保护，不能保证零遮挡。'])
    save_json(cache_file, result)
    return result, False


class DogLibrary:
    def __init__(self, assets):
        self.assets = Path(assets).resolve()
        manifest = json.loads((self.assets / 'dog_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('version') != 1 or not manifest.get('dogs'):
            raise ValueError('manifest须为version=1且dogs非空')
        self.entries = {}
        for dog in manifest['dogs']:
            if dog['id'] in self.entries:
                raise ValueError('manifest包含重复dog id')
            self.entries[dog['id']] = dog
        self.loaded_id, self.loaded = None, None

    def load(self, dog_id):
        if self.loaded_id == dog_id:
            return self.loaded
        e = self.entries[dog_id]
        path = (self.assets / e['file']).resolve()
        if not path.is_relative_to(self.assets) or not path.is_file():
            raise ValueError(f'{dog_id}: file必须在assets内且存在')
        if sha256(path) != e['sha256']:
            raise ValueError(f'{dog_id}: SHA256不匹配')
        with np.load(path, allow_pickle=False) as z:
            frames = z['frames']
            durations_ms = z['frame_durations_ms'] if 'frame_durations_ms' in z else None
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 4:
            raise ValueError(f'{dog_id}: frames必须uint8 N,H,W,4')
        n, h, w, _ = frames.shape
        if (n, w, h) != (int(e['frame_count']), int(e['width']), int(e['height'])) or n < 1:
            raise ValueError(f'{dog_id}: manifest尺寸/帧数不一致')
        if e.get('frame_timing') == 'gif-duration-ms' and durations_ms is None:
            raise ValueError(f'{dog_id}: 缺少GIF逐帧时长')
        if durations_ms is not None:
            if durations_ms.shape != (n,) or durations_ms.dtype.kind not in 'iu' or np.any(durations_ms <= 0):
                raise ValueError(f'{dog_id}: GIF逐帧时长必须为N个正整数毫秒')
            if 'frame_durations_ms' in e and durations_ms.tolist() != e['frame_durations_ms']:
                raise ValueError(f'{dog_id}: manifest逐帧时长不一致')
        fps = float(e['fps'])
        ax, ay = map(float, e['anchor'])
        if not math.isfinite(fps) or fps <= 0 or not all(map(math.isfinite, (ax, ay))):
            raise ValueError(f'{dog_id}: fps/anchor无效')
        if not 0 <= ax <= w or not 0 <= ay <= h:
            raise ValueError(f'{dog_id}: 脚底anchor不在裁切画布内')
        quality = e.get('quality', {})
        loop_start = int(e.get('loop_start_frame', quality.get('loop_start_frame', 0)))
        loop_end = int(e.get('loop_end_frame', quality.get('loop_end_frame', n)))
        if not 0 <= loop_start < loop_end <= n:
            raise ValueError(f'{dog_id}: loop区间须为0 <= start < end <= 原帧数（end不含）')
        jumps = [int(j) for j in quality.get('suspicious_color_jump_frames', [])]
        unsafe = [j for j in jumps if loop_start < j < loop_end]
        warnings = []
        if unsafe:
            loop_end = min(unsafe)
            warnings.append(f'{dog_id}: suspicious_color_jump_frames触发安全前缀[{loop_start},{loop_end})；源素材保留。')
        if loop_start in jumps:
            raise ValueError(f'{dog_id}: 循环起点已标记异常，不能默认安全循环')
        if loop_start != 0 or loop_end != n:
            warnings.append(f'{dog_id}: 仅循环原始帧[{loop_start},{loop_end})，原始{n}帧未删除；不宣称循环接缝已验证。')
        frames = frames[loop_start:loop_end]
        if durations_ms is not None:
            durations_ms = durations_ms[loop_start:loop_end]
        union = np.zeros((h, w), dtype=bool)
        for frame in frames:
            alpha_mask = frame[:, :, 3] > 0
            if not np.any(alpha_mask):
                raise ValueError(f'{dog_id}: 包含全透明帧，会违反每帧恰好新增一只狗')
            union |= alpha_mask
        yy, xx = np.nonzero(union)
        bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        data = dict(frames=frames, width=w, height=h, anchor=(ax, ay), fps=fps,
                    frame_count=len(frames), source_frame_count=n, loop_start_frame=loop_start,
                    loop_end_frame=loop_end, warnings=warnings, union_bbox=bbox, manifest=e)
        if durations_ms is not None:
            data['frame_durations_ms'] = durations_ms
            data['frame_starts_seconds'] = np.concatenate(([0.], np.cumsum(durations_ms[:-1], dtype=np.int64) / 1000.))
            data['loop_duration_seconds'] = int(durations_ms.sum()) / 1000.
        self.loaded_id, self.loaded = dog_id, data
        return data


def dog_frame_index(dog, frame_offset, output_fps, phase_frames=0):
    """Use native GIF delays when present; preserve legacy CFR timing otherwise."""
    if 'frame_starts_seconds' in dog:
        starts = dog['frame_starts_seconds']
        t = (float(starts[phase_frames]) + frame_offset / output_fps) % dog['loop_duration_seconds']
        return min(dog['frame_count'] - 1, max(0, int(np.searchsorted(starts, t + 1e-9, side='right')) - 1))
    return (phase_frames + math.floor(frame_offset * dog['fps'] / output_fps + 1e-7)) % dog['frame_count']


class ShuffleBag:
    def __init__(self, ids, rng):
        self.ids, self.rng, self.bag, self.previous = sorted(ids), rng, [], None

    def next(self):
        if not self.bag:
            self.bag = list(self.ids)
            self.rng.shuffle(self.bag)
            if len(self.bag) > 1 and self.bag[-1] == self.previous:
                self.bag[-1], self.bag[0] = self.bag[0], self.bag[-1]
        self.previous = self.bag.pop()
        return self.previous


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def validate_avoid_regions(regions):
    result = []
    for region in regions or []:
        if len(region) != 4:
            raise ValueError('--avoid-region需要四个坐标X1 Y1 X2 Y2')
        x1, y1, x2, y2 = [float(v) for v in region]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (x1,y1,x2,y2)) or not (x1 < x2 and y1 < y2):
            raise ValueError('--avoid-region须满足有限坐标0≤X1<X2≤1、0≤Y1<Y2≤1')
        result.append([x1,y1,x2,y2])
    return result


class PlacementError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def placement(rgb, info, dog, ratio, mode, rng, explicit_anchor=None, extra_frames=None,
              avoid_regions=None, allow_partial_overlap=False, adaptive=False, support_region=None):
    """Prefer supported, protected locations, then ensure a visible sticker.

    Search clear screen locations before relaxing automatic subject protection.
    Manual exclusions, explicit anchors and full canvas bounds remain hard.
    """
    validate_height_ratio(ratio)
    W, H = info['display_width'], info['display_height']
    sw, sh = rgb.shape[1], rgb.shape[0]
    user_regions = validate_avoid_regions(avoid_regions)
    manual = [[x*W,y*H,x2*W,y2*H] for x,y,x2,y2 in user_regions]
    faces, texts, heads, people, haar = [], [], [], [], True
    frames = [rgb] + [frame if frame.shape==rgb.shape else cv2.resize(frame,(sw,sh))
                     for frame in list(extra_frames or [])[:2]]
    for frame in frames:
        if frame.shape != rgb.shape:
            frame = cv2.resize(frame, (sw, sh))
        found, available = face_boxes(frame)
        faces.extend(found)
        texts.extend(text_regions(frame))
        if adaptive:
            people.extend(person_boxes(frame))
        for head in head_regions(frame):
            x,y,x2,y2 = head['box']
            heads.append(dict(head,box=[x*W/sw,y*H/sh,x2*W/sw,y2*H/sh]))
        haar &= available
    def scaled(boxes):
        return [[x*W/sw,y*H/sh,x2*W/sw,y2*H/sh]
                for x,y,x2,y2 in dict.fromkeys(tuple(b) for b in boxes)]
    faces, texts = scaled(faces), scaled(texts)
    # Both detectors can independently find a person. Merge their head boxes in
    # display coordinates across all sampled frames before extending body bounds.
    proxy_boxes = dict.fromkeys(tuple(box) for box in faces + [head['box'] for head in heads])
    proxies = subject_proxy_zones([dict(box=box) for box in proxy_boxes], H, W)
    people = scaled(people)
    proxies.extend(people)
    supports = detect_horizontal_supports(rgb)
    if adaptive:
        # Only verified foot points are automatic supports. A horizontal
        # building/sky edge alone provides no evidence of a standing surface.
        if support_region is not None:
            x1,y1,x2,y2 = support_region
            supports = [dict(line=[x1*sw,y1*sh,x2*sw,y1*sh],score=1.)]
    ground_regions = detect_ground_regions(rgb)
    saliency_mask = subject_saliency_mask(rgb)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    grad = cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0), cv2.Sobel(gray,cv2.CV_32F,0,1))
    cost_map = grad*.25 + np.abs(gray-float(gray.mean()))*.45
    integral = cv2.integral(cost_map)
    sizes = [ratio] if ratio is not None else [.35, .30, .24]
    if adaptive and ratio is None:
        sizes = [.28, .24, .20, .16]
    choices, setups = [], []
    rejections = dict(bounds=0, user_region=0, face=0, text=0, below_size_floor=0, no_support=0, subject=0, saliency=0)
    evaluated = []
    contact_changes = {}
    grass_points = {(round(gr['anchor'][0]), round(gr['anchor'][1])): gr
                    for gr in ground_regions if gr.get('surface_kind') == 'open_grass'}

    def candidate(x, y, rw, rh, ax, ay, requested, kind, support_py, support_score):
        rect = [x,y,x+rw,y+rh]
        if x < 0 or y < 0 or rect[2] > W or rect[3] > H:
            rejections['bounds'] += 1
            return None
        if any(intersects(rect,b) for b in manual):
            rejections['user_region'] += 1
            return None
        face_hit = any(intersects(rect,b) for b in faces)
        text_hit = any(intersects(rect,b) for b in texts)
        subject_hit = any(intersects(rect,b) for b in proxies)
        a,b,c,d = max(0,int(x*sw/W)),max(0,int(y*sh/H)),min(sw,math.ceil((x+rw)*sw/W)),min(sh,math.ceil((y+rh)*sh/H))
        saliency_overlap = float(np.mean(saliency_mask[b:d, a:c] > 0)) if d > b and c > a else 0.
        density = float((integral[d,c]-integral[b,c]-integral[d,a]+integral[b,a])/max(1,(c-a)*(d-b)))
        margins = [x/W,y/H,(W-x-rw)/W,(H-y-rh)/H]
        edge = min(margins)
        head_overlap = 0.
        for head in heads:
            hx,hy,hx2,hy2 = head['box']
            overlap = max(0,min(x+rw,hx2)-max(x,hx))*max(0,min(y+rh,hy2)-max(y,hy))
            head_overlap = max(head_overlap,head['confidence']*overlap/max(1,(hx2-hx)*(hy2-hy)))
        head_penalty = 8.0 * head_overlap
        point = (round((x+ax)*sw/W), round((y+ay)*sh/H))
        lawn = grass_points.get(point) if kind == 'ground_region' else None
        sky_penalty = 20.0 if y/H < .50 and density < .07 and not lawn else 0.
        soft = float(face_hit)*3. + float(text_hit)*2. + head_penalty + float(subject_hit)*8. + saliency_overlap*4. + sky_penalty
        cost = density + soft + max(0.,.35-rh/H) + .25*max(0.,.06-edge)/.06 - support_score
        contact_change=0.
        if adaptive and len(frames)>1:
            px,py=round((x+ax)*sw/W),round((y+ay)*sh/H)
            key=(px,py)
            if key not in contact_changes:
                dx,dy=max(2,round(sw*.035)),max(2,round(sh*.025))
                colors=[]
                for frame in frames:
                    patch=frame[max(0,py-dy):min(sh,py+dy),max(0,px-dx):min(sw,px+dx)]
                    if patch.size:
                        colors.append(np.median(patch.reshape(-1,3),axis=0))
                contact_changes[key]=max((float(np.linalg.norm(c-colors[0])) for c in colors),default=0.)
            contact_change=contact_changes[key]
            cost += .5*min(1.,contact_change/100.)
        if adaptive and lawn:
            cost -= .65
            cost += .65*abs((x+ax)/W-.80) + .85*abs((y+ay)/H-.64)
        if adaptive and ratio is None:
            cost += 1.5*abs(rh/H-.24) + 8.*max(0.,.16-rh/H)
        if kind == 'screen_fallback':
            cost += .15 * max(0., (H-y-rh)/H)
        return dict(x=x,y=y,width=rw,height=rh,anchor_pixel=[x+ax,y+ay],
                    anchor_normalized=[(x+ax)/W,(y+ay)/H],actual_height_ratio=rh/H,
                    evaluated_height_ratio=requested,cost=cost,joint_cost=cost,kind=kind,
                    ground_kind=kind,support_y=support_py/sh,support_score=support_score,
                    surface_kind='open_grass' if lawn else None,
                    automatic_support_region=lawn['support_region'] if lawn else None,
                    temporal_contact_color_change=contact_change,
                    automatic_overlap_penalty=soft,head_cue_overlap_penalty=head_penalty,
                    head_cue_overlap_fraction=head_overlap,face_hit=face_hit,text_hit=text_hit,
                    subject_hit=subject_hit,saliency_overlap_fraction=saliency_overlap,
                    detected_overlap=bool(face_hit or text_hit or subject_hit or saliency_overlap > .4),
                    canvas_edge_margins_normalized=margins,minimum_canvas_edge_margin=edge)

    def reject_automatic(p):
        for key, hit in (('face',p['face_hit']),('text',p['text_hit']),
                         ('subject',p['subject_hit']),('saliency',p['saliency_overlap_fraction'] > .4)):
            if hit:
                rejections[key] += 1
                return True
        return False

    def setup(requested):
        scale = min(H*requested/dog['height'], W*.88/dog['width'], H*.88/dog['height'])
        if adaptive and support_region is not None:
            scale = min(scale,(support_region[2]-support_region[0])*W*.85/dog['width'])
        rw, rh = max(1,round(dog['width']*scale)),max(1,round(dog['height']*scale))
        ax, ay = dog['anchor'][0]*rw/dog['width'],dog['anchor'][1]*rh/dog['height']
        if explicit_anchor is not None:
            px, py = explicit_anchor[0]*W,explicit_anchor[1]*H
            limits = [1.]
            for room, need in ((px,ax),(W-px,rw-ax),(py,ay),(H-py,rh-ay)):
                if need > 0:
                    limits.append(max(0.,room)/need)
            fit = min(limits)
            rw, rh = max(1,math.floor(rw*fit)),max(1,math.floor(rh*fit))
            ax, ay = dog['anchor'][0]*rw/dog['width'],dog['anchor'][1]*rh/dog['height']
            positions = [(round(px-ax),round(py-ay),'explicit_anchor', 0, 1.0)]
        else:
            positions = []
            for s in supports:
                x1, y1, x2, y2 = s['line']
                if adaptive and x2>x1 and rw > (x2-x1)*W/sw*.85:
                    continue
                for k in (0.2, 0.5, 0.8):
                    px, py = x1 + k*(x2-x1), y1 + k*(y2-y1)
                    positions.append((round(px*W/sw - ax), round(py*H/sh - ay), 'ground_anchored', py, s['score']))
            for gr in ground_regions:
                px, py = gr['anchor']
                positions.append((round(px*W/sw - ax), round(py*H/sh - ay), 'ground_region', py, gr['score']))

        return requested,rw,rh,ax,ay,positions

    for requested in sizes:
        _,rw,rh,ax,ay,positions = setup(requested)
        if not positions:
            rejections['no_support'] += 1
        setups.append((requested,rw,rh,ax,ay,positions))
        if not adaptive and ratio is None and (rh < H*HEIGHT_RATIO_MIN-1 or rh > H*HEIGHT_RATIO_MAX+1):
            rejections['below_size_floor'] += 1
            continue
        evaluated.append(requested)
        for x,y,kind,support_py,support_score in dict.fromkeys(positions):
            p = candidate(x,y,rw,rh,ax,ay,requested,kind,support_py,support_score)
            if p is not None and (allow_partial_overlap or not reject_automatic(p)):
                choices.append(p)

    fallback_used, overlap_required = not choices, False
    primary_rejections = dict(rejections)
    smaller_supported = False
    fallback_sizes = [v for v in [.20, .16, .12, .08] if ratio is None or v<=ratio]
    if not choices:
        small_setups = [setup(requested) for requested in fallback_sizes]
        # Shrink at real support points before moving to a floating sticker.
        for requested,rw,rh,ax,ay,positions in small_setups:
            evaluated.append(requested)
            for x,y,kind,support_py,support_score in dict.fromkeys(positions):
                p = candidate(x,y,rw,rh,ax,ay,requested,kind,support_py,support_score)
                if p is not None and (allow_partial_overlap or not reject_automatic(p)):
                    choices.append(p)
        smaller_supported = bool(choices)
        setups.extend(small_setups)
    if not choices:
        relaxed = []
        for requested,rw,rh,ax,ay,positions in setups:
            fallback_positions = list(positions)
            if explicit_anchor is None:
                # Bounded screen grid, plus exclusion edges to reach clear gaps.
                xs = {round(t*(W-rw)) for t in (0.,.03,.25,.5,.75,.97,1.)}
                ys = {round(t*(H-rh)) for t in (0.,.03,.25,.5,.75,.97,1.)}
                for bx,by,bx2,by2 in manual + faces + proxies:
                    xs.update((math.floor(bx-rw),math.ceil(bx2)))
                    ys.update((math.floor(by-rh),math.ceil(by2)))
                xs = sorted(x for x in xs if 0 <= x <= W-rw)
                ys = sorted(y for y in ys if 0 <= y <= H-rh)
                fallback_positions.extend((x,y,'screen_fallback',0,0.) for x in xs for y in ys)
            for x,y,kind,support_py,support_score in dict.fromkeys(fallback_positions):
                p = candidate(x,y,rw,rh,ax,ay,requested,kind,support_py,support_score)
                if p is None:
                    continue
                if allow_partial_overlap or not p['detected_overlap']:
                    choices.append(p)
                else:
                    # Only the last resort softens automatic protection. Face
                    # and text overlap are more costly than touching a body.
                    overlap = 0.
                    for boxes,weight in ((faces,100.),(texts,30.),(proxies,8.)):
                        for bx,by,bx2,by2 in boxes:
                            area = max(0,min(x+rw,bx2)-max(x,bx))*max(0,min(y+rh,by2)-max(y,by))
                            overlap += weight*area/max(1,(bx2-bx)*(by2-by))
                    overlap += 8.*p['saliency_overlap_fraction']*rw*rh/(W*H)
                    p['cost'] += overlap
                    p['joint_cost'] = p['cost']
                    relaxed.append(p)
            if requested not in evaluated:
                evaluated.append(requested)
        if not choices:
            choices, overlap_required = relaxed, bool(relaxed)
        if not choices:
            raise PlacementError('人工禁区或指定脚底落点与完整小狗画布冲突，请调整这些设置。',rejections)

    if adaptive and ratio is None:
        readable = [p for p in choices if p['actual_height_ratio'] >= .16-1/H]
        if readable:
            choices = readable
    best = min(p['cost'] for p in choices)
    shortlist = [p for p in choices if p['cost'] <= best+.03]
    chosen = dict(rng.choice(shortlist))
    policy = ('detected-soft-authorized' if allow_partial_overlap else
              'detected-soft-fallback' if overlap_required else 'detected-hard')
    ground_anchored = (chosen['kind'] in ('ground_anchored', 'ground_region'))
    size_fallback = chosen['height'] < H*(.16 if adaptive else HEIGHT_RATIO_MIN)-1
    warnings = ['使用有支撑的落点。' if ground_anchored else
                '使用指定脚底落点。' if chosen['kind'] == 'explicit_anchor' else
                '未找到合适的支撑面，已在画面内贴上小狗；此位置不表示真实站立。']
    if overlap_required:
        warnings.append('没有能完全避让主体的位置，已选择遮挡较少的位置以确保添加小狗。')
    if size_fallback:
        warnings.append('已使用更小的小狗，以适应可用空间并减少遮挡。')
    if not haar:
        warnings.append('本地Haar不可用，未做人脸检测，不下载模型。')
    if allow_partial_overlap:
        warnings.append('已显式授权：自动脸/文字保护为软惩罚，可部分遮挡；人工禁区和画面边界仍为硬约束。')
    if chosen['minimum_canvas_edge_margin'] < .055:
        warnings.append('内侧候选受限：使用有限边缘候选，不裁切动作。')
    ranked = []
    for p in sorted(choices,key=lambda p:p['cost']):
        if all(np.linalg.norm(np.asarray(p['anchor_normalized'])-q['anchor_normalized'])>.075 for q in ranked):
            ranked.append(p)
        if len(ranked)==6:
            break
    chosen.update(candidate_count=len(choices),best_candidate_count=len(shortlist),
                  ranked_candidates=ranked, evaluated_sizes=evaluated,
                  auto_size_range=([.16,.28] if adaptive else [.24,.35]) if ratio is None else None,size_mode='manual' if ratio else 'adaptive',
                  fallback_height_ratios=fallback_sizes,
                  requested_height_ratio=ratio,size_fallback=size_fallback,
                  placement_fallback=fallback_used,overlap_fallback=overlap_required,
                  fallback_reason=('automatic_overlap_required' if overlap_required else
                                   'smaller_supported_position' if smaller_supported else
                                   'reduced_size' if size_fallback else
                                   'no_supported_safe_position' if fallback_used else None),
                  primary_placement_rejections=primary_rejections,
                  placement_mode='fixed_ground_anchor' if ground_anchored else
                                 'explicit_anchor' if chosen['kind'] == 'explicit_anchor' else 'fixed_screen_sticker',
                  ground_anchored=ground_anchored,contact_shadow=False,whole_canvas_bounds_checked=True,
                  union_bbox_source=dog['union_bbox'], detected_faces=faces,detected_text_like_regions=texts,
                  detected_subject_proxy_regions=proxies,
                  detected_people_regions=people,
                  detected_head_like_regions=heads,head_cue_policy='bounded compact/dark-band-cheek proposals; always soft',
                  user_avoid_regions_normalized=user_regions,user_avoid_regions_pixels=manual,
                  protection_policy=policy,partial_overlap_accepted=allow_partial_overlap or overlap_required,
                  special_scene_fallback=allow_partial_overlap or fallback_used,protection_sample_count=len(frames),
                  automatic_detection_overlap=chosen['detected_overlap'],
                  protection_search_passes=2 if fallback_used else 1,temporal_relocation=False,haar_available=haar,
                  edge_fallback=chosen['minimum_canvas_edge_margin']<.055,
                  source_asset_resolution=[dog['width'],dog['height']],
                  output_asset_resolution=[chosen['width'],chosen['height']],analysis_resolution=[sw,sh],
                  selection_reason='优先选择有支撑且避让主体的位置；无合适落点时使用画面内备用位置。',
                  warnings=warnings)
    return chosen


class PreparedDog:
    """Current shot's pre-multiplied, resized frames; bounded LRU for larger assets."""
    def __init__(self, dog, position, byte_budget=64 * 1024 * 1024):
        from collections import OrderedDict
        self.dog, self.position, self.cache = dog, position, OrderedDict()
        self.limit = max(1, min(dog['frame_count'], byte_budget // max(1, position['width'] * position['height'] * 16)))

    def frame(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        src = self.dog['frames'][index].astype(np.float32) / 255
        rgb = src[:, :, :3]
        alpha = src[:, :, 3:4]
        if self.dog.get('manifest', {}).get('edge_cleanup', True):
            from edge_cleanup import cleanup_edge_rgb
            rgb, alpha = cleanup_edge_rgb(rgb, alpha)
        premul = np.concatenate([rgb * alpha, alpha], axis=2)
        size = (self.position['width'], self.position['height'])
        # High-quality sampling in premultiplied space. Bound Lanczos ringing to
        # valid associated-alpha colors, so invisible RGB/negative lobes cannot fringe.
        interp = cv2.INTER_AREA if size[0] <= src.shape[1] and size[1] <= src.shape[0] else cv2.INTER_LANCZOS4
        premul = cv2.resize(premul, size, interpolation=interp)
        raw_alpha = premul[:, :, 3:4].copy()
        straight = np.clip(premul[:, :, :3] / np.maximum(raw_alpha, 1e-8), 0, 1)
        straight[raw_alpha[:, :, 0] <= 0] = 0
        premul[:, :, 3:4] = np.clip(raw_alpha, 0, 1)
        premul[:, :, :3] = straight * premul[:, :, 3:4]
        value = (premul[:, :, :3] * 255, premul[:, :, 3:4])
        self.cache[index] = value
        if len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return value

    def overlay(self, rgb, index, offset=(0, 0)):
        p = self.position
        x, y = p['x'] + round(offset[0]), p['y'] + round(offset[1])
        if x < 0 or y < 0 or x+p['width'] > rgb.shape[1] or y+p['height'] > rgb.shape[0]:
            raise ValueError('motion offset exceeds full RGBA canvas bounds')
        roi = rgb[y:y + p['height'], x:x + p['width']]
        color, alpha = self.frame(index)
        roi[:] = np.clip(np.rint(color + roi * (1 - alpha)), 0, 255).astype(np.uint8)
