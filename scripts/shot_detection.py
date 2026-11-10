"""Offline streaming hard-cut features. No models or input-specific timestamps."""
from collections import deque

import cv2
import numpy as np


def features(rgb):
    small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    spatial = cv2.resize(cv2.GaussianBlur(gray, (5, 5), 0), (64, 36))
    # Centering separates lighting pulses from spatial layout changes.
    centered = spatial - spatial.mean()
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edge = cv2.resize(cv2.magnitude(gx, gy), (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return dict(spatial=spatial, centered=centered, edge=edge, hist=hist,
                mean=float(gray.mean()), extreme=float(np.mean((gray < .04) | (gray > .96))))


def distances(a, b):
    return dict(luminance=float(np.mean(np.abs(a['spatial'] - b['spatial']))),
                layout=float(np.mean(np.abs(a['centered'] - b['centered']))),
                edge=float(np.mean(np.abs(a['edge'] - b['edge']))),
                hsv=float(cv2.compareHist(a['hist'], b['hist'], cv2.HISTCMP_BHATTACHARYYA)))


class ShotDetector:
    """Bounded candidate queue; observe every CFR frame, confirm after 0.25s.

    Spatial cuts need both layout and edge evidence, not a global color jump.
    Illumination extremes and their recovery are not independent hard cuts.
    """
    def __init__(self, fps, duration):
        self.fps, self.duration = fps, duration
        self.previous = None
        self.history = deque(maxlen=max(6, round(fps)))
        self.pending = deque()
        self.last_cut = 0.
        self.recovery_until = -1.
        self.rejected_returns = 0
        self.illumination_frames = 0

    def observe(self, frame, t, rgb_metrics):
        current = features(frame)
        committed = []
        extreme = current['mean'] > .85 or current['mean'] < .035
        if extreme:
            self.illumination_frames += 1
            self.recovery_until = t + .25
            self.pending.clear()
        if self.previous is not None:
            m = distances(self.previous, current)
            d, h = rgb_metrics
            baseline = {k: float(np.median([v[k] for v in self.history]))
                        if self.history else 0. for k in m}
            bd = float(np.median([v['rgb'] for v in self.history])) if self.history else 0.
            bh = float(np.median([v['rgb_hist'] for v in self.history])) if self.history else 0.
            recent = [v for v in self.history if t - v['time'] < .25 - 1e-7]
            # A burst of hand/camera motion is not an isolated discontinuity.
            peak = (m['layout'] > max([v['layout'] for v in recent], default=0.) * 1.8
                    and m['edge'] > max([v['edge'] for v in recent], default=0.) * 1.4)
            spatial = (peak and m['layout'] > max(.045, baseline['layout'] * 4 + .015)
                       and m['edge'] > max(.06, baseline['edge'] * 3 + .025))
            strong = d > .13 and h > .38
            color = h > max(.65, bh * 3 + .15) and d > max(.035, bd * 4 + .015)
            # A fixed high structural floor preserves cuts after turbulent motion.
            structural = peak and m['layout'] > .12 and m['edge'] > .12
            turbulent = (not peak and m['layout'] > max(.14, baseline['layout']*3)
                         and m['edge'] > .16 and d > .18 and m['hsv'] > .30)
            rule = ('motion_context_jump' if turbulent else
                    'adaptive_color_jump' if color and not strong else
                    'spatial_edge_jump' if spatial or structural else 'strong_change')
            if (spatial or structural or strong or color or turbulent) and t > self.recovery_until + 1e-7:
                self.pending.append(dict(time=t, before=self.previous, after=current,
                                         frame=frame.copy(), metrics=dict(start=t, difference=d,
                                         histogram=h, **m, baseline=baseline, rule=rule)))
            m.update(rgb=d, rgb_hist=h, time=t)
            self.history.append(m)
            keep = deque()
            returning = False
            for candidate in self.pending:
                # The falling edge of an impulse is not a second new shot.
                if returning and candidate['time'] == t:
                    continue
                # At the deadline the preceding CFR frame covers the end of
                # the confirmation interval. A new cut exactly 0.25s later must
                # not overwrite that evidence with its own first frame.
                mature = t - candidate['time'] >= .25 - 1e-7
                evaluated = self.previous if mature else current
                pre = distances(candidate['before'], evaluated)
                initial = candidate['metrics']
                # Require structural return as well as color similarity. Same-color
                # new shots must not be mistaken for a return to the old shot.
                returned = (pre['layout'] < max(.012, initial['layout'] * .22)
                            and pre['edge'] < max(.025, initial['edge'] * .25)
                            and pre['hsv'] < .22
                            and pre['luminance'] < max(.012, initial['luminance'] * .22))
                if returned:
                    returning = True
                    self.rejected_returns += 1
                    continue
                if not mature:
                    keep.append(candidate)
                    continue
                post = distances(candidate['after'], evaluated)
                if initial['rule'] == 'adaptive_color_jump':
                    persistent = post['hsv'] <= max(.18, pre['hsv'] * .65)
                else:
                    persistent = (post['layout'] <= max(.025, pre['layout'] * .65)
                                  and post['edge'] <= max(.04, pre['edge'] * .8))
                # Allow bounded settling for a real cut followed by zoom/motion.
                # Require independent color AND layout separation; do not accept
                # each scene-score pulse in a mirrored/rapidly moving sequence.
                contextual = (initial['hsv'] > .30 and pre['hsv'] > .28
                              and post['hsv'] < pre['hsv']*.85
                              and post['layout'] < pre['layout']*.90
                              and post['edge'] < pre['edge']*.95)
                if not persistent and not contextual and t-candidate['time'] < .5-1e-7:
                    keep.append(candidate)
                    continue
                extended = not persistent or initial['rule']=='motion_context_jump'
                persistent = contextual if initial['rule']=='motion_context_jump' else persistent or contextual
                ct = candidate['time']
                if (persistent and ct - self.last_cut >= (.75 if extended else .25) - 1e-7
                        and self.duration - ct >= .25 - 1e-7):
                    committed.append(candidate)
                    self.last_cut = ct
                    self.history.clear()
            self.pending = keep
        self.previous = current
        return committed
