"""Local surface tracking and conservative foreground layering, entirely offline."""
from __future__ import annotations
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from motion_foreground import MotionForeground
from lightweight_vision import person_boxes

from overlay_core import (decode_image, encode_image, image_rgb, intersects, small_size,
                          validate_avoid_regions, validate_height_ratio, video_frames)

MUSIC_SUGGESTION = '可选《Runaway》— The National 作为配乐，观看效果更佳～'
IDENTITY = np.eye(3, dtype=np.float64)
MAX_TRACK_FRAMES = 18000


def validate_scene(scene, shot_count, kind, max_dogs, library, base_dir):
    if not isinstance(scene, dict) or set(scene)-{'shots'} or not isinstance(scene.get('shots'), list):
        raise ValueError('scene-plan须为仅含shots数组的对象')
    if len(scene['shots']) != shot_count:
        raise ValueError('scene-plan的shots数量须等于有效镜头数')
    for shot in scene['shots']:
        if not isinstance(shot, dict) or set(shot)-{'dogs'}:
            raise ValueError('scene-plan镜头仅接受dogs')
        dogs = shot.get('dogs')
        if not isinstance(dogs,list) or not 1 <= len(dogs) <= (1 if kind=='image' else max_dogs):
            raise ValueError('图片须一只狗；视频须1至max-dogs只狗')
        for dog in dogs:
            if not isinstance(dog,dict) or set(dog)-{'id','anchor','height_ratio','support_region','foreground'}:
                raise ValueError('scene-plan小狗字段无效')
            if dog.get('id') is not None and dog['id'] not in library.entries:
                raise ValueError('scene-plan包含未知小狗编号')
            validate_height_ratio(dog.get('height_ratio'))
            anchor = dog.get('anchor')
            if anchor is not None and (len(anchor)!=2 or any(not math.isfinite(v) or not 0<=v<=1 for v in anchor)):
                raise ValueError('scene-plan anchor须为两个0–1有限坐标')
            if dog.get('support_region') is not None:
                validate_avoid_regions([dog['support_region']])
                if anchor is None:
                    raise ValueError('人工support_region须同时提供anchor')
                x1,y1,x2,y2=dog['support_region']
                if not x1<=anchor[0]<=x2 or abs(anchor[1]-y1)>.04:
                    raise ValueError('anchor须在support_region上沿附近，横坐标须位于承载区内')
            fg = dog.get('foreground')
            if fg is None:
                continue
            if not isinstance(fg,dict) or set(fg)-{'mask','space','keyframes'}:
                raise ValueError('foreground仅接受mask/space/keyframes')
            if fg.get('space','screen') not in ('screen','support'):
                raise ValueError('foreground.space须为screen或support')
            if ('mask' in fg) == ('keyframes' in fg):
                raise ValueError('foreground须指定mask或keyframes之一')
            if 'mask' in fg:
                path = (Path(base_dir)/fg['mask']).resolve()
                if not path.is_file():
                    raise ValueError('foreground mask文件不存在')
                with Image.open(path) as im:
                    im.verify()
                fg['mask'] = str(path)
            else:
                keys = fg['keyframes']
                previous, vertices = -1., None
                if not isinstance(keys,list) or not keys:
                    raise ValueError('foreground.keyframes不能为空')
                if kind=='image' and len(keys)!=1:
                    raise ValueError('图片背景静止，遮挡仅接受一个polygon关键帧')
                for key in keys:
                    if set(key) != {'time','polygon'}:
                        raise ValueError('遮挡关键帧须含time和polygon')
                    t, poly = key['time'], key['polygon']
                    if not math.isfinite(t) or t<0 or t<=previous:
                        raise ValueError('遮挡time须为递增的镜头内秒数')
                    if len(poly)<3 or (vertices is not None and vertices!=len(poly)):
                        raise ValueError('遮挡polygon须至少3点，各关键帧点数一致')
                    if any(len(p)!=2 or any(not math.isfinite(v) or not 0<=v<=1 for v in p) for p in poly):
                        raise ValueError('遮挡polygon坐标须在0–1')
                    previous, vertices = t, len(poly)
    return scene


def matrix_at(dog, offset):
    transforms = dog.get('transforms')
    return np.asarray(transforms[min(offset,len(transforms)-1)],np.float64) if transforms else IDENTITY


def corners(position, matrix):
    x,y,w,h = (position[k] for k in ('x','y','width','height'))
    points = np.array([[x,y,1],[x+w,y,1],[x+w,y+h,1],[x,y+h,1]],np.float64)
    return (points @ matrix.T)[:,:2]


def rectangle(position, matrix):
    points = corners(position,matrix)
    return [float(points[:,0].min()),float(points[:,1].min()),
            float(points[:,0].max()),float(points[:,1].max())]


class SurfaceTracker:
    """Local incremental similarity; retain inliers and replenish in surface space."""
    def __init__(self, rgb, region):
        self.gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        h,w = self.gray.shape
        mask = np.zeros((h,w),np.uint8)
        x1,y1,x2,y2 = region
        x1,y1,x2,y2 = max(0,round(x1*w)),max(0,round(y1*h)),min(w,round(x2*w)),min(h,round(y2*h))
        mask[y1:y2,x1:x2] = 255
        for a,b,c,d in person_boxes(rgb):
            mask[max(0,int(b)):min(h,int(d)),max(0,int(a)):min(w,int(c))]=0
        self.region_mask = mask
        # Balance local tiles so detailed clothing cannot consume every feature
        # while lower-contrast floor texture goes unrepresented.
        features=[]
        for top,bottom in zip(np.linspace(y1,y2,4,dtype=int)[:-1],np.linspace(y1,y2,4,dtype=int)[1:]):
            for left,right in zip(np.linspace(x1,x2,4,dtype=int)[:-1],np.linspace(x1,x2,4,dtype=int)[1:]):
                tile=np.zeros_like(mask)
                tile[top:bottom,left:right]=mask[top:bottom,left:right]
                found=cv2.goodFeaturesToTrack(self.gray,12,.005,5,mask=tile,blockSize=5)
                if found is not None:
                    features.append(found)
        self.points=np.concatenate(features) if features else None
        self.original = None if self.points is None else self.points.copy()
        self.matrix = IDENTITY.copy()
        self.reason = None if self.points is not None and len(self.points)>=8 else 'insufficient_surface_features'

    def step(self, rgb):
        if self.reason:
            return None
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        nxt,status,_ = cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,
                                              winSize=(21,21),maxLevel=3)
        if nxt is None:
            self.reason = 'optical_flow_failed'
            return None
        back,back_status,_ = cv2.calcOpticalFlowPyrLK(gray,self.gray,nxt,None,
                                                   winSize=(21,21),maxLevel=3)
        if back is None:
            self.reason = 'backward_flow_failed'
            return None
        valid = (status[:,0]>0)&(back_status[:,0]>0)&(np.linalg.norm(back[:,0]-self.points[:,0],axis=1)<1.2)
        src,dst = self.points[valid],nxt[valid]
        if len(src)<8:
            self.reason = 'lost_surface_features'
            return None
        affine,inliers = cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.,maxIters=1000)
        if affine is None or float(inliers.mean())<.65:
            self.reason = 'inconsistent_surface_motion'
            return None
        matrix = IDENTITY.copy()
        matrix[:2] = affine
        delta = matrix.copy()
        matrix = delta @ self.matrix
        scale = math.hypot(matrix[0,0],matrix[1,0])
        angle = abs(math.atan2(delta[1,0],delta[0,0]))
        if not .5<scale<2. or angle>.12 or np.linalg.norm(delta[:2,2])>max(gray.shape)*.12:
            self.reason = 'implausible_surface_transform'
            return None
        keep = inliers[:,0].astype(bool)
        self.points = dst[keep]
        self.gray,self.matrix = gray,matrix
        if len(self.points)<70:
            mask = cv2.warpAffine(self.region_mask,matrix[:2],(gray.shape[1],gray.shape[0]),
                                  flags=cv2.INTER_NEAREST)
            for x,y in self.points[:,0]:
                cv2.circle(mask,(round(float(x)),round(float(y))),7,0,-1)
            fresh = cv2.goodFeaturesToTrack(gray,100-len(self.points),.015,5,mask=mask,blockSize=5)
            if fresh is not None:
                self.points = np.concatenate([self.points,fresh])
        return matrix


def tracking_region(dog, info):
    if dog.get('support_region'):
        return dog['support_region']
    p = dog['placement']
    if p.get('automatic_support_region'):
        return p['automatic_support_region']
    W,H = info['display_width'],info['display_height']
    ax,ay = p['anchor_normalized']
    half = max(.18,p['width']/W*.8)
    # Bottom foot points need an area above them as well as below them.
    return [max(0,ax-half),max(.72,ay-.12),min(1,ax+half),min(1,ay+.14)]


def foreground_mask(dog, frame, matrix, seconds, info):
    """White manual mask covers the dog; auto refines current moving silhouettes.

    Automatic residuals are not semantic segmentation. Only components extending
    below the contact plane are accepted; static rails require a manual mask.
    """
    H,W = frame.shape[:2]
    fg = dog.get('foreground')
    if fg:
        if 'mask' in fg:
            if '_mask' not in dog:
                with Image.open(fg['mask']) as im:
                    # Luminance, not file alpha: white=foreground, black=clear.
                    dog['_mask'] = np.asarray(im.convert('L')).copy()
            mask = cv2.resize(dog['_mask'],(W,H),interpolation=cv2.INTER_NEAREST)
        else:
            keys = fg['keyframes']
            left = keys[0]
            right = keys[-1]
            for a,b in zip(keys,keys[1:]):
                if a['time']<=seconds<=b['time']:
                    left,right = a,b
                    break
            if seconds<=keys[0]['time']:
                left=right=keys[0]
            elif seconds>=keys[-1]['time']:
                left=right=keys[-1]
            f = 0. if left['time']==right['time'] else (seconds-left['time'])/(right['time']-left['time'])
            poly = (1-f)*np.asarray(left['polygon'])+f*np.asarray(right['polygon'])
            mask = np.zeros((H,W),np.uint8)
            cv2.fillPoly(mask,[np.rint(poly*[W,H]).astype(np.int32)],255)
        if fg.get('space','screen')=='support':
            mask = cv2.warpAffine(mask,matrix[:2],(W,H),flags=cv2.INTER_NEAREST)
        return mask
    reference = dog.get('_reference')
    if reference is None:
        return np.zeros((H,W),np.uint8)
    if '_occlusion_state' not in dog:
        dog['_occlusion_state']=MotionForeground()
        # Bootstrap from the shot reference even if rendering starts later.
        dog['_occlusion_state'].previous=reference.copy()
        dog['_occlusion_state'].matrix=IDENTITY.copy()
        dog['_occlusion_state'].mask=np.zeros((H,W),np.uint8)
    p=dog['placement']
    foot=(matrix @ np.array([*p['anchor_pixel'],1.]))[:2]
    return dog['_occlusion_state'].update(frame,matrix,rectangle(p,matrix),foot[1])


def resize_position(p, factor):
    p = dict(p)
    ax,ay = p['anchor_pixel']
    fx,fy = (ax-p['x'])/p['width'],(ay-p['y'])/p['height']
    p['width'],p['height'] = max(1,round(p['width']*factor)),max(1,round(p['height']*factor))
    p['x'],p['y'] = round(ax-fx*p['width']),round(ay-fy*p['height'])
    if p.get('actual_height_ratio') is not None:
        p['actual_height_ratio']*=factor
    p['output_asset_resolution']=[p['width'],p['height']]
    return p


def render_mask(dog, frame, matrix, seconds, info):
    """Use the same analysis resolution and thresholds as the preflight."""
    reference=dog.get('_reference')
    if reference is None:
        W,H=info['display_width'],info['display_height']
        mask=foreground_mask(dog,frame[:H,:W],matrix,seconds,info)
        return cv2.copyMakeBorder(mask,0,frame.shape[0]-H,0,frame.shape[1]-W,cv2.BORDER_CONSTANT)
    sh,sw=reference.shape[:2]
    W,H=info['display_width'],info['display_height']
    scale=np.diag([sw/W,sh/H,1.])
    local=dict(dog,placement=dict(dog['placement']))
    p=local['placement']
    for key,factor in (('x',sw/W),('width',sw/W),('y',sh/H),('height',sh/H)):
        p[key]=round(p[key]*factor)
    p['anchor_pixel']=[p['anchor_pixel'][0]*sw/W,p['anchor_pixel'][1]*sh/H]
    mask=foreground_mask(local,cv2.resize(frame[:H,:W],(sw,sh)),scale @ matrix @ np.linalg.inv(scale),seconds,info)
    if local.get('_mask') is not None:
        dog['_mask']=local['_mask']
    if local.get('_occlusion_state') is not None:
        dog['_occlusion_state']=local['_occlusion_state']
    mask=cv2.resize(mask,(W,H),interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(mask,0,frame.shape[0]-H,0,frame.shape[1]-W,cv2.BORDER_CONSTANT)


def valid_trajectory(dog, info, avoid, peers):
    W,H = info['display_width'],info['display_height']
    for i,m in enumerate(dog['transforms']):
        rect = rectangle(dog['placement'],np.asarray(m))
        if rect[0]<0 or rect[1]<0 or rect[2]>W or rect[3]>H or any(intersects(rect,b) for b in avoid):
            return False
        if any(intersects(rect,rectangle(peer['placement'],matrix_at(peer,i))) for peer in peers):
            return False
    return True


def prepare_scene(path, plan, args):
    """Preflight the whole shot before rendering; failure never switches midshot."""
    info = plan['media']
    plan['foreground_backend']='backward_flow_current_appearance_grabcut_hog'
    if info['kind']=='image':
        plan['tracking'] = 'static'
        frame=cv2.resize(image_rgb(path),(info['display_width'],info['display_height']))
        for shot in plan['shots']:
            for dog in shot['dogs']:
                dog['integration'] = 'static_surface' if dog['placement']['ground_anchored'] or dog.get('support_region') else 'top_layer_sticker'
                if dog.get('foreground') and dog['integration']=='static_surface':
                    mask=foreground_mask(dog,frame,IDENTITY,0.,info)
                    p=dog['placement']
                    roi=mask[p['y']:p['y']+p['height'],p['x']:p['x']+p['width']]
                    if roi.size and np.mean(roi>0)>.85:
                        dog['integration']='top_layer_sticker'
                        dog['fallback_reason']='foreground_hides_dog'
                    dog.pop('_mask',None)
        plan['scene_fallback_count']=sum(any(d['integration']=='top_layer_sticker' for d in s['dogs']) for s in plan['shots'])
        return
    W,H = info['display_width'],info['display_height']
    sw,sh = small_size(info)
    scale = np.diag([W/sw,H/sh,1.])
    avoid = [[x*W,y*H,x2*W,y2*H] for x,y,x2,y2 in args.avoid_region]
    groups = []
    for shot in plan['shots']:
        shot_groups = []
        for dog in shot['dogs']:
            pool = [dog]
            p = dog['placement']
            if not dog.get('manual') and p['ground_anchored'] and not dog.get('support_region'):
                for candidate in p.get('ranked_candidates',[]):
                    if all(np.linalg.norm(np.asarray(candidate['anchor_normalized'])-
                                          d['placement']['anchor_normalized'])>.075 for d in pool):
                        alternate = dict(dog,placement={**p,**candidate})
                        alternate['placement']['output_asset_resolution']=[candidate['width'],candidate['height']]
                        pool.append(alternate)
                    if len(pool)==4:
                        break
            shot_groups.append(pool)
        groups.append(shot_groups)
        shot['dogs']=[d for pool in shot_groups for d in pool]
    index, trackers = -1, []
    frames = video_frames(path,info,sw,sh,info['fps'],args.timeout)
    try:
        for n,frame in enumerate(frames):
            current = max(0,index)
            while current+1<len(plan['shots']) and n>=plan['shots'][current+1]['frame_start']:
                current += 1
            shot = plan['shots'][current]
            if current!=index:
                index=current
                trackers=[]
                shot['reference_png']=encode_image(frame)
                reference=frame.copy()
                for dog in shot['dogs']:
                    supported = dog['placement']['ground_anchored'] or dog.get('support_region')
                    tracker = SurfaceTracker(frame,tracking_region(dog,info)) if supported and args.tracking!='off' else None
                    trackers.append(tracker)
                    dog['transforms']=[]
                    dog['_visible_factors']=[1.,.8,.6,.4]
                    dog['tracking_failure'] = (tracker.reason if tracker else 'tracking_disabled' if args.tracking=='off' else 'no_surface')
            offset=n-shot['frame_start']
            for dog,tracker in zip(shot['dogs'],trackers):
                if offset>=MAX_TRACK_FRAMES:
                    dog['tracking_failure']='shot_exceeds_tracking_budget'
                    continue
                if offset==0:
                    matrix=IDENTITY
                else:
                    matrix=tracker.step(frame) if tracker and not dog['tracking_failure'] else None
                if matrix is None:
                    dog['tracking_failure']=dog['tracking_failure'] or tracker.reason
                    matrix=IDENTITY
                full=scale @ matrix @ np.linalg.inv(scale)
                dog['transforms'].append(full.tolist())
                if not dog['tracking_failure']:
                    # Evaluate foreground visibility at analysis resolution.
                    local=dict(dog,placement={**dog['placement']})
                    p=local['placement']
                    for key,factor in (('x',sw/W),('width',sw/W),('y',sh/H),('height',sh/H)):
                        p[key]=round(p[key]*factor)
                    p['anchor_pixel']=[p['anchor_pixel'][0]*sw/W,p['anchor_pixel'][1]*sh/H]
                    local['_reference']=reference
                    mask=foreground_mask(local,frame,matrix,offset/info['fps'],info)
                    dog['_mask']=local.get('_mask') if local.get('_mask') is not None else dog.get('_mask')
                    if local.get('_occlusion_state') is not None:
                        dog['_occlusion_state']=local['_occlusion_state']
                    for factor in list(dog['_visible_factors']):
                        rect=rectangle(resize_position(p,factor),matrix)
                        x,y,x2,y2=[int(v) for v in rect]
                        roi=mask[max(0,y):min(sh,y2),max(0,x):min(sw,x2)]
                        if roi.size and float(np.mean(roi>0))>.85:
                            dog['_visible_factors'].remove(factor)
                    if not dog['_visible_factors']:
                        dog['tracking_failure']='foreground_hides_dog'
    finally:
        frames.close()
    for shot,shot_groups in zip(plan['shots'],groups):
        selected = []
        for pool in shot_groups:
            winner = pool[0]
            # Try readable trajectories at every point before reducing below 16%.
            trials = [(option, factor) for option in pool
                      for factor in option['_visible_factors']]
            if not winner.get('manual') and winner['placement'].get('requested_height_ratio') is None:
                trials.sort(key=lambda pair: (
                    pair[0]['placement']['height']*pair[1]/H < .16-1/H,
                    pool.index(pair[0]), -pair[1]))
            for option, factor in trials:
                if option['tracking_failure']:
                    continue
                if selected and not option.get('manual') and (
                        selected[0]['tracking_failure'] or
                        abs(option['placement']['anchor_normalized'][0]-
                            selected[0]['placement']['anchor_normalized'][0])<.25):
                    option['tracking_failure']='no_independent_second_role'
                    continue
                original = option['placement']
                option['placement']=resize_position(original,factor)
                if not valid_trajectory(option,info,avoid,selected):
                    option['placement']=original
                    continue
                winner=option
                option['_visible_factors']=[1.]
                break
            else:
                winner['tracking_failure']=winner['tracking_failure'] or 'trajectory_bounds_or_exclusion'
            if selected and not winner.get('manual') and winner['tracking_failure']:
                continue
            selected.append(winner)
        shot['candidate_preflight_counts']=[len(pool) for pool in shot_groups]
        shot['dogs']=selected
        peers=[]
        originals=[d['placement'] for d in shot['dogs']]
        for dog in shot['dogs']:
            original=dog['placement']
            if not dog['tracking_failure']:
                for factor in dog['_visible_factors']:
                    dog['placement']=resize_position(original,factor)
                    if valid_trajectory(dog,info,avoid,peers):
                        break
                else:
                    dog['tracking_failure']='trajectory_bounds_or_exclusion'
            if dog['tracking_failure']:
                dog['placement']=original
                dog['transforms']=[]
                dog['integration']='top_layer_sticker'
                dog['fallback_reason']=dog['tracking_failure']
                plan['warnings'].append(f'镜头{shot["start"]:g}s整镜头使用顶层贴纸：{dog["fallback_reason"]}。')
            else:
                dog['integration']='tracked_surface'
            dog.pop('_mask',None)
            dog.pop('_occlusion_state',None)
            dog.pop('_visible_factors',None)
            peers.append(dog)
        if any(d['integration']=='top_layer_sticker' for d in peers):
            # A shot has one consistent layer policy. Do not mix a failed
            # sticker with a moving surface role midway through the shot.
            for dog,original in zip(peers,originals):
                dog['placement']=original
                dog['transforms']=[]
                dog['integration']='top_layer_sticker'
                dog.setdefault('fallback_reason','peer_integration_failed')
        shot['placement']=shot['dogs'][0]['placement']
        shot['anchor']=shot['placement']['anchor_normalized']
        shot['height_ratio']=shot['placement']['height']/H
    plan['tracking']='local_surface_flow'
    plan['planning_source_decode_passes']=1
    plan['scene_fallback_count']=sum(any(d['integration']=='top_layer_sticker' for d in s['dogs']) for s in plan['shots'])


def composite(prepared, frame, index, matrix, mask=None):
    """Warp associated alpha; restore foreground from the original source frame."""
    color,alpha=prepared.frame(index)
    p=prepared.position
    transform=matrix @ np.array([[1,0,p['x']],[0,1,p['y']],[0,0,1]],np.float64)
    H,W=frame.shape[:2]
    rect=rectangle(p,matrix)
    x,y,x2,y2=max(0,math.floor(rect[0])),max(0,math.floor(rect[1])),min(W,math.ceil(rect[2])),min(H,math.ceil(rect[3]))
    if x2<=x or y2<=y:
        raise ValueError('小狗轨迹离开画面')
    transform=transform.copy()
    transform[0,2]-=x
    transform[1,2]-=y
    rgba=np.concatenate([color,alpha],axis=2)
    warped=cv2.warpAffine(rgba,transform[:2],(x2-x,y2-y),flags=cv2.INTER_LINEAR)
    a=np.clip(warped[:,:,3:4],0,1)
    c=warped[:,:,:3]
    if mask is not None:
        visible=1-mask[y:y2,x:x2,None].astype(np.float32)/255
        a=a*visible
        c=c*visible
    roi=frame[y:y2,x:x2]
    roi[:]=np.clip(np.rint(c+roi*(1-a)),0,255).astype(np.uint8)
