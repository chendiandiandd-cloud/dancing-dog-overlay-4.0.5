"""Delivery recovery without surface detection, tracking or foreground masks."""
from __future__ import annotations


def available_dog(library, requested, previous=None):
    errors = []
    ids = ([requested] if requested != previous else []) + [
        i for i in sorted(library.entries) if i != requested and i != previous]
    if previous in library.entries:
        ids.append(previous)
    for dog_id in dict.fromkeys(ids):
        try:
            return dog_id, library.load(dog_id)
        except Exception as exc:
            errors.append(f'{dog_id}: {exc}')
    raise ValueError('没有可读取的小狗素材：' + '; '.join(errors))


def sticker_position(info, dog, ratio=None, anchor=None, avoid_regions=None, protections=None):
    """Prefer clear visible space; relax exclusions only when they block delivery."""
    W, H = info['display_width'], info['display_height']
    requested = ratio if ratio is not None else .24
    scale = min(H*requested/dog['height'], W*.88/dog['width'], H*.88/dog['height'])
    w, h = max(1, min(W, round(dog['width']*scale))), max(1, min(H, round(dog['height']*scale)))
    ax, ay = dog['anchor'][0]*w/dog['width'], dog['anchor'][1]*h/dog['height']
    manual = [[a*W,b*H,c*W,d*H] for a,b,c,d in (avoid_regions or [])]
    protections = protections or {}
    boxes = [(box, weight) for key, weight in (('faces',100.),('texts',30.),('subjects',8.))
             for box in protections.get(key, [])]
    positions = [(round(fx*(W-w)), round(fy*(H-h)))
                 for fy in (.75,.97,.5,.25,.03) for fx in (.75,.03,.97,.25,.5)]
    desired = None
    if anchor is not None:
        desired = (round(anchor[0]*W-ax), round(anchor[1]*H-ay))
        positions.insert(0, (min(W-w,max(0,desired[0])), min(H-h,max(0,desired[1]))))
    def overlap(rect, box):
        a,b,c,d = rect
        x,y,x2,y2 = box
        return max(0,min(c,x2)-max(a,x))*max(0,min(d,y2)-max(b,y))/max(1,(c-a)*(d-b))
    candidates = []
    for x,y in dict.fromkeys(positions):
        rect = [x,y,x+w,y+h]
        blocked = sum(overlap(rect, b) for b in manual)
        penalty = sum(weight*overlap(rect, b) for b,weight in boxes)
        preferred = 0 if desired == (x,y) else .02 if anchor is not None else 0.
        candidates.append((blocked > 0, penalty > 0, blocked*100+penalty+preferred, x,y))
    # Full bounds always hold; subject/manual protection is best effort here.
    blocked, hit, cost, x,y = min(candidates)
    margins = [x/W,y/H,(W-x-w)/W,(H-y-h)/H]
    warnings = ['交付兜底：使用顶层小狗贴纸，不表示真实支撑或人物遮挡融合。']
    if blocked:
        warnings.append('所有可用位置与人工禁区冲突，为保证生成已放宽禁区。')
    if hit:
        warnings.append('无完全避让主体的位置，已接受一定遮挡以保证生成。')
    anchor_changed = desired is not None and (x,y) != desired
    if anchor_changed:
        warnings.append('指定落点无法容纳完整动作，已移动到画面内备用位置。')
    return dict(x=x,y=y,width=w,height=h,anchor_pixel=[x+ax,y+ay],
                anchor_normalized=[(x+ax)/W,(y+ay)/H],actual_height_ratio=h/H,
                requested_height_ratio=ratio,evaluated_height_ratio=requested,
                ground_anchored=False,ground_kind='screen_fallback',kind='screen_fallback',
                surface_kind=None,automatic_support_region=None,ranked_candidates=[],
                placement_mode='fixed_screen_sticker',placement_fallback=True,
                overlap_fallback=bool(hit or blocked),automatic_detection_overlap=bool(hit),
                partial_overlap_accepted=bool(hit or blocked),cost=cost,
                protection_policy='delivery-best-effort',delivery_sticker=True,
                user_avoid_regions_normalized=avoid_regions or [],
                user_constraints_relaxed=bool(blocked or anchor_changed),
                source_asset_resolution=[dog['width'],dog['height']],
                output_asset_resolution=[w,h],union_bbox_source=dog['union_bbox'],
                canvas_edge_margins_normalized=margins,minimum_canvas_edge_margin=min(margins),
                whole_canvas_bounds_checked=True,contact_shadow=False,
                warnings=warnings)


def make_sticker_plan(plan, library, reason):
    """Retain every planned shot and at least one healthy visible dog per shot."""
    info = plan['media']
    previous = None
    for shot in plan['shots']:
        first = shot['dogs'][0]
        dog_id, dog = available_dog(library, first['dog_id'], previous)
        previous = dog_id
        p = first['placement']
        valid = (0 <= p['x'] and 0 <= p['y'] and p['width'] > 0 and p['height'] > 0
                 and p['x']+p['width'] <= info['display_width']
                 and p['y']+p['height'] <= info['display_height']
                 and dog_id == first['dog_id'])
        if not valid:
            p = sticker_position(info,dog,plan.get('requested_height_ratio'),
                                 plan.get('requested_anchor'),plan.get('requested_avoid_regions'))
        else:
            p = dict(p, ground_anchored=False, surface_kind=None,
                     automatic_support_region=None, delivery_sticker=True)
        p.update(placement_mode='fixed_screen_sticker',kind='screen_fallback',
                 placement_fallback=True,fallback_reason=reason)
        first = dict(dog_id=dog_id,placement=p,phase_frames=first.get('phase_frames',0)%dog['frame_count'],
                     integration='top_layer_sticker',transforms=[],foreground=None,
                     support_region=None,fallback_reason=reason,manual=False)
        shot.update(dogs=[first],placement=p,dog_id=dog_id,anchor=p['anchor_normalized'],
                    height_ratio=p['actual_height_ratio'],dog_sha256=dog['manifest']['sha256'],
                    dog_frame_count=dog['frame_count'],dog_fps=dog['fps'])
        shot.pop('reference_png',None)
    plan.update(tracking='off',scene_fallback_count=len(plan['shots']),delivery_recovery=True,
                status='planned',placement_fallback_count=len(plan['shots']),
                overlap_fallback_count=sum(s['placement'].get('overlap_fallback',False) for s in plan['shots']))
    plan.pop('error',None)
    plan['warnings'].append(f'交付恢复：{reason}；改用每镜头一只顶层贴纸，允许一定遮挡。')
    return plan
