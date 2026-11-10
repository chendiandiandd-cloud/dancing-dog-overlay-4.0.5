#!/usr/bin/env python3
"""Offline surface compositor: one dog/image, one or two dogs/video shot."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
import time
import uuid
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

import cv2
from overlay_core import (ENGINE_VERSION, RNG_SALT, DogLibrary, PipeProcess, PreparedDog,
                          ShuffleBag, analyze, decode_image, image_rgb, media_info,
                          placement, probe, rate, run, save_json, sha256, dog_frame_index,
                          validate_avoid_regions, validate_height_ratio, video_frames,
                          small_size, encode_image)
from scene_integration import (MUSIC_SUGGESTION, composite, render_mask, matrix_at,
                               prepare_scene, validate_scene)
from delivery_fallback import available_dog, sticker_position, make_sticker_plan

SKILL_ROOT = Path(__file__).resolve().parents[1]
BASE_WARNINGS = [
    '图片新增一只狗，视频每镜头新增1–2只；不检测或清除原画面已有狗。',
    '选位使用最多3帧主体提示；视频逐帧跟踪局部纹理，自动前景遮挡为运动差分启发式。',
    '轻量方法不识别语义景深；复杂载体或静态栏杆需scene-plan指定承载区和遮挡。',
    '优先选择有支撑且避让主体的落点；无合适位置时使用备用位置，确保添加小狗并记录选位降级。',
    '有效源fps限制1–60并转换CFR，VFR逐帧PTS不保留；时长可相差约一个输出帧。',
    'quality只缩小不放大；奇数宽高在右/底补1像素。',
]


def parser():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--inputs',nargs='+',metavar='FILE',help='独立图片/视频，最多64个')
    p.add_argument('--output-dir',type=Path,default=SKILL_ROOT/'assets'/'temp_data'/'output',
                   help='结果目录，默认assets/temp_data/output，自动创建独立任务子目录')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--image-duration',type=float,default=5.,help='图片统一5秒，仅接受5')
    p.add_argument('--quality',choices=['preview','hd','source'],default='hd')
    p.add_argument('--placement-mode',choices=['auto','corner'],default='auto')
    p.add_argument('--dog-id',help='指定第一镜头，后续镜头相邻不同')
    p.add_argument('--dog-sequence',help='逗号分隔每镜头dog id；须与镜头数一致且相邻不同')
    p.add_argument('--height-ratio',type=float,help='完整动画画布高度比例[.04,.60]；默认优先可辨识尺寸')
    p.add_argument('--max-dogs',type=int,choices=[1,2],default=2,help='视频每镜头数量上限；图片始终1')
    p.add_argument('--scene-plan',type=Path,help='逐镜头承载区、落点、狗数量、前景遮挡JSON')
    p.add_argument('--anchor',nargs=2,type=float,metavar=('X','Y'),help='优先使用指定归一化脚底；无法容纳完整动作时移动到备用位置以保证交付')
    p.add_argument('--avoid-region',nargs=4,type=float,action='append',default=[],metavar=('X1','Y1','X2','Y2'),help='可重复人工避让区域；阻止交付时放宽并记录原因')
    p.add_argument('--cuts',help='人工切点秒数逗号分隔，空字符串为一个镜头')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--music',type=Path,help='循环/裁切替换音频，编码AAC')
    p.add_argument('--timeout',type=float,default=600.)
    p.add_argument('--allow-special-fallback',action='store_true',help='提前允许部分遮挡；默认先避让，交付兜底可放宽人工区域，完整动作保留在画面内')
    p.add_argument('--tracking',choices=['auto','flow','off'],default='auto',help='默认局部光流跟踪；off整镜头顶层贴纸')
    return p


def validate_args(args,p):
    try:
        args.avoid_region = validate_avoid_regions(args.avoid_region)
    except ValueError as exc:
        p.error(str(exc))
    if args.image_duration != 5.:
        p.error('图片统一5秒，--image-duration仅接受5')
    try:
        validate_height_ratio(args.height_ratio)
    except ValueError as exc:
        p.error(str(exc))
    if args.anchor is not None and any(not math.isfinite(x) or not 0<=x<=1 for x in args.anchor):
        p.error('--anchor每个坐标须在[0,1]')
    if not math.isfinite(args.timeout) or args.timeout<1:
        p.error('--timeout必须为至少1秒的有限数')
    try:
        args.cut_values = (None if args.cuts is None else [] if not args.cuts.strip()
                           else [float(c.strip()) for c in args.cuts.split(',')])
        if args.cut_values is not None and any(not math.isfinite(x) for x in args.cut_values):
            raise ValueError()
    except ValueError:
        p.error('--cuts须为逗号分隔有限秒数')
    if not args.inputs:
        p.error('须提供--inputs')
    if len(args.inputs)>64:
        p.error('单worker批次最多64个输入')
    if args.music is not None and not args.music.is_file():
        p.error('--music文件不存在')
    if args.scene_plan is not None and not args.scene_plan.is_file():
        p.error('--scene-plan文件不存在')
    if args.scene_plan is not None and len(args.inputs)!=1:
        p.error('scene-plan逐文件指定，每次调用仅接受一个输入')
    args.dog_sequence = [x.strip() for x in args.dog_sequence.split(',')] if args.dog_sequence else None
    if args.dog_sequence and (any(not x for x in args.dog_sequence) or
                              any(a==b for a,b in zip(args.dog_sequence,args.dog_sequence[1:]))):
        p.error('--dog-sequence须非空且相邻狗不同')


def create_plan(path,output,info,analysis,library,args,digest):
    rng_seed = int(hashlib.sha256(f'{RNG_SALT}:{args.seed}:{digest}'.encode()).hexdigest(),16)
    # Independent deterministic streams keep dog selection stable when placement
    # candidates or animation phases change.
    dog_rng = random.Random(rng_seed)
    placement_rng = random.Random(rng_seed)
    phase_rng = random.Random(rng_seed ^ 0x5048415345)
    bag = ShuffleBag(library.entries,dog_rng)
    if args.dog_id and args.dog_id not in library.entries:
        raise ValueError(f'未知dog-id: {args.dog_id}')
    plan = dict(schema_version=4,engine_version=ENGINE_VERSION,input=str(path),output=str(output),
                status='planned',shots=[],timings={},warnings=list(BASE_WARNINGS),seed=args.seed,
                rng_input_key=digest,rng_salt=RNG_SALT,rng_schema='isolated-dog-placement-phase-v1',
                dog_selection_policy='explicit sequence replay' if args.dog_sequence else 'isolated shuffle bag',
                requested_dog_sequence=args.dog_sequence,media=info,tracking=args.tracking,
                music_suggestion=MUSIC_SUGGESTION,max_dogs=1 if info['kind']=='image' else args.max_dogs,
                requested_height_ratio=args.height_ratio,requested_anchor=args.anchor,
                requested_cuts=args.cut_values,requested_avoid_regions=args.avoid_region,
                placement_assistance='manual_anchor' if args.anchor is not None else 'user_avoid_regions' if args.avoid_region else 'none',
                analysis={k:v for k,v in analysis.items() if k not in ('shots','signature')},
                protection_backend='lightweight',allow_special_fallback=args.allow_special_fallback,
                protection_policy='detected-soft-authorized' if args.allow_special_fallback else 'detected-hard',
                planning_source_decode_passes=0,tracking_analysis_seconds=0.,
                audio_policy='music replaces first source audio' if args.music else 'first source audio preserved; copy compatible codec else AAC' if info['audio'] else 'silent',
                music=str(args.music.resolve()) if args.music else None,plan_only=args.plan_only)
    if info['kind'] == 'image' and args.cut_values is not None:
        plan['warnings'].append('图片忽略--cuts，始终一个镜头；人工切点只适用于视频。')
    if info.get('animated_first_frame_only'):
        plan['warnings'].append('动画图片仅使用转正首帧。')
    plan['warnings'].extend(analysis['limitations'])
    fps = info['fps']
    effective = []
    for shot in analysis['shots']:
        start = math.ceil(shot['start']*fps-1e-7)/fps
        if effective and (start-effective[-1]['start']<.25-1e-6 or info['duration']-start<.25-1e-6):
            plan['warnings'].append(f'切点{shot["start"]:.6f}s在CFR量化后过短，已合并。')
            continue
        effective.append(dict(shot,start=start))
    if len(effective)>1 and len(library.entries)<2:
        plan['warnings'].append('仅一只素材可用，各镜头重复使用以保证交付。')
    scene = None
    if args.scene_plan:
        scene = validate_scene(json.loads(args.scene_plan.read_text(encoding='utf-8')),
                               len(effective),info['kind'],args.max_dogs,library,args.scene_plan.resolve().parent)
        if args.dog_sequence or args.dog_id or args.anchor is not None or args.height_ratio is not None:
            raise ValueError('scene-plan不能与dog-id/dog-sequence/anchor/height-ratio同时使用')
    if args.dog_sequence:
        if len(args.dog_sequence)!=len(effective):
            raise ValueError('--dog-sequence长度必须等于有效镜头数；可用--cuts固定切点')
        if any(d not in library.entries for d in args.dog_sequence):
            raise ValueError('--dog-sequence包含未知dog id')
        if args.dog_id and args.dog_sequence[0]!=args.dog_id:
            raise ValueError('--dog-id与--dog-sequence首狗冲突')
    for i,shot in enumerate(effective):
        end = effective[i+1]['start'] if i+1<len(effective) else info['duration']
        if args.dog_sequence:
            dog_id = args.dog_sequence[i]
        elif i==0 and args.dog_id:
            dog_id = args.dog_id
            bag.bag = [d for d in bag.ids if d!=dog_id]
            dog_rng.shuffle(bag.bag)
            bag.previous = dog_id
        else:
            dog_id = bag.next()
        requested_id = dog_id
        dog_id, dog = available_dog(library, dog_id, plan['shots'][-1]['dog_id'] if plan['shots'] else None)
        if dog_id != requested_id:
            plan['warnings'].append(f'{requested_id}素材不可用，改用{dog_id}以保证交付。')
        spec = scene['shots'][i]['dogs'][0] if scene else {}
        if spec.get('id'):
            dog_id = spec['id']
            dog_id, dog = available_dog(library, dog_id)
        ratio,anchor = spec.get('height_ratio',args.height_ratio),spec.get('anchor',args.anchor)
        rgb = decode_image(shot['representative_png'])
        extra = [decode_image(s['png']) for s in shot.get('temporal_samples',[])[:2]]
        try:
            if getattr(args,'_delivery_recovery',False):
                pos = sticker_position(info,dog,ratio,anchor,args.avoid_region)
            else:
                pos = placement(rgb,info,dog,ratio,args.placement_mode,placement_rng,
                                anchor,extra_frames=extra,avoid_regions=args.avoid_region,
                                allow_partial_overlap=args.allow_special_fallback,adaptive=True,
                                support_region=spec.get('support_region'))
        except Exception as exc:
            pos = sticker_position(info,dog,ratio,anchor,args.avoid_region)
            pos['recovered_from'] = dict(type=type(exc).__name__,message=str(exc))
            plan['warnings'].append(f'镜头{shot["start"]:g}s选位异常，已生成备用贴纸：{exc}')
        plan['shots'].append(dict(start=shot['start'],end=end,dog_id=dog_id,
                                  anchor=pos['anchor_normalized'],height_ratio=pos['actual_height_ratio'],
                                  phase_frames=phase_rng.randrange(dog['frame_count']),frame_start=round(shot['start']*fps),
                                  dog_fps=dog['fps'],dog_frame_count=dog['frame_count'],
                                  dog_frame_timing=dog['manifest'].get('frame_timing','constant-fps'),
                                  dog_frame_durations_ms=(dog['frame_durations_ms'].tolist()
                                                          if 'frame_durations_ms' in dog else None),
                                  dog_source_frame_count=dog['source_frame_count'],dog_loop_start_frame=dog['loop_start_frame'],
                                  dog_loop_end_frame=dog['loop_end_frame'],dog_sha256=dog['manifest']['sha256'],
                                  placement=pos,placement_segments=[],temporal_placement_fallback=False))
        plan['warnings'].extend(dog['warnings'])
        plan['warnings'].extend(pos['warnings'])
        first = dict(dog_id=dog_id,placement=pos,phase_frames=plan['shots'][-1]['phase_frames'],
                     support_region=spec.get('support_region'),foreground=spec.get('foreground'),manual=bool(scene))
        dogs = [first]
        specs = scene['shots'][i]['dogs'][1:] if scene else (
            [{}] if info['kind']=='video' and args.max_dogs==2 and pos['ground_anchored']
            and pos.get('surface_kind') != 'open_grass' else [])
        for second_spec in specs:
            available=[d for d in sorted(library.entries) if d!=dog_id]
            if not available and not second_spec.get('id'):
                continue
            second_id=second_spec.get('id') or placement_rng.choice(available)
            try:
                second_dog=library.load(second_id)
            except (OSError,ValueError,KeyError):
                plan['warnings'].append(f'第二只狗{second_id}素材不可用，保留第一只狗。')
                continue
            # Full animation canvases must remain disjoint. Second automatic dog
            # is accepted only on a clear support with a distinct screen role.
            exclusion=[pos['x']/info['display_width'],pos['y']/info['display_height'],
                       (pos['x']+pos['width'])/info['display_width'],
                       (pos['y']+pos['height'])/info['display_height']]
            try:
                second_pos=placement(rgb,info,second_dog,second_spec.get('height_ratio'),args.placement_mode,
                                     placement_rng,second_spec.get('anchor'),extra_frames=extra,
                                     avoid_regions=args.avoid_region+[exclusion],adaptive=True,
                                     support_region=second_spec.get('support_region'))
            except Exception:
                plan['warnings'].append('第二只狗选位不可用，保留第一只狗以保证交付。')
                continue
            if scene or (second_pos['ground_anchored'] and not second_pos['automatic_detection_overlap']
                         and second_pos['actual_height_ratio']<=.25
                         and abs(second_pos['anchor_normalized'][0]-pos['anchor_normalized'][0])>.25):
                dogs.append(dict(dog_id=second_id,placement=second_pos,
                                 phase_frames=phase_rng.randrange(second_dog['frame_count']),
                                 support_region=second_spec.get('support_region'),
                                 foreground=second_spec.get('foreground'),manual=bool(scene)))
        plan['shots'][-1]['dogs']=dogs
    plan['warnings'] = list(dict.fromkeys(plan['warnings']))
    plan['placement_fallback_count'] = sum(s['placement']['placement_fallback'] for s in plan['shots'])
    plan['overlap_fallback_count'] = sum(s['placement']['overlap_fallback'] for s in plan['shots'])
    if plan['overlap_fallback_count'] and not args.allow_special_fallback:
        plan['protection_policy'] = 'detected-hard-with-last-resort-fallback'
    return plan


def write_all(stream,data):
    view = memoryview(data)
    while len(view):
        n = stream.write(view)
        if not n:
            raise RuntimeError('FFmpeg编码器管道已关闭')
        view = view[n:]


def render_video(path,target,plan,library,timeout):
    info = plan['media']
    W,H,fps = info['output_width'],info['output_height'],info['fps']
    encoded,muxed = target.with_suffix('.video.part.mp4'),target.with_suffix('.mux.part.mp4')
    encoder,frames,n = None,None,0
    timings = dict(decode_overlay_encode=0.,mux=0.,verify=0.)
    started = time.perf_counter()
    try:
        encoder = PipeProcess(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-filter_threads','1',
                               '-f','rawvideo','-pix_fmt','rgb24','-s:v',f'{W}x{H}','-r',f'{fps:.10f}',
                               '-i','pipe:0','-an','-c:v','libx264','-preset','ultrafast','-crf',
                               '23' if info['quality']=='preview' else '20','-pix_fmt','yuv420p','-threads','1',
                               '-movflags','+faststart',encoded],timeout=timeout,writer=True)
        if info['kind']=='image':
            base = cv2.resize(image_rgb(path),(info['display_width'],info['display_height']),interpolation=cv2.INTER_AREA)
            base = cv2.copyMakeBorder(base,0,H-base.shape[0],0,W-base.shape[1],cv2.BORDER_CONSTANT)
            frames = (base.copy() for _ in range(max(1,math.ceil(info['duration']*fps-1e-7))))
        else:
            frames = video_frames(path,info,info['display_width'],info['display_height'],fps,timeout,pad=True)
        shot_index,prepared = -1,None
        for n0,frame in enumerate(frames):
            current = max(0,shot_index)
            while current+1<len(plan['shots']) and n0>=plan['shots'][current+1]['frame_start']:
                current += 1
            if current!=shot_index:
                shot_index = current
                shot = plan['shots'][current]
                prepared = [PreparedDog(library.load(d['dog_id']),d['placement']) for d in shot['dogs']]
                for d in shot['dogs']:
                    if shot.get('reference_png'):
                        d['_reference']=decode_image(shot['reference_png'])
            shot = plan['shots'][current]
            source=frame.copy()
            offset=n0-shot['frame_start']
            for d,asset in zip(shot['dogs'],prepared):
                index = dog_frame_index(asset.dog,offset,fps,d['phase_frames'])
                if plan.get('delivery_recovery'):
                    # Independent of the scene compositor and foreground backend.
                    asset.overlay(frame,index)
                else:
                    matrix=matrix_at(d,offset)
                    mask=(render_mask(d,source,matrix,offset/fps,info)
                          if d['integration']!='top_layer_sticker' else None)
                    composite(asset,frame,index,matrix,mask)
            write_all(encoder.p.stdin,frame.tobytes())
            n = n0+1
        encoder.finish()
        if n==0:
            raise RuntimeError('没有输出视频帧')
        timings['decode_overlay_encode'] = time.perf_counter()-started
        duration = n/fps
        plan.update(encoded_frames=n,effective_duration=duration,render_source_decode_passes=1 if info['kind']=='video' else 0)
        if abs(duration-info['duration'])>1/fps+.025:
            plan['warnings'].append('有效解码时长与容器声明差异超过一个输出帧；全部有效帧保留。')
        plan['shots'] = [s for s in plan['shots'] if s['frame_start']<n]
        for i,shot in enumerate(plan['shots']):
            shot['end'] = plan['shots'][i+1]['start'] if i+1<len(plan['shots']) else duration
        t = time.perf_counter()
        cmd = ['ffmpeg','-nostdin','-v','error','-y','-threads','1','-filter_threads','1','-i',encoded]
        audio_source = plan['music'] or (str(path) if info['audio'] else None)
        if audio_source:
            if plan['music']:
                cmd += ['-stream_loop','-1']
            cmd += ['-threads','1','-i',audio_source,'-map','0:v:0','-map','1:a:0','-c:v','copy']
            codec = info['audio']['codec_name'] if info['audio'] else None
            compatible = codec in {'aac','mp3','ac3','eac3','alac'} and not plan['music']
            audio_args = ['-c:a','copy'] if compatible else ['-c:a','aac','-b:a','192k','-threads','1']
            tail = ['-t',f'{duration:.10f}','-map_metadata','-1','-movflags','+faststart',muxed]
            try:
                try:
                    run(cmd+audio_args+tail,timeout)
                    plan['audio_encoding'] = 'copy' if compatible else 'aac'
                except RuntimeError:
                    if not compatible:
                        raise
                    muxed.unlink(missing_ok=True)
                    run(cmd+['-c:a','aac','-b:a','192k','-threads','1']+tail,timeout)
                    plan['audio_encoding'] = 'aac_fallback'
                    plan['warnings'].append('音轨copy失败转AAC；视频未重编码。')
            except RuntimeError as audio_exc:
                if not plan.get('delivery_recovery'):
                    raise
                muxed.unlink(missing_ok=True)
                encoded.replace(muxed)
                audio_source=None
                plan['audio_encoding']='none'
                plan['warnings'].append(f'音轨复制与转码均不可用，已交付无声视频：{audio_exc}')
        else:
            encoded.replace(muxed)
            plan['audio_encoding'] = 'none'
        timings['mux'] = time.perf_counter()-t
        t = time.perf_counter()
        verify = probe(muxed)
        v = next(s for s in verify['streams'] if s['codec_type']=='video')
        a = [s for s in verify['streams'] if s['codec_type']=='audio']
        if (int(v['width']),int(v['height']))!=(W,H):
            raise RuntimeError('ffprobe核验：输出尺寸不一致')
        vf = int(v.get('nb_frames',n))
        vd = float(v.get('duration') or verify['format']['duration'])
        if vf!=n or abs(vd-duration)>1/fps+.025:
            raise RuntimeError('ffprobe核验：输出帧数/时长不一致')
        if bool(a)!=bool(audio_source):
            raise RuntimeError('ffprobe核验：音轨缺失或意外存在')
        plan['verification'] = dict(width=W,height=H,frame_count=vf,video_duration=vd,
                                    format_duration=float(verify['format']['duration']),fps=rate(v.get('avg_frame_rate')),
                                    video_codec=v['codec_name'],audio_codecs=[s['codec_name'] for s in a])
        timings['verify'] = time.perf_counter()-t
        muxed.replace(target)
        for shot in plan['shots']:
            for d in shot['dogs']:
                d.pop('_reference',None)
                d.pop('_mask',None)
                d.pop('_occlusion_state',None)
        return timings
    finally:
        for shot in plan['shots']:
            for d in shot.get('dogs',[]):
                d.pop('_reference',None)
                d.pop('_mask',None)
                d.pop('_occlusion_state',None)
        try:
            if frames is not None and hasattr(frames,'close'):
                frames.close()
        finally:
            try:
                if encoder is not None:
                    encoder.close()
            finally:
                encoded.unlink(missing_ok=True)
                muxed.unlink(missing_ok=True)


def batch(args,assets=None,cache=None):
    root = args.output_dir.expanduser().resolve()
    protected = {Path(p).expanduser().resolve() for p in args.inputs}
    if args.music:
        protected.add(args.music.expanduser().resolve())
    if (root/'summary.json').resolve() in protected:
        raise ValueError('output-dir/summary.json与输入/音乐路径冲突')
    root.mkdir(parents=True,exist_ok=True)
    task_id = 'task_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:12]
    task = root/task_id
    task.mkdir(exist_ok=False)
    assets = Path(assets) if assets is not None else SKILL_ROOT/'assets'
    cache = Path(cache) if cache is not None else SKILL_ROOT/'assets'/'temp_data'/'cache'
    started = time.perf_counter()
    summary = dict(schema_version=4,engine_version=ENGINE_VERSION,task_id=task_id,task_dir=str(task),
                   music_suggestion=MUSIC_SUGGESTION,
                   summary_path=str(task/'summary.json'),status='running',seed=args.seed,plan_only=args.plan_only,
                   worker_count=1,max_inputs=64,results=[],counts={},timings={})
    save_json(task/'summary.json',summary)
    for i,name in enumerate(args.inputs):
        item_start = time.perf_counter()
        path = Path(name).expanduser().resolve()
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in path.stem)[:64] or 'input'
        stem = f'{i+1:03d}_{safe}_{hashlib.sha256(str(path).encode()).hexdigest()[:10]}'
        target,plan_path = task/(stem+'.mp4'),task/(stem+'.json')
        plan = dict(schema_version=4,engine_version=ENGINE_VERSION,input=str(path),output=str(target),
                    music_suggestion=MUSIC_SUGGESTION,
                    status='failed',shots=[],timings={},warnings=list(BASE_WARNINGS),plan_path=str(plan_path))
        info,analysis,library,digest = None,None,None,None
        try:
            if not path.is_file():
                raise ValueError('输入文件不存在或不是普通文件')
            t = time.perf_counter()
            digest = sha256(path)
            plan['timings']['hash'] = time.perf_counter()-t
            t = time.perf_counter()
            info = media_info(path,args.image_duration,args.quality)
            plan['timings']['probe'] = time.perf_counter()-t
            library = DogLibrary(assets)
            t = time.perf_counter()
            analysis,hit = analyze(path,info,digest,cache,args.cut_values,args.timeout)
            plan['timings']['analysis'] = time.perf_counter()-t
            t = time.perf_counter()
            timings = plan['timings']
            plan = create_plan(path,target,info,analysis,library,args,digest)
            plan.update(plan_path=str(plan_path),input_sha256=digest,analysis_cache_hit=hit)
            plan['timings'] = timings
            plan['timings']['planning_assets'] = time.perf_counter()-t
            t=time.perf_counter()
            if any(d['placement'].get('delivery_sticker') for s in plan['shots'] for d in s['dogs']):
                make_sticker_plan(plan,library,'placement_recovery')
            else:
                prepare_scene(path,plan,args)
            plan['tracking_analysis_seconds']=time.perf_counter()-t
            plan['timings']['scene_preflight']=plan['tracking_analysis_seconds']
            if args.music and not any(s.get('codec_type')=='audio' for s in probe(args.music)['streams']):
                raise ValueError('--music没有可用音轨')
            save_json(plan_path,plan)
            if not args.plan_only:
                plan['timings'].update(render_video(path,target,plan,library,args.timeout))
                plan['status'] = 'success'
        except Exception as exc:
            if hasattr(exc,'plan'):
                timings = plan['timings']
                plan = exc.plan
                plan['timings'] = timings
                plan['plan_path'] = str(plan_path)
            # Retry once through a detection-free, top-layer compositor.
            # Keep the real source and audio; never label an absent output success.
            cause = dict(type=type(exc).__name__,message=str(exc))
            try:
                if info is None or library is None:
                    raise RuntimeError('尚无可解码媒体或可用素材，不能生成真实合成结果')
                recovery = copy(args)
                recovery._delivery_recovery = True
                recovery.tracking,recovery.max_dogs = 'off',1
                recovery.scene_plan,recovery.dog_id,recovery.dog_sequence = None,None,None
                recovery.music = None
                if not plan.get('media') or not plan.get('shots') or not all(s.get('dogs') for s in plan['shots']):
                    if analysis is None:
                        if info['kind']=='image':
                            frame=image_rgb(path)
                        else:
                            stream=video_frames(path,info,*small_size(info),1,args.timeout)
                            try:
                                frame=next(stream)
                            finally:
                                stream.close()
                        png=encode_image(frame)
                        points=[0.]+sorted(c for c in (args.cut_values or []) if 0<c<info['duration'])
                        analysis=dict(shots=[dict(start=t,representative_png=png,temporal_samples=[]) for t in points],
                                      limitations=['分析异常，备用计划不使用支撑、人物或运动检测。'])
                    plan=create_plan(path,target,info,analysis,library,recovery,digest)
                plan.update(plan_path=str(plan_path),music=None,recovered_from=cause)
                if args.music:
                    plan['warnings'].append('交付恢复未使用替换配乐，优先保留原音轨。')
                make_sticker_plan(plan,library,f'{cause["type"]}: {cause["message"]}')
                if not args.plan_only:
                    plan['timings'].update(render_video(path,target,plan,library,args.timeout))
                    plan['status']='success'
            except Exception as recovery_exc:
                plan['status']='failed'
                plan['error']=dict(type=type(recovery_exc).__name__,message=str(recovery_exc),
                                   original_cause=cause)
                target.unlink(missing_ok=True)
        finally:
            for shot in plan.get('shots',[]):
                for dog in shot.get('dogs',[]):
                    dog.pop('_reference',None)
                    dog.pop('_mask',None)
                    dog.pop('_occlusion_state',None)
            plan['timings']['total'] = time.perf_counter()-item_start
            save_json(plan_path,plan)
        summary['results'].append(dict(input=str(path),output=str(target),plan_path=str(plan_path),
                                        status=plan['status'],timings=plan['timings'],error=plan.get('error'),
                                        delivery_recovery=plan.get('delivery_recovery',False),
                                        recovered_from=plan.get('recovered_from'),
                                        scene_fallback_count=plan.get('scene_fallback_count',0),
                                        placement_fallback_count=plan.get('placement_fallback_count',0),
                                        overlap_fallback_count=plan.get('overlap_fallback_count',0)))
        success = sum(r['status'] in ('success','planned') for r in summary['results'])
        summary['counts'] = dict(total=len(summary['results']),success=success,failed=len(summary['results'])-success,
                                 requested=len(args.inputs))
        save_json(task/'summary.json',summary)
        print(json.dumps({'status':plan['status'],'input':str(path),'output':str(target),
                          'plan':str(plan_path),'music_suggestion':MUSIC_SUGGESTION,
                          'error':plan.get('error')},ensure_ascii=False),flush=True)
    summary['status'] = 'success' if summary['counts']['failed']==0 else 'partial_failure' if summary['counts']['success'] else 'failed'
    summary['timings']['total'] = time.perf_counter()-started
    save_json(task/'summary.json',summary)
    save_json(root/'summary.json',summary)
    print(json.dumps({'summary':str(root/'summary.json'),'task_summary':str(task/'summary.json'),
                      'status':summary['status'],'counts':summary['counts']},ensure_ascii=False),flush=True)
    return (0 if summary['counts']['failed']==0 else 1),summary


def main():
    p = parser()
    args = p.parse_args()
    validate_args(args,p)
    try:
        code,_ = batch(args)
        return code
    except (OSError,ValueError,RuntimeError) as exc:
        print(json.dumps({'status':'fatal','error':str(exc)},ensure_ascii=False),file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
