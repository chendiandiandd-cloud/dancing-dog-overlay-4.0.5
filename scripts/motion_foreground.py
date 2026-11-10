"""Camera-compensated current-frame foreground silhouettes; no external weights."""
import cv2
import numpy as np

from lightweight_vision import person_boxes


def silhouette(frame, seed, rect, foot_y, motion=None):
    """Refine moving components by appearance; never fill a whole body rectangle."""
    H,W=frame.shape[:2]
    groups=cv2.morphologyEx(seed,cv2.MORPH_CLOSE,np.ones((15,9),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(groups)
    result=np.zeros((H,W),np.uint8)
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<20 or h<H*.04 or y+h<foot_y+H*.015:
            continue
        if x>=rect[2] or x+w<=rect[0] or y>=rect[3] or y+h<=rect[1]:
            continue
        if area>H*W*.55:
            continue
        if w/max(1,h)>1.5:
            component_motion=motion[labels==label] if motion is not None else np.array([])
            if not component_motion.size or np.median(component_motion)<2.5:
                continue
        pad=max(8,round(H*.10))
        a,b,c,d=max(0,x-pad),max(0,y-pad),min(W,x+w+pad),min(H,y+h+pad)
        crop=frame[b:d,a:c]
        component=((labels[b:d,a:c]==label)&(seed[b:d,a:c]>0)).astype(np.uint8)
        assignment=np.full(component.shape,cv2.GC_PR_BGD,np.uint8)
        assignment[component>0]=cv2.GC_PR_FGD
        border=np.concatenate([crop[0],crop[-1],crop[:,0],crop[:,-1]])
        background=np.median(border,axis=0)
        distinct=np.linalg.norm(crop.astype(float)-background,axis=2)>35
        interior=(component>0)&distinct
        assignment[interior]=cv2.GC_FGD
        assignment[[0,-1],:]=cv2.GC_BGD
        assignment[:,[0,-1]]=cv2.GC_BGD
        if not np.any(assignment==cv2.GC_FGD):
            continue
        try:
            cv2.setRNGSeed(17)
            cv2.grabCut(crop,assignment,None,np.zeros((1,65),np.float64),
                        np.zeros((1,65),np.float64),1,cv2.GC_INIT_WITH_MASK)
            mask=np.isin(assignment,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
        except cv2.error:
            mask=component*255
        result[b:d,a:c]=np.maximum(result[b:d,a:c],mask)
    return result


class MotionForeground:
    """Backward flow rejects departed-person ghosts; transport visible silhouettes."""
    def __init__(self):
        self.previous=None
        self.matrix=None
        self.mask=None
        self.offset=0

    def update(self, frame, matrix, rect, foot_y):
        H,W=frame.shape[:2]
        seed=np.zeros((H,W),np.uint8)
        motion=None
        if self.previous is not None:
            gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            prev_gray=cv2.cvtColor(self.previous,cv2.COLOR_RGB2GRAY)
            back=cv2.calcOpticalFlowFarneback(gray,prev_gray,None,.5,3,15,3,5,1.2,0)
            yy,xx=np.mgrid[:H,:W].astype(np.float32)
            coordinates=np.stack([xx,yy,np.ones_like(xx)],axis=2)
            inverse=self.matrix @ np.linalg.inv(matrix)
            expected=coordinates @ inverse.T
            residual=np.linalg.norm(back-(expected[:,:,:2]-coordinates[:,:,:2]),axis=2)
            motion=residual
            delta=matrix @ np.linalg.inv(self.matrix)
            registered=cv2.warpAffine(self.previous,delta[:2],(W,H))
            valid=cv2.warpAffine(np.full((H,W),255,np.uint8),delta[:2],(W,H))>0
            difference=frame.astype(np.float32)-registered.astype(np.float32)
            illumination=np.median(difference[valid],axis=0) if valid.any() else np.zeros(3)
            difference=np.max(np.abs(difference-illumination),axis=2)
            seed[(residual>.65)&(difference>12)&valid]=255
            # Appearance can precede measurable flow when a person enters.
            # Current-frame contrast rejects an uncovered flat-background ghost.
            radius=min(51,max(5,(min(H,W)//6)|1))
            median=cv2.medianBlur(frame,radius)
            contrast=np.max(cv2.absdiff(frame,median),axis=2)
            seed[(difference>35)&(contrast>22)&valid]=255
            # Retain stopped/slow silhouettes only when current appearance still
            # matches the previous foreground at backward-flow coordinates.
            mx,my=xx+back[:,:,0],yy+back[:,:,1]
            previous_mask=cv2.remap(self.mask,mx,my,cv2.INTER_NEAREST)
            previous_rgb=cv2.remap(self.previous,mx,my,cv2.INTER_LINEAR)
            match=np.max(np.abs(frame.astype(np.float32)-previous_rgb.astype(np.float32)-illumination),axis=2)<25
            seed[(previous_mask>0)&match&valid]=255
        if self.offset%10==0:
            for x,y,x2,y2 in person_boxes(frame):
                if y2<foot_y+H*.015 or x>=rect[2] or x2<=rect[0]:
                    continue
                a,b,c,d=map(int,(max(0,x),max(0,y),min(W,x2),min(H,y2)))
                if c-a<4 or d-b<4:
                    continue
                # Rectangle mode estimates a silhouette from person evidence.
                labels=np.zeros((H,W),np.uint8)
                try:
                    cv2.setRNGSeed(17)
                    cv2.grabCut(frame,labels,(a,b,c-a,d-b),np.zeros((1,65),np.float64),
                                np.zeros((1,65),np.float64),1,cv2.GC_INIT_WITH_RECT)
                    seed[np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD])]=255
                except cv2.error:
                    pass
        mask=silhouette(frame,seed,rect,foot_y,motion)
        self.previous=frame.copy()
        self.matrix=matrix.copy()
        self.mask=mask
        self.offset+=1
        return mask
