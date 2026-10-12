"""Conservative optional candidates, with explicit coordinate transforms and gates."""
from dataclasses import dataclass, asdict
import math
import cv2
import numpy as np
from PIL import Image
from .images import region

PREPROCESS_VERSION='1'

@dataclass(frozen=True)
class PreprocessOptions:
    clahe: bool = True
    denoise: bool = True
    deskew: bool = True
    perspective: bool = False
    scale: int = 2

    def __post_init__(self):
        if self.scale not in (1,2,3):
            raise ValueError('Crop scale must be 1, 2 or 3')

@dataclass
class Candidate:
    name: str
    image: Image.Image
    # Original/full oriented pixel coordinates -> candidate pixel coordinates.
    transform: np.ndarray
    metadata: dict


def gray_clahe(image):
    gray=cv2.cvtColor(np.array(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(gray))


def estimate_skew(image):
    gray=np.array(image.convert('L'))
    lines=cv2.HoughLinesP(cv2.Canny(gray,60,180),1,np.pi/180,
                         threshold=max(30,image.width//10),minLineLength=max(40,image.width//5),maxLineGap=15)
    angles=[]
    if lines is not None:
        for x0,y0,x1,y1 in np.asarray(lines).reshape(-1,4):
            if x1 == x0: continue
            angle=math.degrees(math.atan2(y1-y0,x1-x0))
            while angle>90: angle-=180
            while angle<-90: angle+=180
            if abs(angle)<=7: angles.append(angle)
    angle=float(np.median(angles)) if angles else 0.0
    spread=float(np.median(np.abs(np.array(angles)-angle))) if angles else 999.0
    reliable=len(angles)>=6 and spread<=.5 and .3<=abs(angle)<=5
    return {'angle_degrees':angle,'line_count':len(angles),'median_deviation':spread,'applied':reliable}


def deskew(image):
    meta=estimate_skew(image)
    if not meta['applied']: return None,meta
    w,h=image.size
    matrix=cv2.getRotationMatrix2D((w/2,h/2),meta['angle_degrees'],1)
    cosine,sine=abs(matrix[0,0]),abs(matrix[0,1])
    width,height=int(math.ceil(h*sine+w*cosine)),int(math.ceil(h*cosine+w*sine))
    matrix[0,2]+=(width-w)/2; matrix[1,2]+=(height-h)/2
    warped=cv2.warpAffine(np.array(image),matrix,(width,height),flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    return Candidate('deskew',Image.fromarray(warped),np.vstack([matrix,[0,0,1]]),meta),meta


def order_quad(points):
    points=np.asarray(points,dtype=np.float32)
    sums=points.sum(axis=1); differences=np.diff(points,axis=1).ravel()
    return np.array([points[np.argmin(sums)],points[np.argmin(differences)],points[np.argmax(sums)],points[np.argmax(differences)]],dtype=np.float32)


def rectify_perspective(image,quad=None):
    """Reject incomplete/ambiguous page boundaries; never rectify a small bill panel."""
    w,h=image.size
    if quad is None:
        gray=np.array(image.convert('L'))
        contours,_=cv2.findContours(cv2.Canny(gray,60,180),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        proposals=[]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:8]:
            approximation=cv2.approxPolyDP(contour,.015*cv2.arcLength(contour,True),True)
            if len(approximation)==4 and cv2.isContourConvex(approximation) and cv2.contourArea(approximation)/(w*h)>=.80:
                proposals.append(np.asarray(approximation).reshape(-1,2))
        if len(proposals)!=1:
            return None,{'applied':False,'reason':'no unique convex near-full-page boundary'}
        quad=proposals[0]
    source=order_quad(quad)
    area=abs(cv2.contourArea(source))/(w*h)
    if area<.80 or len(np.unique(source,axis=0))!=4:
        return None,{'applied':False,'reason':'boundary does not cover enough of the page'}
    # Every corner must be in its respective corner zone to exclude internal tables.
    if not (source[0,0]<.18*w and source[0,1]<.18*h and source[1,0]>.82*w and source[1,1]<.18*h and source[2,0]>.82*w and source[2,1]>.82*h and source[3,0]<.18*w and source[3,1]>.82*h):
        return None,{'applied':False,'reason':'corners do not support a complete page'}
    top=np.linalg.norm(source[1]-source[0]); bottom=np.linalg.norm(source[2]-source[3])
    left=np.linalg.norm(source[3]-source[0]); right=np.linalg.norm(source[2]-source[1])
    if max(top,bottom)/min(top,bottom)>1.2 or max(left,right)/min(left,right)>1.2:
        return None,{'applied':False,'reason':'projective distortion too large'}
    width,height=round(max(top,bottom)),round(max(left,right))
    target=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype=np.float32)
    transform=cv2.getPerspectiveTransform(source,target)
    warped=cv2.warpPerspective(np.array(image),transform,(width,height),flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    meta={'applied':True,'page_area_fraction':area,'source_corners':source.tolist()}
    return Candidate('perspective',Image.fromarray(warped),transform,meta),meta


def candidates(image,options):
    identity=np.eye(3)
    result=[Candidate('original',image,identity,{'applied':False})]
    diagnostics={}
    if options.clahe: result.append(Candidate('gray_clahe',gray_clahe(image),identity,{'clip_limit':1.5}))
    if options.denoise:
        reduced=cv2.bilateralFilter(np.array(image),5,25,25)
        result.append(Candidate('denoise',Image.fromarray(reduced),identity,{'diameter':5,'sigma':25}))
    if options.deskew:
        candidate,diagnostics['deskew']=deskew(image)
        if candidate: result.append(candidate)
    if options.perspective:
        candidate,diagnostics['perspective']=rectify_perspective(image)
        if candidate: result.append(candidate)
    return result,diagnostics


def high_resolution_crop(image,box,scale=2):
    cropped=region(image,box)
    if min(cropped.size)<2: raise ValueError('Crop is empty or too small')
    return cropped.resize((cropped.width*scale,cropped.height*scale),Image.Resampling.LANCZOS)


def crop_candidate(image,box,scale=2):
    """High-resolution OCR region with an exact crop/scale inverse transform."""
    pixels=(int(box[0]*image.width),int(box[1]*image.height))
    transform=np.array([[scale,0,-scale*pixels[0]],[0,scale,-scale*pixels[1]],[0,0,1]],dtype=float)
    return Candidate('financial_crop',high_resolution_crop(image,box,scale),transform,{'region':list(box),'scale':scale})


def original_bbox(box,transform,original_size):
    """Map all four OCR box corners back through affine/projective candidates."""
    x0,y0,x1,y1=box
    points=np.array([[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]],dtype=np.float32)
    original=cv2.perspectiveTransform(points,np.linalg.inv(transform))[0]
    w,h=original_size
    return [max(0,min(1,float(original[:,0].min()/w))),max(0,min(1,float(original[:,1].min()/h))),
            max(0,min(1,float(original[:,0].max()/w))),max(0,min(1,float(original[:,1].max()/h)))]
