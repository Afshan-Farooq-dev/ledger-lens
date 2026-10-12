"""Independent local OCR. Full-page tokens stay in memory; never dump raw OCR."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
from statistics import mean
from .preprocessing import original_bbox

DEFAULT_TESSERACT=Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')

@dataclass
class OCRToken:
    text: str
    confidence: float
    bbox: list[float]
    line: tuple

class OCR:
    def __init__(self,executable=None,languages=('eng','urd'),timeout=30):
        self.timeout=timeout
        self.info={'available':False,'languages':[],'selected_languages':[],'calls':0,'cache_hits':0}
        self.module=None
        try:
            import pytesseract
            self.module=pytesseract
            resolved=Path(executable) if executable is not None else DEFAULT_TESSERACT
            if not resolved.is_file():
                fallback=shutil.which('tesseract') if executable is None else None
                if not fallback:
                    self.info['reason']='Tesseract executable not found'
                    return
                resolved=Path(fallback)
            self.module.pytesseract.tesseract_cmd=str(resolved)
            installed=self.module.get_languages(config='')
            selected=[language for language in languages if language in installed]
            self.info.update(executable=str(resolved),version=str(self.module.get_tesseract_version()),languages=installed,selected_languages=selected)
            if not selected:
                self.info['reason']='No requested English/Urdu language pack available'
                return
            self.info['available']=True
        except (ImportError,OSError,RuntimeError) as exc:
            self.info['reason']='OCR unavailable: '+type(exc).__name__

    def recognize(self,candidate,original_size):
        if not self.info['available']: return [],{'status':'unavailable','reason':self.info.get('reason')}
        try:
            self.info['calls']+=1
            data=self.module.image_to_data(candidate.image,lang='+'.join(self.info['selected_languages']),
                config='--psm 11',output_type=self.module.Output.DICT,timeout=self.timeout)
            tokens=[]
            for i,text in enumerate(data['text']):
                if not text.strip(): continue
                conf=float(data['conf'][i])
                if conf<0: continue
                x,y,w,h=(int(data[k][i]) for k in ('left','top','width','height'))
                tokens.append(OCRToken(text.strip(),conf,original_bbox([x,y,x+w,y+h],candidate.transform,original_size),
                         tuple(int(data[k][i]) for k in ('page_num','block_num','par_num','line_num'))))
            return tokens,{'status':'ok','token_count':len(tokens),'mean_confidence':mean(t.confidence for t in tokens) if tokens else None}
        except (OSError,RuntimeError) as exc:
            return [],{'status':'failed','reason':'OCR processing failed: '+type(exc).__name__}


def lines(tokens):
    groups={}
    for token in tokens: groups.setdefault(token.line,[]).append(token)
    return [sorted(group,key=lambda t:t.bbox[0]) for group in groups.values()]
