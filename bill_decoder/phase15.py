"""Repeatable Phase 1.5 experiment. Official benchmark is evaluation only."""
from dataclasses import asdict,replace
from hashlib import sha256
from pathlib import Path
from copy import deepcopy
import logging
from .api import Gemini,APIProblem,event
from .config import MODEL,FREE_MODELS
from .engine import extract,atomic_write,ExtractionProblem
from .images import ImageOptions,load_image
from .schema import Bill,Extraction,dumps,parse_json,consistency_flags
from .preprocessing import PREPROCESS_VERSION,PreprocessOptions,candidates,crop_candidate
from .ocr import OCR
from .verification import VERIFY_VERSION,collect_support,suspicious_fields,recheck,crop_regions
from .benchmark import compare


def save_json(path,value): atomic_write(path,dumps(value))


def run(folder,output,settings,models=FREE_MODELS,options=None,verify=True,refine_ocr=False):
    folder,output=Path(folder),Path(output)
    if output.resolve()==folder.resolve() or output.resolve().is_relative_to(folder.resolve()):
        raise ValueError('Output must be outside source image directory')
    if not models or any(model not in FREE_MODELS for model in models): raise ValueError('Unapproved model')
    if MODEL not in models: raise ValueError('Working 3.1 baseline must be included')
    options=options or PreprocessOptions()
    paths=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    if not paths or len({p.stem for p in paths})!=len(paths): raise ValueError('Missing images or duplicate stems')
    before={p.name:sha256(p.read_bytes()).hexdigest() for p in paths}
    ocr=OCR(); api_clients={}; baseline={}; results={}; baseline_hits=0
    summary={'free_tier_basis':'Caller explicitly confirmed Free tier; no billing changes',
             'models':list(models),'refine_ocr':refine_ocr,'preprocessing':asdict(options),'bills':results,'baseline_cache_hits':0,'verification_cache_hits':0,'ocr_cache_hits':0,'api_calls':{},'ocr':ocr.info}
    def checkpoint():
        summary['api_calls']={model:api.calls for model,api in api_clients.items()}
        summary['baseline_cache_hits']=baseline_hits
        save_json(output/'experiment.json',summary)
    try:
        for model in models:
            api=Gemini(replace(settings,model=model)); api_clients[model]=api
            permanent=None
            for path in paths:
                result=results.setdefault(path.stem,{'baselines':{}})
                try:
                    if permanent: raise APIProblem(*permanent)
                    calls=api.calls
                    envelope=extract(path,api,output/'cache/baseline',ImageOptions())
                    extraction=Extraction.model_validate(envelope['extraction'])
                    baseline[(path.stem,model)]=extraction
                    destination=output/'baseline'/model/(path.stem+'.json')
                    save_json(destination,extraction.bill)
                    save_json(destination.with_suffix('.evidence.json'),envelope)
                    cache_hit=api.calls==calls
                    baseline_hits+=int(cache_hit)
                    result['baselines'][model]={'status':'ok','cache_hit':cache_hit,'api_calls':api.calls-calls,'flags':envelope['flags']}
                    event('phase15_baseline',bill_id=path.stem,model=model,api_calls=api.calls-calls,cache_hit=cache_hit)
                except (APIProblem,ExtractionProblem,OSError,ValueError) as exc:
                    if isinstance(exc,APIProblem):
                        detail={'code':exc.code,'error':str(exc)}
                        if exc.code in (400,401,403,404): permanent=(exc.code,str(exc))
                    else: detail={'code':type(exc).__name__,'error':'Baseline failed; no fabricated replacement'}
                    result['baselines'][model]={'status':'failed',**detail}
                checkpoint()
        for path in paths:
            result=results[path.stem]
            if (path.stem,MODEL) not in baseline: continue
            original=baseline[(path.stem,MODEL)]
            other=baseline.get((path.stem,'gemini-3.5-flash-lite'))
            image=load_image(path,ImageOptions())
            key=sha256(dumps({'source':before[path.name],'ocr':{k:v for k,v in ocr.info.items() if k not in ('calls','cache_hits')},
                'options':asdict(options),'preprocessing_version':PREPROCESS_VERSION,'verification_version':VERIFY_VERSION,
                'baseline':original,**({'refine_ocr':True} if refine_ocr else {})}).encode()).hexdigest()
            local_path=output/'cache/ocr_analysis'/(key+'.json')
            local=None
            if local_path.exists():
                try: local=parse_json(local_path.read_text())
                except (ValueError,OSError): local=None
            if local is None:
                views,diagnostics=candidates(image,options)
                if refine_ocr:
                    for box in crop_regions(original.bill.provider,['charges','taxes'],[]):
                        views.append(crop_candidate(image,box,options.scale))
                support=[]; candidate_results=[]; net_metering=False
                for candidate in views:
                    tokens,metrics=ocr.recognize(candidate,image.size)
                    net_metering|=any('NET' in t.text.upper() for t in tokens) and any('METERING' in t.text.upper() for t in tokens)
                    matched=collect_support(original,tokens)
                    for entry in matched: entry['candidate']=candidate.name
                    support.extend(matched)
                    candidate_results.append({'name':candidate.name,'size':list(candidate.image.size),'transform':candidate.transform.tolist(),
                         'preprocessing':candidate.metadata,'ocr':metrics,'field_support':matched})
                local={'candidates':candidate_results,'diagnostics':diagnostics,'support':support,'net_metering':net_metering}
                save_json(local_path,local)
            else:
                summary['ocr_cache_hits']+=1; ocr.info['cache_hits']+=1
            save_json(output/'ocr'/(path.stem+'.json'),local)
            result['ocr_candidates']=[{'name':c['name'],'ocr':c['ocr']} for c in local['candidates']]
            fields=suspicious_fields(original,local['support'],other,local['net_metering'])
            result['suspicious_fields']=fields
            if verify:
                try:
                    enhanced,verification=recheck(image,original,fields,local['support'],api_clients[MODEL],output/'cache/verification',before[path.name],options.scale)
                except (APIProblem,ValueError,ExtractionProblem) as exc:
                    enhanced=original
                    verification={'status':'failed_baseline_retained','code':getattr(exc,'code',type(exc).__name__),
                                  'error':str(exc) if isinstance(exc,APIProblem) else 'Verification failed validation','decisions':[]}
            else:
                enhanced,verification=original,{'status':'local_only','decisions':[]}
            summary['verification_cache_hits']+=int(verification.get('cache_hit',False))
            result['verification']=verification
            result['baseline_vs_enhanced']=compare(enhanced.bill.model_dump(),original.bill.model_dump())
            result['model_disagreements']=[] if other is None else [item for item in compare(other.bill.model_dump(),original.bill.model_dump())['fields'] if not item['match']]
            result['enhanced_flags']=consistency_flags(enhanced.bill)
            save_json(output/'enhanced'/(path.stem+'.json'),enhanced.bill)
            save_json(output/'enhanced'/(path.stem+'.evidence.json'),{'extraction':enhanced,'image_sha256':before[path.name],'model':MODEL,'verification':verification,'flags':result['enhanced_flags']})
            event('phase15_enhanced',bill_id=path.stem,status=verification['status'],changed_fields=18-result['baseline_vs_enhanced']['exact_fields'])
            checkpoint()
        # Evaluation is isolated from generation/acceptance logic and runs only afterwards.
        gold_path=Path('tests/fixtures/KESC_0008.gold.json')
        if gold_path.exists():
            gold=parse_json(gold_path.read_text(encoding='utf-8-sig'))
            benchmarks={}
            for model in models:
                sample=baseline.get(('KESC_0008',model))
                if sample: benchmarks[model]=compare(sample.bill.model_dump(),gold)
            enhanced_path=output/'enhanced/KESC_0008.json'
            if enhanced_path.exists(): benchmarks['enhanced_3.1']=compare(parse_json(enhanced_path.read_text()),gold)
            summary['official_benchmark']=benchmarks
            save_json(output/'gold_comparisons.json',benchmarks)
        after={p.name:sha256(p.read_bytes()).hexdigest() for p in paths}
        summary['originals_unchanged']=before==after
        if before!=after: raise RuntimeError('Original image changed')
        summary['baseline_model_retained']=MODEL
        summary['model_selection_reason']='No second-model targeting enabled; accuracy superiority must be demonstrated independently'
        checkpoint()
        return summary
    finally:
        for api in api_clients.values(): api.close()
