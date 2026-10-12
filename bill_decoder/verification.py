"""OCR disagreement signals and crop-only Gemini rechecks. Never copy OCR into bills."""
from copy import deepcopy
from decimal import Decimal,InvalidOperation
from datetime import date
from hashlib import sha256
import re
from pathlib import Path
from pydantic import ValidationError
from google.genai import types
from .engine import atomic_write,ExtractionProblem,PROMPT
from .schema import Bill,Extraction,Observation,api_schema,dumps,parse_json,consistency_flags
from .ocr import lines
from .images import png_bytes
from .preprocessing import high_resolution_crop,gray_clahe

VERIFY_VERSION='1'
PRIVATE_LABEL=re.compile(r'\b(name|address|cnic|account|reference|consumer id|meter no|invoice no|contract no|feeder|transformer)\b',re.I)
NUMERIC_FIELDS={'sanctioned_load_kw','previous_reading','current_reading','units_consumed','total_charges','total_taxes','current_bill','arrears','payable_within_due_date','payable_after_due_date'}


def printed_number(text):
    if not isinstance(text,str): return None
    text=re.sub(r'(?i)\b(?:PKR|RS)\.?','',text).strip().replace(',','').replace(' ','')
    credit=text.upper().endswith('CR')
    if credit: text=text[:-2]
    if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?',text): return None
    value=Decimal(text)
    return -abs(value) if credit else value


MONTHS={name:index+1 for index,name in enumerate(('JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'))}
DATE_FIELDS={'bill_month','reading_date','issue_date','due_date'}

def printed_date(text,month_only=False):
    if not isinstance(text,str): return None
    text=re.sub(r'(?<=\d)(?:st|nd|rd|th)\b','',text,flags=re.I)
    clean=re.sub(r'[-/,]',' ',text.upper()).strip()
    if month_only:
        found=re.search(r'\b('+'|'.join(MONTHS)+r')\s*(\d{4}|\d{2})\b',clean)
        if not found: return text if re.fullmatch(r'\d{4}-\d{2}',text) else None
        month,year=found.groups(); year=int(year); year=2000+year if year<100 else year
        return f'{year:04d}-{MONTHS[month]:02d}'
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}',text):
        try: return date.fromisoformat(text).isoformat()
        except ValueError: return None
    found=re.search(r'\b(\d{1,2})\s*('+'|'.join(MONTHS)+r')\s*(\d{4}|\d{2})\b',clean)
    if not found: return None
    day,month,year=found.groups(); year=int(year); year=2000+year if year<100 else year
    try: return date(year,MONTHS[month],int(day)).isoformat()
    except ValueError: return None


def field_value(bill,field):
    match=re.fullmatch(r'(charges|taxes)\[(\d+)\]\.amount',field)
    if match:
        items=getattr(bill,match.group(1)); index=int(match.group(2))
        return items[index].amount if index<len(items) else None
    return getattr(bill,field)


def collect_support(extraction,tokens):
    """Anchor OCR rows by model-provided billing labels; preserve uncertainty.

    These are candidate numbers from a row, not confirmed field assignments.
    Only billing numeric candidates are retained, never whole OCR text or IDs.
    """
    support=[]
    grouped=lines(tokens)
    for observation in extraction.evidence:
        field=observation.field
        if field not in NUMERIC_FIELDS | DATE_FIELDS and not re.fullmatch(r'(charges|taxes)\[\d+\]\.amount',field): continue
        if PRIVATE_LABEL.search(observation.printed_label): continue
        words=re.findall(r'[a-z]+',observation.printed_label.lower())
        words=[w for w in words if len(w)>1 and w not in ('rs','pkr','the','in','of','for','and','to')]
        if not words: continue
        choices=[]
        for group in grouped:
            text=' '.join(t.text for t in group)
            if PRIVATE_LABEL.search(text): continue
            seen=set(re.findall(r'[a-z]+',text.lower()))
            score=sum(w in seen for w in set(words))/len(set(words))
            if score < .7: continue
            y0=min(t.bbox[1] for t in group); y1=max(t.bbox[3] for t in group)
            x0=min(t.bbox[0] for t in group)
            row=[t for t in tokens if abs((t.bbox[1]+t.bbox[3])/2-(y0+y1)/2)<max(.008,(y1-y0)*.7) and t.bbox[0]>=x0]
            numbers=[]
            for token in row:
                value=printed_number(token.text)
                if value is not None and token.confidence>=55:
                    numbers.append({'value':value,'confidence':token.confidence,'bbox':token.bbox})
            if field in DATE_FIELDS:
                normalized=printed_date(' '.join(t.text for t in row),field=='bill_month')
                if normalized:
                    numbers=[{'value':normalized,'confidence':min(t.confidence for t in row),'bbox':[x0,y0,1,y1]}]
            if numbers:
                choices.append((score,numbers,[max(0,x0-.02),max(0,y0-.02),1,min(1,y1+.02)]))
        if not choices: continue
        _,numbers,box=max(choices,key=lambda c:c[0])
        expected=field_value(extraction.bill,field)
        expected_raw=printed_number(observation.printed_value)
        # A subsidy may have a positive printed magnitude while canonical value is negative.
        equivalents={expected,abs(expected)} if isinstance(expected,Decimal) and expected<0 else {expected}
        numbers=[n for n in numbers if isinstance(n['value'],str) or abs(n['value'])<Decimal('1000000000')]
        agrees=any(n['value'] in equivalents for n in numbers)
        distinct={n['value'] for n in numbers if n['confidence']>=75}
        status='agrees' if agrees else 'disagrees' if len(distinct)==1 else 'ambiguous'
        support.append({'field':field,'status':status,'baseline':expected,'candidates':numbers,'region':box,'printed_numeric_value':expected_raw})
    return support


def suspicious_fields(baseline,support,other=None,net_metering=False):
    fields={}
    for item in support:
        if item['status']=='disagrees':
            top=item['field'].split('[')[0]
            fields.setdefault(top,[]).append('OCR row disagrees; association is uncertain')
    if other:
        for field in Bill.model_fields:
            if baseline.bill.model_dump()[field] != other.bill.model_dump()[field]:
                fields.setdefault(field,[]).append('Approved-model baseline disagreement')
    for flag in consistency_flags(baseline.bill):
        field=flag.split(':')[0]
        fields.setdefault(field,[]).append(flag)
        if field=='total_charges': fields.setdefault('charges',[]).append('Review printed subtotal and separately printed adjustments')
    if net_metering:
        for field in ('previous_reading','current_reading','units_consumed'):
            fields.setdefault(field,[]).append('Multiple net-metering registers require explicit printed labels or remain ambiguous')
    for observation in baseline.evidence:
        if observation.status in ('ambiguous','unreadable'):
            fields.setdefault(observation.field.split('[')[0],[]).append('Baseline evidence '+observation.status)
    return fields


def crop_regions(provider,fields,support):
    boxes=[]
    def add(box):
        box=tuple(float(x) for x in box)
        if box not in boxes: boxes.append(box)
    for field in fields:
        for item in support:
            if item['field'].split('[')[0]==field: add(item['region'])
    fields=set(fields)
    if fields & {'previous_reading','current_reading','units_consumed'}:
        add((.02,.28,.72,.56) if provider=='LESCO' else (.02,.20,.70,.33) if provider=='IESCO' else (.01,.30,.52,.44))
    if fields & {'charges','taxes','total_charges','total_taxes','current_bill','arrears'}:
        add((.02,.38,.76,.75) if provider=='LESCO' else (.02,.33,.69,.55) if provider=='IESCO' else (.51,.30,.99,.66))
    if fields & {'reading_date','issue_date','due_date','bill_month','tariff','sanctioned_load_kw','provider'}:
        add((.02,.02,.99,.38))
    if fields & {'payable_within_due_date','payable_after_due_date'}:
        add((.69,.22,.99,.51) if provider in ('LESCO','IESCO') else (.01,.16,.99,.30))
    if not boxes: add((0,0,1,1))
    # Bounded image count; broad provider context comes last and takes priority.
    return boxes[-5:]


def patch_schema(fields):
    source=api_schema()
    properties=source['$defs']['Bill']['properties']
    return {'type':'object','additionalProperties':False,'properties':{
        'fields':{'type':'object','additionalProperties':False,'properties':{k:properties[k] for k in fields},'required':list(fields)},
        'support':{'type':'array','items':{'$ref':'#/$defs/Observation'}}},
        'required':['fields','support'],'$defs':source['$defs']}


def supported_change(field,value,observations):
    """A changed value requires fresh image-read evidence, not an OCR vote."""
    if field in ('charges','taxes'):
        if value==[]:
            return any(o.field==field and o.status in ('visible','missing','ambiguous') for o in observations)
        for index,item in enumerate(value):
            evidence=[o for o in observations if o.field==f'{field}[{index}].amount' and o.status=='visible']
            if not any(printed_number(o.printed_value) is not None and (printed_number(o.printed_value)==item['amount'] or (field=='charges' and item['type']=='subsidy' and -abs(printed_number(o.printed_value))==item['amount'])) for o in evidence): return False
        return True
    evidence=[o for o in observations if o.field==field]
    if value is None: return any(o.status in ('missing','unreadable','ambiguous') for o in evidence)
    if field in NUMERIC_FIELDS:
        return any(o.status=='visible' and printed_number(o.printed_value)==value for o in evidence)
    if field in DATE_FIELDS:
        return any(o.status=='visible' and printed_date(o.printed_value,field=='bill_month')==value for o in evidence)
    return any(o.status=='visible' and o.printed_value==value for o in evidence)


def recheck(image,baseline,fields,support,api,cache_dir,source_hash,scale=2):
    if not fields:
        return baseline,{'status':'not_needed','cache_hit':False,'api_calls':0,'decisions':[]}
    schema=patch_schema(fields)
    # Send only field-specific safe numeric OCR support, never full-page OCR.
    safe_support=[item for item in support if item['field'].split('[')[0] in fields]
    instructions= PROMPT + '\nTARGETED VERIFICATION ONLY. Re-read ONLY these fields from attached contextual crops: '+dumps(list(fields))+'.\nReasons: '+dumps(fields)+'.\nBaseline candidate values: '+dumps({field:baseline.bill.model_dump()[field] for field in fields})+'.\nIndependent OCR candidates (may be wrong or refer to neighboring columns; never override solely on OCR): '+dumps(safe_support)+'.\nReturn fields with ONLY the requested keys and support with fresh printed labels/values and status. Do not calculate net readings or repair printed subtotals. For multiple registers, a scalar must be explicitly printed or remain null. For bbox use null because these are cropped views. If an entire list is empty, include an observation for that list explaining its printed section in printed_label, printed_value=null. Do not include personal identifiers.'
    boxes=crop_regions(baseline.bill.provider,fields,safe_support)
    contents=[instructions]
    for box in boxes:
        crop=high_resolution_crop(image,box,scale)
        contents.extend(['original crop='+dumps(box),types.Part.from_bytes(data=png_bytes(crop),mime_type='image/png')])
        contents.extend(['same crop with gentle grayscale CLAHE',types.Part.from_bytes(data=png_bytes(gray_clahe(crop)),mime_type='image/png')])
    key=sha256(dumps({'version':VERIFY_VERSION,'image':source_hash,'model':api.settings.model,'instructions':instructions,'schema':schema,'boxes':boxes,'scale':scale}).encode()).hexdigest()
    path=Path(cache_dir)/(key+'.json')
    cached=False; calls=api.calls; usage=[]
    def validate_payload(payload):
        if not isinstance(payload,dict) or set(payload)!= {'fields','support'} or set(payload['fields'])!=set(fields):
            raise ValueError('Targeted response must contain exactly the requested field keys')
        observations=[Observation.model_validate(item) for item in payload['support']]
        if any(o.field.split('[')[0] not in fields for o in observations): raise ValueError('Unrequested evidence')
        merged=baseline.bill.model_dump(); merged.update(payload['fields'])
        bill=Bill.model_validate(merged)
        retained=[o for o in baseline.evidence if o.field.split('[')[0] not in fields]
        extraction=Extraction.model_validate({'bill':bill,'evidence':retained+observations})
        return extraction,observations
    if path.exists():
        try:
            payload=parse_json(path.read_text())
            proposal,observations=validate_payload(payload)
            cached=True
        except (ValueError,ValidationError,KeyError,OSError): cached=False
    if not cached:
        for attempt in range(2):
            text,call_usage=api.generate(contents,schema); usage.append(call_usage)
            try:
                payload=parse_json(text); proposal,observations=validate_payload(payload)
                atomic_write(path,dumps(payload)); break
            except (ValueError,ValidationError,KeyError) as exc:
                if attempt==1:
                    return baseline,{'status':'validation_failed_baseline_retained','api_calls':api.calls-calls,'cache_hit':False,'decisions':[],'usage':usage}
                contents.append('Previous response failed validation. Return exactly requested keys, valid official types and billing-only support. Do not change unrelated fields.')
    result=deepcopy(baseline)
    data=result.bill.model_dump(); decisions=[]
    for field in fields:
        value=proposal.bill.model_dump()[field]
        changed=value!=data[field]
        accepted=not changed or supported_change(field,value,observations)
        decisions.append({'field':field,'baseline':data[field],'proposed':value,'changed':changed,'accepted':accepted,
                          'reason':'fresh crop evidence supplied' if changed and accepted else 'insufficient evidence; baseline retained' if changed else 're-read unchanged'})
        if accepted:
            data[field]=value
            result.evidence=[o for o in result.evidence if o.field.split('[')[0]!=field]+[o for o in observations if o.field.split('[')[0]==field]
    result=Extraction.model_validate({'bill':data,'evidence':result.evidence})
    return result,{'status':'completed','api_calls':api.calls-calls,'cache_hit':cached,'fields':fields,'regions':boxes,'decisions':decisions,'usage':usage}
