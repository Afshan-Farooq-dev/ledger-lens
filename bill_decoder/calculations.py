"""Grounded statement catalog and deterministic, auditable Decimal calculations."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext, ROUND_HALF_UP
import re
from typing import Literal
from pydantic import Field, field_validator, model_validator
from .schema import StrictModel, Number, Bill
from .evidence import ExtraEvidence

CALC_VERSION='2'
D=Decimal

class Step(StrictModel):
    id: str
    operation: Literal['facts','sum','average','min','max','count','count_above','count_at_least','count_below','count_at_most','difference','percentage','ratio','percent_change','compare','days_between','estimate','increase_percent','decrease_percent','product']
    refs: list[str] = Field(max_length=100)
    series: Literal['none','history_units','history_billed_amount','history_payment']
    period: Literal['previous','including_current','all']
    start_month: str | None
    end_month: str | None
    last_n: Number | None
    threshold: Number | None

    @field_validator('id')
    @classmethod
    def identifier(cls,v):
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,31}',v): raise ValueError('Invalid step ID')
        return v

    @field_validator('start_month','end_month')
    @classmethod
    def month(cls,v): return Bill.month(v)

    @model_validator(mode='after')
    def valid(self):
        if self.last_n is not None and (self.last_n != self.last_n.to_integral_value() or not 1<=self.last_n<=24): raise ValueError('last_n must be an integer 1..24')
        if self.start_month and self.end_month and self.start_month>self.end_month: raise ValueError('Reversed month range')
        if self.series!='none' and self.refs:
            self.refs = []  # Gracefully clear explicit refs if a series is used
        if self.series=='none' and self.refs:
            # Explicit references already identify the requested months. A model
            # sometimes repeats last_n alongside them; it is not an operand.
            self.start_month=None;self.end_month=None;self.last_n=None
        return self

class QuestionPlan(StrictModel):
    question_id: str
    steps: list[Step] = Field(max_length=12)
    limitations: list[Literal['not_printed','ambiguous_registers','external_tariff','unsupported_cause','incomplete_history','ambiguous_question','estimate_only','private_information']]

class Plans(StrictModel):
    questions: list[QuestionPlan]

@dataclass(frozen=True)
class Fact:
    id: str
    label: str
    value: object
    unit: str
    source: str='printed'


def fmt(value, places=None):
    if not isinstance(value,D): return str(value)
    if places is not None:
        with localcontext() as ctx:
            ctx.prec=50
            value=value.quantize(D(1).scaleb(-places),rounding=ROUND_HALF_UP)
        return format(value,'f')
    return format(value,'f')


def quantity(value,unit,places=None):
    text=fmt(value,places)
    if unit=='PKR': return 'PKR '+text
    if unit=='percent': return text+'%'
    return text+((' '+unit) if unit not in ('none','text','date','month') else '')

LABELS={'provider':'electricity provider','tariff':'tariff code','sanctioned_load_kw':'sanctioned load',
'bill_month':'billing month','reading_date':'meter reading date','issue_date':'bill issue date','due_date':'payment due date',
'previous_reading':'previous meter reading','current_reading':'current meter reading','units_consumed':'current billed consumption',
'total_charges':'printed charges subtotal','total_taxes':'total taxes and duties','current_bill':'current bill',
'arrears':'arrears / carried balance','payable_within_due_date':'amount payable within due date',
'payable_after_due_date':'highest printed amount payable after due date'}


def catalog(bill:Bill, extra:ExtraEvidence):
    facts={}
    for key,label in LABELS.items():
        value=getattr(bill,key)
        unit=('date' if key.endswith('_date') and not key.startswith('payable') else 'month' if key=='bill_month' else
              'kW' if key=='sanctioned_load_kw' else 'kWh' if key in ('previous_reading','current_reading','units_consumed') else
              'text' if key in ('provider','tariff') else 'PKR')
        facts[key]=Fact(key,label,value,unit)
    for group in ('charges','taxes'):
        for index,item in enumerate(getattr(bill,group)):
            key=f'{group}.{index}'
            facts[key]=Fact(key,item.type.replace('_',' ')+f' ({group} line {index+1})',item.amount,'PKR')
    for row in extra.history:
        if row.month is None: continue
        for field,unit in [('units','kWh'),('billed_amount','PKR'),('payment','PKR')]:
            key=f'history.{row.month}.{field}'
            facts[key]=Fact(key,row.month+' '+field.replace('_',' '),getattr(row,field),unit)
    for item in extra.facts:
        key='extra.'+item.id
        value=item.value if item.kind=='number' else item.text
        if item.status!='visible' or (item.section=='net_metering' and item.unit=='kWh' and extra.net_metering_status!='clearly_labeled'):
            value=None
        facts[key]=Fact(key,item.label,value,item.unit if item.kind=='number' else item.kind)
    return facts


def public_catalog(facts):
    return [{'id':f.id,'label':f.label,'value':f.value,'unit':f.unit,'source':f.source} for f in facts.values()]


def question_numbers(question):
    values={D(x.replace(',','')) for x in re.findall(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?',question)}
    words=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty']
    for i,word in enumerate(words):
        if re.search(r'\b'+word+r'\b',question,re.I): values.add(D(i))
    return values


def _month_index(month):
    year,mon=map(int,month.split('-')); return year*12+mon-1


def _month_string(index): return f'{index//12:04d}-{index%12+1:02d}'


def resolve(step,facts,bill,extra,computed):
    notes=[]
    if step.series=='none':
        values=[]
        if len(set(step.refs))!=len(step.refs): raise ValueError('Duplicate references would double-count a value')
        for ref in step.refs:
            if ref.startswith('$'):
                if ref[1:] not in computed: raise ValueError('Unknown or forward calculation reference')
                result=computed[ref[1:]]
                values.append(Fact(ref,result['label'],result['value'] if result['status']=='ok' else None,result['unit'],'calculated'))
            elif ref in facts: values.append(facts[ref])
            elif ref.startswith('external.'): raise ValueError(f"External reference {ref} is not allowed")
            else: values.append(Fact(ref, f"Unknown reference {ref}", None, "none", "unknown"))
        return values,notes
    field=step.series.removeprefix('history_')
    rows={row.month:row for row in extra.history if row.month}
    selected={m for m in rows if step.period!='previous' or (bill.bill_month is not None and m<bill.bill_month)}
    if step.period=='previous' and bill.bill_month is None:
        return [],['The current billing month is unavailable, so previous-month selection is uncertain.']
    if step.period=='including_current' and bill.bill_month and field in ('units','billed_amount'):
        selected.add(bill.bill_month)
    if step.start_month: selected={m for m in selected if m>=step.start_month}
    if step.end_month: selected={m for m in selected if m<=step.end_month}
    # last_n means calendar months, not the last N available rows.
    anchor=(bill.bill_month if step.period!='all' else max(selected,default=bill.bill_month))
    if step.last_n and anchor:
        end=_month_index(anchor)-(1 if step.period=='previous' else 0)
        wanted={_month_string(i) for i in range(end-int(step.last_n)+1,end+1)}
        selected &= wanted
        if len(selected)<len(wanted): notes.append(f'Only {len(selected)} of the requested {int(step.last_n)} calendar months are printed.')
    values=[]
    for month in sorted(selected):
        key=f'history.{month}.{field}'
        if key in facts: values.append(facts[key])
        elif month==bill.bill_month and step.period=='including_current':
            key={'units':'units_consumed','billed_amount':'current_bill'}.get(field)
            if key: values.append(Fact(key,month+' current '+field.replace('_',' '),facts[key].value,facts[key].unit))
    if step.last_n and anchor: notes.append('Selected calendar window: '+_month_string(end-int(step.last_n)+1)+' through '+_month_string(end)+'.')
    if step.period=='previous': notes.append('History scope: previous months only; current month excluded.')
    elif step.period=='including_current': notes.append('History scope: includes the current month once.')
    else: notes.append('History scope: all selected printed months.')
    if extra.history_status!='visible' or any(r.month is None for r in extra.history):
        notes.append('The history is incomplete or contains unreadable month labels; missing months are not zero.')
    return values,notes


LIMITATIONS={
'not_printed':'The required information is not available in the bill evidence, so this part cannot be answered reliably.',
'ambiguous_registers':'The bill shows multiple registers without clear import/export labels; a reliable net-metering total cannot be established.',
'external_tariff':'The bill does not provide the required tariff rules or rates; an exact cost cannot be established without outside information.',
'unsupported_cause':'The bill does not establish the cause, so an explanation of why this happened would be speculative.',
'incomplete_history':'The available history is incomplete; calculations cover only the readable months and missing months are not zero.',
'ambiguous_question':'The question has more than one reasonable interpretation; the scopes and denominators used are stated explicitly.',
'estimate_only':'This is an estimate, not a forecast or an official tariff calculation.',
'private_information':'Personal identifiers are excluded from bill answers.'}


def calculate(step, facts, bill, extra, computed, question):
    values,notes=resolve(step,facts,bill,extra,computed)
    op=step.operation; threshold=step.threshold
    if threshold is not None and threshold not in question_numbers(question):
        raise ValueError(f'Threshold {threshold} was not found in the question.')
    missing=[f for f in values if f.value is None]
    available=[f for f in values if f.value is not None]
    source_refs=[f.id for f in values]
    result={'id':step.id,'operation':op,'refs':source_refs,'value':None,'unit':'none','label':op.replace('_',' '),'notes':notes,'status':'ok'}
    if step.series=='none' and not values:
        result['status']='unavailable'
        result['notes'].append('No evidence was selected for this calculation.')
        result['statement']='No evidence was selected to answer this part of the question.'
        return result
    if op=='facts':
        clauses=[f'The printed {f.label} is {quantity(f.value,f.unit)}.' if f.source=='printed' else f'The calculated {f.label} is {quantity(f.value,f.unit)}.' for f in available]
        clauses += [f'The {f.label} is unavailable or ambiguous on this bill.' for f in missing]
        if not values: clauses=['The requested information is not available in the bill.']; result['status']='unavailable'
        if missing: result['status']='partial' if available else 'unavailable'
        if len(values)==1:
            result.update(value=values[0].value,unit=values[0].unit,label=values[0].label)
        result['statement']=' '.join(clauses+notes); return result
    if missing:
        if step.series=='none':
            result.update(status='unavailable',statement='The calculation cannot be established because '+', '.join(f.label for f in missing)+' is unavailable or ambiguous.'); return result
        notes.append(f'{len(missing)} selected month(s) have missing values and are excluded, not treated as zero.')
    if not available:
        result.update(status='unavailable',statement='The required readable values are unavailable, so this calculation cannot be established. '+ ' '.join(notes)); return result
    if op=='days_between':
        if len(available)!=2 or any(f.unit!='date' for f in available): raise ValueError('Two date facts required')
        a,b=available; value=D((date.fromisoformat(a.value)-date.fromisoformat(b.value)).days)
        statement=f'The calendar-day difference between {a.label} ({a.value}) and {b.label} ({b.value}) is {value} days (end minus start).'
        result.update(value=value,unit='days',label='calendar-day difference',statement=statement); return result
    if any(not isinstance(f.value,D) for f in available): raise ValueError('Arithmetic requires numeric evidence')
    units={f.unit for f in available}; nums=[f.value for f in available]
    same_unit=next(iter(units)) if len(units)==1 else None
    if op not in ('ratio','estimate','product') and same_unit is None: raise ValueError('Incompatible units')
    details='; '.join(f'{f.label}: {quantity(f.value,f.unit)}' for f in available)
    suffix=(' '+ ' '.join(notes)) if notes else ''
    unit=same_unit or 'none'; value=None; label=op.replace('_',' ')
    with localcontext() as ctx:
        ctx.prec=40
        if op in ('sum','average','min','max','count'):
            if op=='sum': value=sum(nums,D(0)); explanation=' + '.join(fmt(n) for n in nums)
            elif op=='average': value=sum(nums,D(0))/D(len(nums)); explanation=f'{fmt(sum(nums,D(0)))} / {len(nums)} readable values'
            elif op=='min': value=min(nums); explanation='minimum of the selected values'
            elif op=='max': value=max(nums); explanation='maximum of the selected values'
            else: value=D(len(nums)); unit='none'; explanation='count of readable selected values'
            match=''
            if op in ('min','max'): match='; occurring at '+', '.join(f.label for f in available if f.value==value)
            statement=f'Calculated {label}: {quantity(value,unit,2 if op=="average" else None)} ({explanation}{match}). Inputs: {details}.'
        elif op.startswith('count_'):
            if threshold is None: raise ValueError('Threshold required')
            predicates={'count_above':lambda n:n>threshold,'count_at_least':lambda n:n>=threshold,'count_below':lambda n:n<threshold,'count_at_most':lambda n:n<=threshold}
            selected=[f for f in available if predicates[op](f.value)]; value=D(len(selected));unit='none'
            rel={'count_above':'strictly above','count_at_least':'at least','count_below':'strictly below','count_at_most':'at most'}[op]
            statement=f'{value} of {len(available)} readable values are {rel} {quantity(threshold,same_unit)}.'
            if selected: statement+=' Matching months/values: '+'; '.join(f'{f.label}: {quantity(f.value,f.unit)}' for f in selected)+'.'
        elif op in ('difference','percentage','ratio','percent_change','compare'):
            if len(nums)!=2: raise ValueError('Exactly two operands required')
            a,b=available; x,y=nums
            label=f'{a.label} versus {b.label}'
            if op in ('percentage','ratio','percent_change') and y==0:
                result.update(status='unavailable',statement=f'The calculation is undefined because {b.label}, the denominator, is zero.'); return result
            if op=='difference':
                value=x-y; statement=f'The calculated difference ({a.label} minus {b.label}) is {quantity(value,unit)} ({fmt(x)} - {fmt(y)}).'
            elif op=='percentage':
                value=x/y*100;unit='percent'; statement=f'{a.label.capitalize()} is approximately {fmt(value,2)}% of {b.label} ({fmt(x)} / {fmt(y)} x 100).'
            elif op=='ratio':
                value=x/y;unit='none' if same_unit else a.unit+'/'+b.unit
                statement=f'The calculated ratio of {a.label} to {b.label} is {quantity(value,unit,4)} ({fmt(x)} / {fmt(y)}).'
            elif op=='percent_change':
                if y<0:
                    result.update(status='unavailable',statement='Percentage change from a negative credit balance is ambiguous; use the signed amount difference instead.'); return result
                value=(x-y)/y*100;unit='percent';statement=f'The calculated change from {b.label} to {a.label} is {fmt(value,2)}% (({fmt(x)} - {fmt(y)}) / {fmt(y)} x 100).'
            else:
                value=x-y; rel='greater than' if x>y else 'less than' if x<y else 'equal to'
                statement=f'{a.label.capitalize()} ({quantity(x,a.unit)}) is {rel} {b.label} ({quantity(y,b.unit)}). The signed difference is {quantity(value,unit)}.'
        elif op in ('increase_percent','decrease_percent'):
            if len(nums)!=1 or threshold is None or threshold<0 or (op=='decrease_percent' and threshold>100): raise ValueError('Percentage adjustment requires one value and a nonnegative percentage from the question')
            factor=D(1)+(threshold/D(100) if op=='increase_percent' else -threshold/D(100))
            value=nums[0]*factor;label='hypothetical '+('increase' if op=='increase_percent' else 'decrease')
            statement=f'Calculated hypothetical {label.removeprefix("hypothetical ")} of {fmt(threshold)}% in {available[0].label}: {quantity(value,unit,2)} ({fmt(nums[0])} x {fmt(factor)}). This is a scenario, not a printed bill value.'
        elif op=='product':
            if len(nums)!=2: raise ValueError('Product requires two operands')
            a,b=available
            if {a.unit,b.unit}=={'PKR/kWh','kWh'}: unit='PKR'
            elif a.unit=='none': unit=b.unit
            elif b.unit=='none': unit=a.unit
            else: raise ValueError('Product has unsupported dimensions')
            value=nums[0]*nums[1]
            statement=f'Calculated product of {a.label} and {b.label}: {quantity(value,unit,2)} ({fmt(nums[0])} x {fmt(nums[1])}). This applies only to the stated printed rate conditions.'
        elif op=='estimate':
            # Explicit per-unit linear scaling; never an invented slab schedule.
            if len(nums)==3 and threshold is None and available[2].unit=='kWh': threshold=nums[2]
            if len(nums) not in (2,3) or threshold is None or threshold<0 or available[0].unit!='PKR' or available[1].unit!='kWh': raise ValueError('Estimate requires printed cost, units, and nonnegative target from question or calculation')
            cost,consumption=nums[:2]
            if consumption<=0:
                result.update(status='unavailable',statement='A per-unit estimate cannot be established because the reference consumption is zero or negative.'); return result
            value=cost/consumption*threshold;unit='PKR';label='linear cost estimate'
            statement=f'Estimated cost for {fmt(threshold)} kWh: PKR {fmt(value,2)}, using {available[0].label} / {available[1].label} x target consumption ({fmt(cost)} / {fmt(consumption)} x {fmt(threshold)}). This assumes the same average cost per unit; fixed charges, slabs, taxes and adjustments may change, so it is not an exact tariff-based bill.'
        else: raise ValueError('Unsupported operation')
    if value is not None and not value.is_finite(): raise ValueError('Nonfinite result')
    result.update(value=value,unit=unit,label=label,statement=statement+suffix)
    return result


def execute_plan(plan,bill,extra,question):
    facts=catalog(bill,extra); computed={}; statements=[]
    for step in plan.steps:
        if step.id in computed: raise ValueError('Duplicate calculation ID')
        result=calculate(step,facts,bill,extra,computed,question)
        computed[step.id]=result; statements.append({'id':step.id,'text':result['statement']})
    for i,limitation in enumerate(dict.fromkeys(plan.limitations)):
        text=LIMITATIONS[limitation]
        if limitation=='ambiguous_registers' and extra.net_metering_status!='unlabeled_registers':
            text='Clearly labeled net import/export information is unavailable in this bill evidence.'
        statements.append({'id':f'note_{i}','text':text})
    if not statements: statements=[{'id':'unavailable','text':LIMITATIONS['not_printed']}]
    return {'question_id':plan.question_id,'calculations':list(computed.values()),'statements':statements,
            'answer':' '.join(s['text'] for s in statements)}
