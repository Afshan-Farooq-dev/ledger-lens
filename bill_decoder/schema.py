"""Strict official schema. Decimal never passes through float on JSON input/output."""
from __future__ import annotations
import json
import re
from datetime import date
from decimal import Decimal
from typing import Annotated, Literal
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError('Expected a JSON number, not a string or boolean')
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError('Nonfinite number')
    return result

Number = Annotated[Decimal, BeforeValidator(number)]
ChargeType = Literal['energy', 'fixed', 'fpa', 'quarterly_adjustment', 'surcharge', 'meter_rent', 'subsidy', 'other']
TaxType = Literal['gst', 'electricity_duty', 'income_tax', 'municipal_tax', 'other_tax']

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class Charge(StrictModel):
    type: ChargeType
    amount: Number

    @model_validator(mode='after')
    def official_rules(self):
        if self.amount == 0:
            raise ValueError('Zero charge lines must be omitted')
        if self.type == 'subsidy' and self.amount > 0:
            raise ValueError('Subsidies must be negative deductions')
        return self

class Tax(StrictModel):
    type: TaxType
    amount: Number

class Bill(StrictModel):
    provider: Literal['KE', 'LESCO', 'IESCO'] | None
    tariff: str | None
    sanctioned_load_kw: Number | None
    bill_month: str | None
    reading_date: str | None
    issue_date: str | None
    due_date: str | None
    previous_reading: Number | None
    current_reading: Number | None
    units_consumed: Number | None
    charges: list[Charge]
    total_charges: Number | None
    taxes: list[Tax]
    total_taxes: Number | None
    current_bill: Number | None
    arrears: Number | None
    payable_within_due_date: Number | None
    payable_after_due_date: Number | None

    @field_validator('bill_month')
    @classmethod
    def month(cls, value):
        if value is not None:
            if not re.fullmatch(r'\d{4}-\d{2}', value):
                raise ValueError('Expected YYYY-MM')
            date.fromisoformat(value + '-01')
        return value

    @field_validator('reading_date', 'issue_date', 'due_date')
    @classmethod
    def dates(cls, value):
        if value is not None:
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                raise ValueError('Expected YYYY-MM-DD')
            date.fromisoformat(value)
        return value

class Observation(StrictModel):
    field: str
    printed_label: str
    printed_value: str | None
    # Fractional x0,y0,x1,y1 on EXIF-corrected and explicitly rotated full image.
    bbox: Annotated[list[Annotated[Number, Field(ge=0, le=1)]], Field(min_length=4, max_length=4)] | None
    status: Literal['visible', 'missing', 'unreadable', 'ambiguous']

    @field_validator('field')
    @classmethod
    def valid_field(cls, value):
        if value not in Bill.model_fields and not re.fullmatch(r'(charges|taxes)\[\d+\]\.amount', value):
            raise ValueError('Evidence must reference an official field')
        return value

    @field_validator('bbox')
    @classmethod
    def box(cls, value):
        if value is not None and (len(value) != 4 or not all(0 <= x <= 1 for x in value) or value[0] >= value[2] or value[1] >= value[3]):
            raise ValueError('Expected normalized ordered bounding box')
        return value

class Extraction(StrictModel):
    bill: Bill
    evidence: list[Observation]

    @model_validator(mode='after')
    def check_printed_label_conflicts(self):
        for item in self.evidence:
            label = re.sub(r'[^a-z0-9 ]', ' ', item.printed_label.lower())
            label = ' '.join(label.split())
            if item.field == 'total_charges' and self.bill.total_charges is not None:
                if label in ('current bill', 'grand total') or 'payable' in label:
                    raise ValueError('total_charges must cite printed charges subtotal, not Current Bill/Grand Total/payable')
            match = re.fullmatch(r'taxes\[(\d+)\]\.amount', item.field)
            if match:
                index = int(match.group(1))
                if index >= len(self.bill.taxes):
                    raise ValueError('Tax evidence index is outside extracted tax lines')
                if ('sales tax' in label or re.search(r'\bgst\b', label)) and self.bill.taxes[index].type != 'gst':
                    raise ValueError('Sales Tax / GST printed tax line must have type gst, including statutory suffixes')
                if label in ('taxes', 'total taxes', 'taxes and duties', 'total taxes and duties'):
                    raise ValueError('Combined Taxes summary belongs in total_taxes, not a tax line; taxes=[] if no separate tax lines')
        return self


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if not match:
            raise ValueError('Invalid fenced JSON')
        text = match.group(1)
    def reject_constant(value):
        raise ValueError('Nonfinite JSON literal')
    return json.loads(text, parse_float=Decimal, parse_int=Decimal,
                      parse_constant=reject_constant, object_pairs_hook=_pairs)


def dumps(value, indent=None):
    """Exact Decimal JSON. Compact default is retained for prompts and future CSV."""
    def encode(item, level):
        if isinstance(item, BaseModel):
            item = item.model_dump(mode='python')
        if isinstance(item, Decimal):
            if not item.is_finite():
                raise ValueError('Nonfinite number')
            return format(item, 'f')
        if isinstance(item, (dict, list, tuple)):
            is_dict = isinstance(item, dict)
            opening, closing = ('{', '}') if is_dict else ('[', ']')
            entries = list(item.items()) if is_dict else list(enumerate(item))
            if not entries:
                return opening + closing
            values = []
            for key, child in entries:
                prefix = json.dumps(key, ensure_ascii=False) + (': ' if indent else ':') if is_dict else ''
                values.append(prefix + encode(child, level + 1))
            if indent is None:
                return opening + ','.join(values) + closing
            pad = ' ' * (indent * (level + 1))
            return opening + '\n' + pad + (',\n' + pad).join(values) + '\n' + ' ' * (indent * level) + closing
        return json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return encode(value, 0)


def api_schema():
    """Replace Pydantic Decimal's string alternative with JSON number only."""
    schema = Extraction.model_json_schema()
    def clean(node):
        if isinstance(node, list):
            return [clean(x) for x in node]
        if not isinstance(node, dict):
            return node
        if 'anyOf' in node and any(x.get('type') == 'number' for x in node['anyOf']):
            alternatives = [clean(x) for x in node['anyOf'] if x.get('type') != 'string']
            return alternatives[0] if len(alternatives) == 1 else {'anyOf': alternatives}
        return {k: clean(v) for k, v in node.items()}
    return clean(schema)


def consistency_flags(bill):
    """Diagnostics only; never revise printed fields."""
    flags = []
    for lines, total in [('charges', 'total_charges'), ('taxes', 'total_taxes')]:
        entries, printed = getattr(bill, lines), getattr(bill, total)
        if entries and printed is not None and sum((x.amount for x in entries), Decimal(0)) != printed:
            flags.append(total + ': printed subtotal differs from extracted line sum; retained')
    if all(getattr(bill, k) is not None for k in ['previous_reading', 'current_reading', 'units_consumed']):
        if bill.current_reading - bill.previous_reading != bill.units_consumed:
            flags.append('units_consumed: differs from reading difference; retained')
    return flags
