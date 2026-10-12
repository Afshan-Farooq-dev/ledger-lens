"""Exact field and array comparison; evaluation never affects extraction."""
from pathlib import Path
from .schema import Bill, dumps, parse_json
from .engine import atomic_write


def compare(actual, expected):
    actual = Bill.model_validate(actual).model_dump()
    expected = Bill.model_validate(expected).model_dump()
    fields, details = [], []
    def walk(path, observed, reference):
        if isinstance(reference, dict) and isinstance(observed, dict):
            for key in reference:
                walk(path + '.' + key, observed.get(key), reference[key])
        elif isinstance(reference, list) and isinstance(observed, list):
            details.append({'path': path + '.length', 'match': len(observed) == len(reference), 'actual': len(observed), 'expected': len(reference)})
            for i in range(max(len(observed), len(reference))):
                walk(f'{path}[{i}]', observed[i] if i < len(observed) else None, reference[i] if i < len(reference) else None)
        else:
            details.append({'path': path, 'match': observed == reference, 'actual': observed, 'expected': reference})
    for field, value in expected.items():
        fields.append({'field': field, 'match': actual[field] == value, 'actual': actual[field], 'expected': value})
        if field in ('charges', 'taxes'):
            walk(field, actual[field], value)
    return {'exact_fields': sum(row['match'] for row in fields), 'total_fields': len(fields),
            'fields': fields, 'array_details': details,
            'comparison': 'Exact numeric equality and ordered arrays; no tolerance or corrections'}


def benchmark(actual_path, gold_path, output):
    report = compare(parse_json(Path(actual_path).read_text(encoding='utf-8-sig')),
                     parse_json(Path(gold_path).read_text(encoding='utf-8-sig')))
    atomic_write(output, dumps(report))
    return report
