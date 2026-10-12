"""Extraction, atomic local cache and independent Level 1 projection."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import os
import tempfile
from pydantic import ValidationError
from google.genai import types
from .api import event
from .images import IMAGE_VERSION, ImageOptions, load_image, variants
from .schema import Extraction, Observation, api_schema, consistency_flags, dumps, parse_json

PROMPT = Path(__file__).with_name('prompt.txt').read_text(encoding='utf-8-sig')

class ExtractionProblem(RuntimeError):
    pass


def atomic_write(path, text):
    path = Path(path)
    if path.suffix.lower() == '.json':
        text = dumps(parse_json(text), indent=2) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def cache_identity(image_hash, model, options):
    return sha256(dumps({'image_sha256': image_hash, 'model': model, 'prompt': PROMPT,
        'schema': api_schema(), 'image_version': IMAGE_VERSION, 'options': asdict(options),
        'engine_version': '1'}).encode()).hexdigest()


def extract(path, api, cache_dir=Path('.cache/level1'), options=None, refresh=False):
    path = Path(path)
    options = options or ImageOptions()
    original_hash = sha256(path.read_bytes()).hexdigest()
    image = load_image(path, options)
    key = cache_identity(original_hash, api.settings.model, options)
    cached = Path(cache_dir) / (key + '.json')
    if cached.exists() and not refresh:
        try:
            envelope = parse_json(cached.read_text(encoding='utf-8'))
            if envelope['cache_key'] != key or envelope['image_sha256'] != original_hash:
                raise ValueError('Cache identity mismatch')
            Extraction.model_validate(envelope['extraction'])
            event('cache_hit', bill_id=path.stem)
            return envelope
        except (ValueError, KeyError, ValidationError):
            event('cache_invalid', bill_id=path.stem)
    contents = [PROMPT]
    view_names = []
    for name, data in variants(image, options):
        contents.extend([name, types.Part.from_bytes(data=data, mime_type='image/png')])
        view_names.append(name)
    usage = []
    validation_feedback = ''
    # At most one semantic/JSON repair; it re-reads the same image, never sees a gold answer.
    for round_number in range(2):
        text, call_usage = api.generate(contents + ([validation_feedback] if validation_feedback else []), api_schema())
        usage.append(call_usage)
        try:
            payload = parse_json(text)
            unavailable_boxes = 0
            # Spatial assertions are optional and must not invalidate valid bill fields.
            # Never infer pixel/1000 coordinate transforms from an invalid model box.
            if isinstance(payload, dict) and isinstance(payload.get('evidence'), list):
                for item in payload['evidence']:
                    if isinstance(item, dict) and item.get('bbox') is not None:
                        try:
                            Observation.model_validate(item)
                        except ValidationError as error:
                            if all(e['loc'] and e['loc'][0] == 'bbox' for e in error.errors()):
                                item['bbox'] = None
                                unavailable_boxes += 1
            extraction = Extraction.model_validate(payload)
            if unavailable_boxes:
                event('evidence_boxes_unavailable', bill_id=path.stem, count=unavailable_boxes)
            break
        except (ValueError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                safe_errors = [{ 'path': list(e['loc']), 'type': e['type'], 'message': e['msg']} for e in exc.errors(include_input=False, include_url=False)]
            else:
                safe_errors = [{'type': 'json_parse_error'}]
            event('validation_retry' if round_number == 0 else 'validation_failed', bill_id=path.stem, errors=safe_errors)
            validation_feedback = 'Re-read the bill and return the complete JSON. Previous output failed validation at: ' + dumps(safe_errors)
    else:
        raise ExtractionProblem('Model output failed strict JSON/schema validation after two reads')
    if sha256(path.read_bytes()).hexdigest() != original_hash:
        raise ExtractionProblem('Source image changed during extraction')
    envelope = {
        'cache_key': key, 'image_sha256': original_hash, 'model': api.settings.model,
        'image_options': asdict(options), 'oriented_size': list(image.size), 'views': view_names,
        'extraction': extraction.model_dump(mode='python'),
        'flags': consistency_flags(extraction.bill), 'usage': usage,
        'unavailable_evidence_boxes': unavailable_boxes,
        'evidence_status': 'model_assertions_not_independently_verified',
    }
    atomic_write(cached, dumps(envelope))
    event('extracted', bill_id=path.stem, flags=envelope['flags'])
    return envelope
