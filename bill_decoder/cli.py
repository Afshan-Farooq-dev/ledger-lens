"""Level 1 extraction and Level 2 question answering; no submission side effects."""
import argparse
from hashlib import sha256
import logging
from pathlib import Path
from .api import Gemini, APIProblem
from .config import Settings
from .engine import extract, atomic_write, ExtractionProblem
from .images import ImageOptions
from .schema import dumps
from .benchmark import benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description='WTQ utility bill extraction and grounded question answering')
    parser.add_argument('--env', type=Path, default=Path('.env'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight')
    for name in ('extract', 'batch'):
        p = sub.add_parser(name)
        p.add_argument('input', type=Path)
        p.add_argument('--output', type=Path, default=Path('output/training'))
        p.add_argument('--cache', type=Path, default=Path('.cache/level1'))
        p.add_argument('--refresh', action='store_true')
        p.add_argument('--rotate', type=int, choices=(0, 90, 180, 270), default=0)
        p.add_argument('--contrast', action='store_true')
        p.add_argument('--detail-crops', action='store_true')
        p.add_argument('--crop', type=float, nargs=4, metavar=('X0','Y0','X1','Y1'))
    p = sub.add_parser('benchmark')
    p.add_argument('actual', type=Path)
    p.add_argument('--gold', type=Path, default=Path('tests/fixtures/KESC_0008.gold.json'))
    p.add_argument('--output', type=Path, default=Path('output/training/KESC_0008.comparison.json'))
    p = sub.add_parser('phase15', help='Independent OCR and two approved-model comparison')
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, default=Path('output/phase1_5'))
    p.add_argument('--confirm-free-tier', action='store_true', required=True, help='Confirm API project is Free tier; no billing changes are made')
    p.add_argument('--models', nargs='+', choices=('gemini-3.1-flash-lite','gemini-3.5-flash-lite'), default=['gemini-3.1-flash-lite','gemini-3.5-flash-lite'])
    p.add_argument('--ocr-crops', action='store_true', help='Add high-resolution financial-crop OCR to independent analysis')
    p.add_argument('--no-clahe', action='store_true')
    p.add_argument('--no-denoise', action='store_true')
    p.add_argument('--no-deskew', action='store_true')
    p.add_argument('--perspective', action='store_true')
    p.add_argument('--scale', type=int, choices=(1,2,3), default=2)
    p.add_argument('--local-only', action='store_true', help='Skip targeted verification; baseline generation still uses the API')
    p = sub.add_parser('answer', help='Answer free-text or CSV questions for one bill')
    p.add_argument('input', type=Path)
    p.add_argument('--question', action='append', default=[])
    p.add_argument('--questions-csv', type=Path)
    p.add_argument('--output', type=Path, default=Path('output/level2'))
    p.add_argument('--cache', type=Path, default=Path('.cache/level2'))
    p.add_argument('--level1-evidence', type=Path)
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--focus-evidence', action='store_true', help='Re-read extra printed evidence for the supplied questions, with a separate cache')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    logging.getLogger('bill_decoder').setLevel(logging.INFO)
    api = None
    try:
        if args.command == 'benchmark':
            report = benchmark(args.actual, args.gold, args.output)
            print(dumps({'exact_fields': report['exact_fields'], 'total_fields': report['total_fields'], 'report': str(args.output)}))
            return 0
        if args.command == 'phase15':
            from .phase15 import run
            from .preprocessing import PreprocessOptions
            options=PreprocessOptions(not args.no_clahe,not args.no_denoise,not args.no_deskew,args.perspective,args.scale)
            report=run(args.input,args.output,Settings.load(args.env),args.models,options,not args.local_only,args.ocr_crops)
            print(dumps({'api_calls':report['api_calls'],'baseline_cache_hits':report['baseline_cache_hits'],
                         'verification_cache_hits':report['verification_cache_hits'],'ocr_cache_hits':report['ocr_cache_hits'],
                         'originals_unchanged':report['originals_unchanged'],'report':str(args.output/'experiment.json')}))
            return 0 if all(item.get('baselines',{}).get('gemini-3.1-flash-lite',{}).get('status')=='ok' for item in report['bills'].values()) else 1
        if args.command == 'answer':
            from .qa import BillQA, Question, load_questions_csv
            if args.output.resolve().is_relative_to(args.input.parent.resolve()):
                raise ValueError('Output must be outside source image directory')
            if args.questions_csv and args.question:
                raise ValueError('Choose CSV questions or free text, not both')
            questions = load_questions_csv(args.questions_csv,args.input.stem) if args.questions_csv else [Question(question_id=f'Q{i+1}',question=q) for i,q in enumerate(args.question)]
            with BillQA(Settings.load(args.env),cache_dir=args.cache) as service:
                report=service.answer(args.input,questions,level1_evidence=args.level1_evidence,refresh=args.refresh,focus_evidence=args.focus_evidence)
            atomic_write(args.output/(args.input.stem+'.answers.json'),dumps(report))
            atomic_write(args.output/(args.input.stem+'.evidence.json'),dumps({'image_sha256':report['image_sha256'],'bill':report['bill'],'extra':report['evidence']}))
            for result in report['results']:
                print(dumps({'question_id':result['question_id'],'question':result['question'],'answer':result['answer']}))
            print(dumps({'api_calls':report['api_calls'],'answer_cache_hit':report['answer_cache_hit'],'report':str(args.output/(args.input.stem+'.answers.json'))}))
            return 0
        api = Gemini(Settings.load(args.env))
        if args.command == 'preflight':
            print(dumps(api.preflight()))
            return 0
        options = ImageOptions(rotate=args.rotate, contrast=args.contrast,
                               detail_crops=args.detail_crops, crop=tuple(args.crop) if args.crop else None)
        paths = [args.input] if args.command == 'extract' else sorted(p for p in args.input.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
        if not paths:
            raise ValueError('No bill images found')
        if len({p.stem for p in paths}) != len(paths):
            raise ValueError('Duplicate bill_id image stems')
        # Outputs and cache must never overwrite input image locations.
        for directory in (args.output, args.cache):
            resolved = directory.resolve()
            if any(resolved == p.parent.resolve() or resolved.is_relative_to(p.parent.resolve()) for p in paths):
                raise ValueError('Output/cache directory must be outside source image directory')
        results = []
        permanent_failure = None
        for path in paths:
            try:
                if permanent_failure:
                    raise APIProblem(permanent_failure[0], permanent_failure[1])
                envelope = extract(path, api, args.cache, options, args.refresh)
                out = args.output / (path.stem + '.json')
                atomic_write(out, dumps(envelope['extraction']['bill']))
                atomic_write(args.output / (path.stem + '.evidence.json'), dumps(envelope))
                result = {'bill_id': path.stem, 'status': 'ok', 'output': str(out),
                          'image_sha256': envelope['image_sha256'], 'flags': envelope['flags']}
            except (APIProblem, ExtractionProblem, ValueError, OSError) as exc:
                # Do not expose arbitrary parser/model/image exceptions or model text.
                if isinstance(exc, APIProblem):
                    error, code = str(exc), exc.code
                    if code in (400, 401, 403, 404):
                        permanent_failure = (code, error)
                elif isinstance(exc, ExtractionProblem):
                    error, code = str(exc), 'VALIDATION'
                else:
                    error, code = 'Local input/output failure: ' + type(exc).__name__, 'LOCAL'
                result = {'bill_id': path.stem, 'status': 'failed', 'code': code, 'error': error}
            results.append(result)
            print(dumps(result), flush=True)
            atomic_write(args.output / ('run_summary.json' if args.command == 'batch' else path.stem + '.run_summary.json'), dumps({'model': api.settings.model, 'api_calls': api.calls, 'results': results}))
        return 0 if all(r['status'] == 'ok' for r in results) else 1
    except APIProblem as exc:
        print(dumps({'status': 'failed', 'code': exc.code, 'error': str(exc)}))
        return 1
    except (ValueError, OSError, ExtractionProblem) as exc:
        print(dumps({'status': 'failed', 'error': 'Configuration/input failure: ' + type(exc).__name__}))
        return 1
    finally:
        if api:
            api.close()
