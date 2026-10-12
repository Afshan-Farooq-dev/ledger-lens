"""Reproducible training smoke check; these are development questions, not official Q1-Q12."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bill_decoder.qa import BillQA,Question
from bill_decoder.config import Settings
from bill_decoder.engine import atomic_write
from bill_decoder.schema import dumps

QUESTIONS=[
'How much of my bill is taxes?',
'How many months in my history went above 200 units?',
'When is payment due and how much more would I pay after the due date?',
'What was my average consumption over the previous 12 months?',
'How does current consumption compare with the previous month?',
"Estimate the cost for 180 units using this bill\'s average current cost per unit.",
'What is my account number?',
'Why did my bill increase?',
'What are my net import and export units? Are the registers clearly labeled?',
'Which previous month had the highest consumption, and which had the lowest?',
'What is my tariff code, and what are the printed current meter readings?',
'What electricity tariff will apply next year?'
]

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--bill',action='append');args=parser.parse_args()
    out=Path('output/level2');out.mkdir(parents=True,exist_ok=True)
    summary={'purpose':'Development smoke questions, not official competition template','questions':QUESTIONS,'results':[],'api_calls':0}
    with BillQA(Settings.load()) as service:
        for image in sorted(Path('train/bills').iterdir()):
            if image.suffix.lower() not in ('.jpg','.png','.jpeg'):continue
            if args.bill and image.stem not in args.bill: continue
            start=service.api.calls
            try:
                report=service.answer(image,[Question(question_id=f'T{i+1}',question=q) for i,q in enumerate(QUESTIONS)])
                atomic_write(out/(image.stem+'.answers.json'),dumps(report))
                atomic_write(out/(image.stem+'.evidence.json'),dumps({'image_sha256':report['image_sha256'],'bill':report['bill'],'extra':report['evidence']}))
                result={'bill_id':image.stem,'status':'ok','api_calls':service.api.calls-start,'answers':len(report['results']),
                        'history_rows':len(report['evidence'].history),'evidence_cache_hit':report['evidence_cache_hit'],
                        'answer_cache_hit':report['answer_cache_hit'],'level1_cache_hit':report['level1_cache_hit'],
                        'composition_status':report['composition_status']}
            except Exception as exc:
                result={'bill_id':image.stem,'status':'failed','api_calls':service.api.calls-start,'error_type':type(exc).__name__,'reason':str(exc) if type(exc).__name__ in ('ExtractionProblem','APIProblem') else 'Local validation failed'}
            summary['results'].append(result);summary['api_calls']=service.api.calls
            atomic_write(out/'training_summary.json',dumps(summary));print(dumps(result),flush=True)
    return 0 if all(r['status']=='ok' for r in summary['results']) else 1

if __name__=='__main__':raise SystemExit(main())
