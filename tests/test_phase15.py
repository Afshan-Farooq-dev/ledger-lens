"""OCR/preprocessing and verification safety tests, separate from existing tests."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch,Mock
import unittest
import cv2
import numpy as np
from PIL import Image,ImageDraw
from bill_decoder.ocr import OCR,OCRToken
from bill_decoder.preprocessing import Candidate,PreprocessOptions,candidates,deskew,rectify_perspective,original_bbox,high_resolution_crop,crop_candidate
from bill_decoder.verification import printed_number,printed_date,collect_support,suspicious_fields,recheck
from bill_decoder.schema import Bill,Extraction,dumps,parse_json
from bill_decoder.engine import atomic_write
from bill_decoder.config import Settings,MODEL
from bill_decoder.api import Gemini

GOLD=parse_json(Path('tests/fixtures/KESC_0008.gold.json').read_text(encoding='utf-8-sig'))

class PrettyJSONTests(unittest.TestCase):
    def test_pretty_precision_and_compact_csv_unchanged(self):
        value={'amount':Decimal('123456789.123456789'),'credit':Decimal('-0.5300'),'items':[GOLD]}
        pretty=dumps(value,indent=2)
        self.assertIn('\n  "amount": 123456789.123456789',pretty)
        self.assertIn('-0.5300',pretty)
        self.assertEqual(parse_json(pretty),value)
        self.assertNotIn('\n',dumps(value))
        with TemporaryDirectory() as directory:
            path=Path(directory)/'bill.json'; atomic_write(path,dumps(value))
            self.assertEqual(parse_json(path.read_text()),value)
            self.assertIn('\n  "amount"',path.read_text())

class OCRTests(unittest.TestCase):
    def test_missing_executable_graceful(self):
        with TemporaryDirectory() as directory:
            ocr=OCR(Path(directory)/'missing.exe')
            self.assertFalse(ocr.info['available'])
            tokens,info=ocr.recognize(Candidate('original',Image.new('RGB',(10,10)),np.eye(3),{}),(10,10))
            self.assertEqual(tokens,[]); self.assertEqual(info['status'],'unavailable')

    def test_language_selection_and_image_to_data(self):
        with patch('pytesseract.get_languages',return_value=['eng','urd','osd']),patch('pytesseract.get_tesseract_version',return_value='5.4.0'),patch('pathlib.Path.is_file',return_value=True),patch('pytesseract.image_to_data') as data:
            data.return_value={'text':['','123.45'],'conf':['-1','91.2'],'left':[0,20],'top':[0,40],'width':[0,40],'height':[0,20],'page_num':[1,1],'block_num':[0,1],'par_num':[0,1],'line_num':[0,1]}
            ocr=OCR(); image=Image.new('RGB',(200,200))
            tokens,info=ocr.recognize(Candidate('scaled',image,np.diag([2,2,1]),{}),(100,100))
            self.assertTrue(ocr.info['available']); self.assertEqual(ocr.info['selected_languages'],['eng','urd'])
            self.assertEqual(data.call_args.kwargs['lang'],'eng+urd')
            self.assertEqual(data.call_args.kwargs['output_type'],'dict')
            self.assertEqual(len(tokens),1); self.assertAlmostEqual(tokens[0].confidence,91.2)
            self.assertAlmostEqual(tokens[0].bbox[0],.1); self.assertAlmostEqual(tokens[0].bbox[3],.3)

    def test_english_only_when_urdu_absent(self):
        with patch('pytesseract.get_languages',return_value=['eng','osd']),patch('pytesseract.get_tesseract_version',return_value='5.4.0'),patch('pathlib.Path.is_file',return_value=True):
            self.assertEqual(OCR().info['selected_languages'],['eng'])

    def test_timeout_graceful(self):
        with patch('pytesseract.get_languages',return_value=['eng']),patch('pytesseract.get_tesseract_version',return_value='5.4.0'),patch('pathlib.Path.is_file',return_value=True),patch('pytesseract.image_to_data',side_effect=RuntimeError('timeout')):
            ocr=OCR(); tokens,info=ocr.recognize(Candidate('original',Image.new('RGB',(10,10)),np.eye(3),{}),(10,10))
            self.assertEqual(info['status'],'failed'); self.assertEqual(tokens,[])

class PreprocessingTests(unittest.TestCase):
    def test_original_preserved_and_candidates_configurable(self):
        image=Image.new('RGB',(200,300),'white'); before=image.tobytes()
        views,diagnostics=candidates(image,PreprocessOptions(perspective=True))
        self.assertEqual(views[0].name,'original')
        self.assertIn('gray_clahe',[v.name for v in views]); self.assertIn('denoise',[v.name for v in views])
        self.assertFalse(diagnostics['deskew']['applied']); self.assertFalse(diagnostics['perspective']['applied'])
        self.assertEqual(image.tobytes(),before)
        views,_=candidates(image,PreprocessOptions(False,False,False,False))
        self.assertEqual(len(views),1)

    def test_measurable_skew_corrected_without_clipping_canvas(self):
        image=Image.new('RGB',(700,900),'white'); draw=ImageDraw.Draw(image)
        for y in range(60,850,70): draw.line((50,y,650,y),fill='black',width=3)
        tilted=image.rotate(2,expand=True,fillcolor='white')
        view,meta=deskew(tilted)
        self.assertTrue(meta['applied']); self.assertIsNotNone(view)
        self.assertGreaterEqual(view.image.width,tilted.width)

    def test_perspective_rejects_internal_panel(self):
        image=Image.new('RGB',(400,600),'white')
        result,meta=rectify_perspective(image,[[100,100],[300,100],[300,400],[100,400]])
        self.assertIsNone(result); self.assertFalse(meta['applied'])

    def test_reliable_perspective_transform_and_inverse(self):
        image=Image.new('RGB',(400,600),'white')
        result,meta=rectify_perspective(image,[[5,5],[393,12],[390,590],[6,592]])
        self.assertTrue(meta['applied']); self.assertIsNotNone(result)
        self.assertTrue(all(0<=v<=1 for v in original_bbox([10,10,50,50],result.transform,image.size)))

    def test_crop_ocr_coordinates_map_to_full_image(self):
        image=Image.new('RGB',(100,200))
        view=crop_candidate(image,(.1,.2,.5,.6),2)
        self.assertEqual(view.image.size,(80,160))
        box=original_bbox([0,0,80,160],view.transform,image.size)
        for actual,expected in zip(box,[.1,.2,.5,.6]): self.assertAlmostEqual(actual,expected)

    def test_high_resolution_crop(self):
        image=Image.new('RGB',(100,200))
        self.assertEqual(high_resolution_crop(image,(.1,.2,.5,.6),3).size,(120,240))

class VerificationTests(unittest.TestCase):
    def test_printed_credit_and_dates_normalized_without_summing(self):
        self.assertEqual(printed_number('Rs. 38,209CR'),Decimal('-38209'))
        self.assertEqual(printed_number('2770'),Decimal('2770'))
        self.assertIsNone(printed_number('meter no 12345'))
        self.assertEqual(printed_date('22nd Sep 2026'),'2026-09-22')
        self.assertEqual(printed_date('MAR 2026',True),'2026-03')
        self.assertIsNone(printed_date('31 Feb 26'))

    def test_ocr_disagreement_flags_not_overwrites(self):
        base=Extraction.model_validate({'bill':GOLD,'evidence':[]})
        before=base.bill.model_dump()
        support=[{'field':'current_bill','status':'disagrees'}]
        fields=suspicious_fields(base,support)
        self.assertIn('current_bill',fields); self.assertEqual(base.bill.model_dump(),before)

    def test_multiregister_nulls_flagged_without_inference(self):
        base=Extraction.model_validate({'bill':dict(GOLD,previous_reading=None,current_reading=None,units_consumed=None),'evidence':[]})
        fields=suspicious_fields(base,[],net_metering=True)
        self.assertEqual(set(fields),{'previous_reading','current_reading','units_consumed'})
        self.assertIsNone(base.bill.units_consumed)

    def mock_recheck(self,with_evidence):
        base=Extraction.model_validate({'bill':GOLD,'evidence':[]})
        support=[]
        if with_evidence: support=[{'field':'current_bill','printed_label':'Current Bill','printed_value':'123.45','bbox':None,'status':'visible'}]
        api=SimpleNamespace(settings=Settings('private-key'),calls=0)
        def generate(*args):
            api.calls+=1
            return dumps({'fields':{'current_bill':Decimal('123.45')},'support':support}),{}
        api.generate=generate
        return base,api

    def test_changed_number_requires_fresh_evidence_and_cache_resume(self):
        base,api=self.mock_recheck(True)
        with TemporaryDirectory() as directory:
            image=Image.new('RGB',(200,300),'white')
            result,report=recheck(image,base,{'current_bill':['suspicious']},[],api,directory,'source-hash')
            self.assertEqual(result.bill.current_bill,Decimal('123.45'))
            self.assertEqual(api.calls,1)
            again,report=recheck(image,base,{'current_bill':['suspicious']},[],api,directory,'source-hash')
            self.assertTrue(report['cache_hit']); self.assertEqual(api.calls,1)
            self.assertEqual(base.bill.current_bill,GOLD['current_bill'])

    def test_without_fresh_evidence_baseline_retained(self):
        base,api=self.mock_recheck(False)
        with TemporaryDirectory() as directory:
            result,report=recheck(Image.new('RGB',(200,300)),base,{'current_bill':['OCR disagrees']},[],api,directory,'source-hash')
            self.assertEqual(result.bill.current_bill,base.bill.current_bill)
            self.assertFalse(report['decisions'][0]['accepted'])

    def test_unknown_target_key_rejected_without_changing_bill(self):
        base,api=self.mock_recheck(True)
        api.generate=lambda *args:(dumps({'fields':{'current_bill':123.45,'consumer_number':'not permitted'},'support':[]}),{})
        with TemporaryDirectory() as directory:
            result,report=recheck(Image.new('RGB',(200,300)),base,{'current_bill':['suspicious']},[],api,directory,'source-hash')
            self.assertEqual(result.bill,base.bill); self.assertEqual(report['status'],'validation_failed_baseline_retained')

    def test_both_exact_free_models_allowed_no_substitutions(self):
        with patch('bill_decoder.api.genai.Client'):
            adapter=Gemini(Settings('private-key',model='gemini-3.5-flash-lite'))
            adapter.client.models.generate_content.return_value=SimpleNamespace(text='OK',usage_metadata=None,model_version='gemini-3.5-flash-lite')
            self.assertEqual(adapter.preflight()['model'],'gemini-3.5-flash-lite')
            self.assertEqual(adapter.client.models.generate_content.call_args.kwargs['model'],'gemini-3.5-flash-lite')

if __name__=='__main__': unittest.main()
