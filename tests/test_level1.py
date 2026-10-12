"""Offline contract/failure tests. Live runs are separately recorded, never faked."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from pydantic import ValidationError
from google.genai import errors
from bill_decoder.api import Gemini, APIProblem
from bill_decoder.config import Settings, MODEL
from bill_decoder.schema import Bill, Extraction, parse_json, dumps, api_schema, consistency_flags
from bill_decoder.images import ImageOptions, load_image, variants
from bill_decoder.engine import extract, cache_identity, ExtractionProblem
from bill_decoder.benchmark import compare

GOLD = parse_json(Path('tests/fixtures/KESC_0008.gold.json').read_text(encoding='utf-8-sig'))

class SchemaTests(unittest.TestCase):
    def test_exact_required_schema(self):
        self.assertEqual(len(Bill.model_fields), 18)
        self.assertTrue(all(f.is_required() for f in Bill.model_fields.values()))
        self.assertEqual(Bill.model_validate(GOLD).model_dump(), GOLD)

    def test_missing_and_extra_keys(self):
        for data in [dict(GOLD, name='not permitted'), {k:v for k,v in GOLD.items() if k != 'arrears'}]:
            with self.assertRaises(ValidationError): Bill.model_validate(data)

    def test_strings_bool_nonfinite_rejected(self):
        for value in ['3430.24', True, Decimal('NaN'), float('inf')]:
            with self.assertRaises(ValidationError): Bill.model_validate(dict(GOLD, current_bill=value))

    def test_null_and_empty_combined_taxes(self):
        data = {k: [] if k in ('charges','taxes') else None for k in GOLD}
        self.assertEqual(Bill.model_validate(data).taxes, [])
        data['total_taxes'] = Decimal('20.50')
        self.assertEqual(Bill.model_validate(data).total_taxes, Decimal('20.50'))

    def test_duplicate_types_and_credit_preserved(self):
        bill = Bill.model_validate(GOLD)
        self.assertEqual([x.amount for x in bill.charges if x.type == 'energy'], [Decimal('1054'),Decimal('663.51')])
        self.assertEqual(bill.arrears, Decimal('-.53'))

    def test_zero_charge_positive_subsidy_rejected(self):
        for charge in [{'type':'energy','amount':0}, {'type':'subsidy','amount':1}]:
            with self.assertRaises(ValidationError): Bill.model_validate(dict(GOLD,charges=[charge]))
        bill = Bill.model_validate(dict(GOLD,charges=[{'type':'subsidy','amount':-100}]))
        self.assertEqual(bill.charges[0].amount, -100)

    def test_zero_tax_preserved(self):
        self.assertEqual(Bill.model_validate(dict(GOLD,taxes=[{'type':'gst','amount':0}])).taxes[0].amount,0)

    def test_invalid_enums_and_dates(self):
        for key,value in [('provider','KESC'),('bill_month','2026-13'),('bill_month','2026-4'),('due_date','2026-02-30'),('issue_date','7 Apr 2026')]:
            with self.assertRaises(ValidationError): Bill.model_validate(dict(GOLD,**{key:value}))
        with self.assertRaises(ValidationError): Bill.model_validate(dict(GOLD,charges=[{'type':'gst','amount':1}]))

    def test_json_decimal_exact_roundtrip(self):
        text = dumps(dict(GOLD,current_bill=Decimal('123456789.123456789')))
        self.assertIn('123456789.123456789',text)
        self.assertNotIn('"123456789.123456789"',text)
        self.assertEqual(parse_json(text)['current_bill'],Decimal('123456789.123456789'))

    def test_json_fences_duplicates_invalid(self):
        self.assertEqual(parse_json('```json\n{"x":1.01}\n```')['x'],Decimal('1.01'))
        for text in ['{"x":1,"x":2}','{"x":NaN}','prefix {}','{} trailing']:
            with self.assertRaises(ValueError): parse_json(text)

    def test_api_schema_numbers_not_strings(self):
        node = api_schema()['$defs']['Bill']['properties']['current_bill']
        self.assertEqual(node['anyOf'],[{'type':'number'},{'type':'null'}])

    def test_consistency_does_not_change_print(self):
        bill = Bill.model_validate(dict(GOLD,total_charges=Decimal('999.01')))
        self.assertTrue(consistency_flags(bill))
        self.assertEqual(bill.total_charges,Decimal('999.01'))
        self.assertEqual(bill.current_bill,Decimal('3430.24'))

    def test_evidence_bounds_and_unknown_field(self):
        observation = {'field':'current_bill','printed_label':'Current Bill','printed_value':'3,430.24','bbox':[.1,.1,.5,.5],'status':'visible'}
        Extraction.model_validate({'bill':GOLD,'evidence':[observation]})
        for update in [{'field':'consumer_number'},{'bbox':[.5,.1,.1,.5]},{'bbox':[0,0,2,1]}]:
            with self.assertRaises(ValidationError): Extraction.model_validate({'bill':GOLD,'evidence':[dict(observation,**update)]})

    def test_semantic_label_conflicts_reject_without_correction(self):
        for field,label,updates in [
            ('total_charges','Current Bill',{}),
            ('taxes[1].amount','Sales Tax u/s 3(1)', {'taxes':[{'type':'electricity_duty','amount':29.41},{'type':'other_tax','amount':520.21},{'type':'municipal_tax','amount':20}]}),
            ('taxes[0].amount','Taxes',{'taxes':[{'type':'other_tax','amount':569.62}]}),
        ]:
            observation={'field':field,'printed_label':label,'printed_value':None,'bbox':None,'status':'visible'}
            with self.assertRaises(ValidationError):
                Extraction.model_validate({'bill':dict(GOLD,**updates),'evidence':[observation]})

    def test_comparison_checks_arrays_and_all_fields(self):
        report = compare(GOLD,GOLD)
        self.assertEqual(report['exact_fields'],18)
        actual = deepcopy(GOLD)
        actual['charges'][1]['amount'] += 1
        report = compare(actual,GOLD)
        self.assertEqual(report['exact_fields'],17)
        self.assertTrue(any(x['path']=='charges[1].amount' and not x['match'] for x in report['array_details']))

class ImageTests(unittest.TestCase):
    def test_png_jpg_rotation_contrast_crop_preserves_original(self):
        with TemporaryDirectory() as directory:
            for suffix in ['png','jpg']:
                path = Path(directory)/('bill.'+suffix)
                Image.new('RGB',(80,120),'white').save(path)
                before = path.read_bytes()
                options = ImageOptions(rotate=90,contrast=True,detail_crops=True,crop=(0,0,.5,.5))
                image = load_image(path,options)
                self.assertEqual(image.size,(120,80))
                self.assertEqual(len(variants(image,options)),5)
                self.assertEqual(path.read_bytes(),before)

    def test_exif_orientation(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'bill.jpg'
            image=Image.new('RGB',(80,120))
            exif=image.getexif(); exif[274]=6
            image.save(path,exif=exif)
            self.assertEqual(load_image(path,ImageOptions()).size,(120,80))

    def test_bad_crop(self):
        with self.assertRaises(ValueError): ImageOptions(crop=(.8,0,.2,1))

class ConfigTests(unittest.TestCase):
    def test_allowlist_and_secret_repr(self):
        self.assertNotIn('private-key',repr(Settings('private-key')))
        for model in ['gemini-3.1-flash-lite-preview','gemini-2.5-flash','gpt-5-mini']:
            with self.assertRaises(ValueError): Settings('private-key',model=model)

    def test_env_alias_and_missing(self):
        with TemporaryDirectory() as directory, patch.dict('os.environ',{},clear=True):
            path=Path(directory)/'.env'
            path.write_text('GEMINI_API_KEY=private-key\n',encoding='utf-8')
            self.assertEqual(Settings.load(path).api_key,'private-key')
            with self.assertRaises(ValueError): Settings.load(Path(directory)/'missing')

class APITests(unittest.TestCase):
    def adapter(self):
        with patch('bill_decoder.api.genai.Client'):
            api=Gemini(Settings('private-key'),sleeper=Mock(),clock=lambda:1)
        api.client=Mock()
        return api

    def test_429_then_success_and_no_substitution(self):
        api=self.adapter()
        api.client.models.generate_content.side_effect=[errors.ClientError(429,{'error':{'message':'rate limited'}}),SimpleNamespace(text='OK',usage_metadata=None,model_version=MODEL)]
        self.assertEqual(api.preflight()['status'],'connected')
        self.assertEqual(api.calls,2)
        self.assertTrue(all(x.kwargs['model']==MODEL for x in api.client.models.generate_content.call_args_list))

    def test_auth_not_retried_and_key_redacted(self):
        api=self.adapter()
        api.client.models.generate_content.side_effect=errors.ClientError(403,{'error':{'message':'private-key denied'}})
        with self.assertRaises(APIProblem) as cm: api.preflight()
        self.assertNotIn('private-key',str(cm.exception))
        self.assertEqual(api.calls,1)

    def test_retry_bounded(self):
        api=self.adapter()
        api.client.models.generate_content.side_effect=errors.ServerError(503,{'error':{'message':'unavailable'}})
        with self.assertRaises(APIProblem): api.preflight()
        self.assertEqual(api.calls,3)

class EngineTests(unittest.TestCase):
    def test_cache_resume_invalidation_and_corruption(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); path=root/'bill.png'
            Image.new('RGB',(80,120)).save(path)
            api=SimpleNamespace(settings=Settings('private-key'),generate=Mock(return_value=(dumps({'bill':GOLD,'evidence':[]}),{})))
            original=sha256(path.read_bytes()).hexdigest()
            extract(path,api,root/'cache')
            extract(path,api,root/'cache')
            self.assertEqual(api.generate.call_count,1)
            next((root/'cache').glob('*.json')).write_text('{}',encoding='utf-8')
            extract(path,api,root/'cache')
            self.assertEqual(api.generate.call_count,2)
            extract(path,api,root/'cache',ImageOptions(rotate=90))
            self.assertEqual(api.generate.call_count,3)
            self.assertEqual(sha256(path.read_bytes()).hexdigest(),original)
            self.assertNotEqual(cache_identity(original,MODEL,ImageOptions()),cache_identity(original,MODEL,ImageOptions(contrast=True)))

    def test_invalid_optional_box_does_not_change_bill(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); path=root/'bill.png'
            Image.new('RGB',(80,120)).save(path)
            observation={'field':'current_bill','printed_label':'Current Bill','printed_value':'3430.24','bbox':[100,100,900,900],'status':'visible'}
            api=SimpleNamespace(settings=Settings('private-key'),generate=Mock(return_value=(dumps({'bill':GOLD,'evidence':[observation]}),{})))
            result=extract(path,api,root/'cache')
            self.assertEqual(result['extraction']['bill'],GOLD)
            self.assertIsNone(result['extraction']['evidence'][0]['bbox'])
            self.assertEqual(result['unavailable_evidence_boxes'],1)
            self.assertEqual(api.generate.call_count,1)

    def test_invalid_output_repair_bounded(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); path=root/'bill.png'
            Image.new('RGB',(80,120)).save(path)
            api=SimpleNamespace(settings=Settings('private-key'),generate=Mock(return_value=('{}',{})))
            with self.assertRaises(ExtractionProblem): extract(path,api,root/'cache')
            self.assertEqual(api.generate.call_count,2)
            self.assertFalse(list((root/'cache').glob('*.json')) if (root/'cache').exists() else [])

if __name__=='__main__': unittest.main()
