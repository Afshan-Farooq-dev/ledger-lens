"""Level 2 semantics and failure tests. Model calls mocked here; live results separate."""
from copy import deepcopy
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
import csv
import unittest
from pydantic import ValidationError
from bill_decoder.schema import Bill,parse_json,dumps
from bill_decoder.evidence import ExtraEvidence,HistoryRow,PrintedFact,json_schema,conservative_facts
from bill_decoder.calculations import Step,QuestionPlan,Plans,catalog,calculate,execute_plan
from bill_decoder.qa import BillQA,Question,AnswerOrders,apply_orders,load_questions_csv,validate_plans
from bill_decoder.config import MODEL
from bill_decoder.engine import ExtractionProblem

GOLD=parse_json(Path('tests/fixtures/KESC_0008.gold.json').read_text(encoding='utf-8-sig'))

def extra(rows=(),**changes):
    base={'history':list(rows),'history_status':'visible','facts':[],'net_metering_status':'not_shown'}
    base.update(changes);return ExtraEvidence.model_validate(base)

def row(month,units,amount=None,payment=None):
    return {'month':month,'printed_month':month,'units':units,'raw_units':None if units is None else str(units),
            'units_label':'Units','billed_amount':amount,'raw_billed_amount':None if amount is None else str(amount),
            'amount_label':'Bill','payment':payment,'raw_payment':None if payment is None else str(payment),'payment_label':'Payment'}

def step(operation,refs=(),**changes):
    data={'id':'result','operation':operation,'refs':list(refs),'series':'none','period':'all','start_month':None,'end_month':None,'last_n':None,'threshold':None}
    data.update(changes);return Step.model_validate(data)

class CalculationsTests(unittest.TestCase):
    def setUp(self): self.bill=Bill.model_validate(GOLD)
    def calc(self,s,e=None,q=''):
        e=e or extra();return calculate(s,catalog(self.bill,e),self.bill,e,{},q)
    def test_tax_share_decimal_denominator_and_print_unchanged(self):
        result=self.calc(step('percentage',['total_taxes','current_bill']))
        self.assertIn('16.61%',result['statement'])
        self.assertEqual(result['refs'],['total_taxes','current_bill'])
        self.assertEqual(self.bill.model_dump(),GOLD)
    def test_due_date_and_late_difference(self):
        self.assertIn('2026-04-21',self.calc(step('facts',['due_date']))['statement'])
        self.assertEqual(self.calc(step('difference',['payable_after_due_date','payable_within_due_date']))['value'],D(286))
    def test_explicit_month_reference_does_not_apply_redundant_series_filters(self):
        s=step('compare',['units_consumed','history.2026-03.units'],last_n=1)
        self.assertIsNone(s.last_n)
        self.assertEqual(self.calc(s,extra([row('2026-03',100)]))['value'],51)
    def test_dates_use_calendar_days(self):
        self.assertEqual(self.calc(step('days_between',['due_date','issue_date']))['value'],D(14))
    def test_threshold_strict_and_inclusive(self):
        e=extra([row('2026-01',200),row('2026-02',201),row('2026-03',None)])
        a=self.calc(step('count_above',series='history_units',period='previous',threshold=200),e,'above 200 units')
        b=self.calc(step('count_at_least',series='history_units',period='previous',threshold=200),e,'at least 200 units')
        self.assertEqual(a['value'],1);self.assertEqual(b['value'],2)
        self.assertIn('missing',a['statement'])
    def test_partial_average_not_zero(self):
        e=extra([row('2026-01',100),row('2026-02',None),row('2026-03',200)])
        result=self.calc(step('average',series='history_units',period='previous',last_n=12),e)
        self.assertEqual(result['value'],150)
        self.assertIn('3 of the requested 12',result['statement']);self.assertIn('2 readable values',result['statement'])
    def test_last_month_is_calendar_not_last_readable(self):
        e=extra([row('2026-02',100)])
        result=self.calc(step('average',series='history_units',period='previous',last_n=1),e)
        self.assertEqual(result['status'],'unavailable')
    def test_current_month_included_once(self):
        e=extra([row('2026-03',100),row('2026-04',151)])
        result=self.calc(step('count',series='history_units',period='including_current'),e)
        self.assertEqual(result['value'],2)
    def test_month_range_and_minimum(self):
        e=extra([row('2026-01',50),row('2026-02',100),row('2026-03',120)])
        result=self.calc(step('min',series='history_units',period='previous',start_month='2026-02',end_month='2026-03'),e)
        self.assertEqual(result['value'],100);self.assertIn('2026-02',result['statement'])
    def test_comparison_and_change(self):
        e=extra([row('2026-03',100)])
        self.assertEqual(self.calc(step('compare',['units_consumed','history.2026-03.units']),e)['value'],51)
        self.assertEqual(self.calc(step('percent_change',['units_consumed','history.2026-03.units']),e)['value'],51)
    def test_zero_denominator(self):
        self.bill=Bill.model_validate(dict(GOLD,current_bill=0))
        self.assertEqual(self.calc(step('percentage',['total_taxes','current_bill']))['status'],'unavailable')
    def test_missing_scalar_cannot_be_summed(self):
        self.bill=Bill.model_validate(dict(GOLD,current_bill=None))
        self.assertEqual(self.calc(step('sum',['total_taxes','current_bill']))['status'],'unavailable')
    def test_unlabeled_net_metering_is_unavailable(self):
        self.bill=Bill.model_validate(dict(GOLD,units_consumed=None,previous_reading=None,current_reading=None))
        e=extra(net_metering_status='unlabeled_registers')
        plan=QuestionPlan(question_id='Q1',steps=[step('difference',['current_reading','previous_reading'])],limitations=['ambiguous_registers'])
        answer=execute_plan(plan,self.bill,e,'What is net usage?')['answer']
        self.assertIn('unavailable',answer);self.assertIn('without clear import/export labels',answer)
    def test_unsupported_constant_and_reference(self):
        for s in [step('count_above',['units_consumed'],threshold=999),step('facts',['external.tariff'])]:
            with self.assertRaises(ValueError): self.calc(s,q='What is the usage?')
    def test_duplicate_refs_and_incompatible_units(self):
        for s in [step('sum',['total_taxes','total_taxes']),step('sum',['total_taxes','units_consumed'])]:
            with self.assertRaises(ValueError): self.calc(s)
    def test_estimate_labeled_formula_and_no_external_rate(self):
        result=self.calc(step('estimate',['current_bill','units_consumed'],threshold=180),q='Estimate 180 units')
        self.assertIn('Estimated',result['statement']);self.assertIn('not an exact tariff',result['statement'])
        self.assertEqual(result['value'].quantize(D('0.01')),D('4089.03'))
    def test_net_metering_absence_cannot_claim_multiple_registers(self):
        plan=QuestionPlan(question_id='Q1',steps=[],limitations=['ambiguous_registers'])
        answer=execute_plan(plan,self.bill,extra(),'Net usage?')['answer']
        self.assertNotIn('multiple registers',answer)
        self.assertIn('unavailable',answer)
    def test_percentage_scenario_then_estimate_is_python_calculated(self):
        e=extra();facts=catalog(self.bill,e)
        change=calculate(step('decrease_percent',['units_consumed'],threshold=10),facts,self.bill,e,{},'Estimate cost if units decrease 10%')
        self.assertEqual(change['value'],D('135.9'))
        value=calculate(step('estimate',['current_bill','units_consumed','$result']),facts,self.bill,e,{'result':change},'Estimate cost if units decrease 10%')
        self.assertEqual(value['value'].quantize(D('0.01')),D('3087.22'))
        self.assertIn('Estimated',value['statement'])
    def test_missing_external_data_explicit(self):
        plan=QuestionPlan(question_id='Q1',steps=[],limitations=['external_tariff'])
        self.assertIn('required tariff',execute_plan(plan,self.bill,extra(),'What will next year cost?')['answer'])
    def test_negative_credit_sign_preserved(self):
        self.assertIn('PKR -0.53',self.calc(step('facts',['arrears']))['statement'])
    def test_unlabeled_net_fact_blocked(self):
        fact={'id':'export_units','label':'Export','kind':'number','value':100,'text':None,'unit':'kWh','raw_value':'100','status':'visible','section':'net_metering'}
        e=extra(facts=[fact],net_metering_status='unlabeled_registers')
        self.assertIsNone(catalog(self.bill,e)['extra.export_units'].value)

class EvidenceTests(unittest.TestCase):
    def test_unverified_numeric_fact_becomes_ambiguous_not_corrected(self):
        payload={'facts':[{'kind':'number','status':'visible','value':D('100'),'raw_value':'99','text':None}]}
        result=conservative_facts(payload)['facts'][0]
        self.assertIsNone(result['value']);self.assertEqual(result['status'],'ambiguous');self.assertEqual(result['raw_value'],'99')
    def test_explicit_printed_month_year_normalizes_without_guessing(self):
        e=extra([dict(row(None,20),printed_month='Sep25')])
        self.assertEqual(e.history[0].month,'2025-09')
    def test_connection_label_does_not_prove_register_mapping(self):
        e=extra(net_metering_status='clearly_labeled')
        self.assertEqual(e.net_metering_status,'unreadable')
    def test_duplicate_months_and_raw_mismatch(self):
        for rows in [[row('2026-01',10),row('2026-01',20)],[dict(row('2026-01',10),raw_units='20')]]:
            with self.assertRaises(ValidationError): extra(rows)
    def test_ambiguous_year_does_not_become_zero_or_guessed(self):
        e=extra([dict(row(None,10),printed_month='Jan')])
        self.assertFalse(any(k.startswith('history.') for k in catalog(Bill.model_validate(GOLD),e)))
    def test_private_fact_and_unsupported_raw_value_rejected(self):
        base={'id':'rate','label':'Rate','kind':'number','value':10,'text':None,'unit':'PKR/kWh','raw_value':'10','status':'visible','section':'tariff'}
        for change in [{'label':'Account number'},{'raw_value':'20'},{'status':'ambiguous'}]:
            with self.assertRaises(ValidationError): PrintedFact.model_validate(dict(base,**change))
    def test_json_stays_readable_exact_decimal(self):
        e=extra([row('2026-01',D('12.123456789'))])
        encoded=dumps(e,indent=2)
        self.assertIn('\n  "history"',encoded);self.assertIn('12.123456789',encoded)
        self.assertEqual(ExtraEvidence.model_validate(parse_json(encoded)),e)

class QATests(unittest.TestCase):
    def prepared(self): return {'bill':Bill.model_validate(GOLD),'extra':extra(),'image_sha256':'abc','flags':[]}
    def api(self,responses):
        api=SimpleNamespace(settings=SimpleNamespace(model=MODEL),calls=0)
        def generate(*args):
            api.calls+=1
            if not responses: raise AssertionError('Unexpected API call')
            return dumps(responses.pop(0)),{}
        api.generate=generate;return api
    def plan(self): return Plans(questions=[QuestionPlan(question_id='Q1',steps=[step('facts',['due_date'])],limitations=[])]).model_dump()
    def test_unsupported_composition_cannot_invent_claims(self):
        results=[{'question_id':'Q1','statements':[{'id':'a','text':'Verified fact.'}]}]
        for ids in [['invented'],['a','a'],[]]:
            with self.assertRaises(ValueError): apply_orders(AnswerOrders.model_validate({'questions':[{'question_id':'Q1','statement_ids':ids}]}),results)
    def test_cached_api_and_changed_question_invalidates(self):
        with TemporaryDirectory() as tmp:
            api=self.api([self.plan(),{'questions':[{'question_id':'Q1','statement_ids':['result']}]}])
            service=BillQA(api=api,cache_dir=Path(tmp))
            question=[Question(question_id='Q1',question='When is payment due?')]
            first=service.answer_prepared(self.prepared(),question)
            second=service.answer_prepared(self.prepared(),question)
            self.assertEqual(first['results'][0]['answer'],second['results'][0]['answer'])
            self.assertTrue(second['answer_cache_hit']);self.assertEqual(api.calls,2)
            with self.assertRaises(AssertionError): service.answer_prepared(self.prepared(),[Question(question_id='Q1',question='What is the issue date?')])
    def test_invalid_plan_retry_is_bounded(self):
        with TemporaryDirectory() as tmp:
            api=self.api([{'bad':1},{'bad':2}]);service=BillQA(api=api,cache_dir=Path(tmp))
            with self.assertRaises(ExtractionProblem): service.answer_prepared(self.prepared(),[Question(question_id='Q1',question='Due date?')])
            self.assertEqual(api.calls,2)
    def test_bad_composition_falls_back_to_grounded_prose(self):
        with TemporaryDirectory() as tmp:
            api=self.api([self.plan(),{'questions':[{'question_id':'Q1','statement_ids':['invented']}]}]);service=BillQA(api=api,cache_dir=Path(tmp))
            result=service.answer_prepared(self.prepared(),[Question(question_id='Q1',question='Due date?')])
            self.assertEqual(result['composition_status'],'deterministic_fallback')
            self.assertIn('2026-04-21',result['results'][0]['answer']);self.assertNotIn('invented',result['results'][0]['answer'])
    def test_other_model_rejected(self):
        api=self.api([]);api.settings.model='gemini-3.5-flash-lite'
        with self.assertRaises(ValueError): BillQA(api=api)
    def test_csv_input_preserves_id_question_and_template(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'questions.csv'
            with path.open('w',encoding='utf-8',newline='') as f:
                writer=csv.writer(f);writer.writerow(['bill_id','question_id','question','answer']);writer.writerow(['B','Q1','What are taxes, fees and "extras"?','']);writer.writerow(['C','Q1','Other bill',''])
            before=path.read_bytes();questions=load_questions_csv(path,'B')
            self.assertEqual(questions[0].question,'What are taxes, fees and "extras"?');self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(ValueError):load_questions_csv(path,'missing')
    def test_tax_share_must_include_calculation(self):
        plans=Plans(questions=[QuestionPlan(question_id='Q1',steps=[step('facts',['total_taxes'])],limitations=[])])
        with self.assertRaises(ValueError):validate_plans(plans,[Question(question_id='Q1',question='How much of my bill is taxes?')],Bill.model_validate(GOLD),extra())
    def test_current_bill_estimate_cannot_use_pretax_subtotal(self):
        plans=Plans(questions=[QuestionPlan(question_id='Q1',steps=[step('estimate',['total_charges','units_consumed'],threshold=180)],limitations=[])])
        with self.assertRaises(ValueError):validate_plans(plans,[Question(question_id='Q1',question='Estimate current bill cost for 180 units')],Bill.model_validate(GOLD),extra())
    def test_question_identity_rejected(self):
        with self.assertRaises(ValueError): validate_plans(Plans.model_validate(self.plan()),[Question(question_id='Q2',question='due?')],Bill.model_validate(GOLD),extra())

if __name__=='__main__':unittest.main()
