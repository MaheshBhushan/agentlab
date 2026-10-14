"""Contract boundaries and independent baseline regression checks."""
import json
import unittest
from agentlab.baseline import run_baseline
from agentlab.fixtures import load_cases
from agentlab.models import TaskOutput
from agentlab.tasks import build_request, request_messages


class CoreTests(unittest.TestCase):
    def test_all_cases_are_supported_and_evidenced(self):
        cases = load_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual({c['task'] for c in cases}, {'invoice_triage', 'document_question', 'meeting_followup'})
        for case in cases:
            with self.subTest(case=case['case_id']):
                request = build_request(case)
                output = run_baseline(request)
                self.assertEqual(output.status, case['expected']['status'])
                self.assertEqual(TaskOutput.from_dict(output.to_dict()), output)
                sources = {d.document_id: d.text for d in request.documents}
                for evidence in output.evidence:
                    self.assertIn(evidence.quote, sources[evidence.document_id])
                if 'invoice' in case['expected']:
                    for key, value in case['expected']['invoice'].items():
                        self.assertEqual(output.invoice[key], value)
                if 'action_items' in case['expected']:
                    self.assertEqual([{k: getattr(item, k) for k in ('description','owner','due_date')} for item in output.action_items], case['expected']['action_items'])
                for fragment in case['expected'].get('answer_contains', []):
                    self.assertIn(fragment, output.answer)

    def test_expected_labels_never_enter_request_or_baseline(self):
        case = load_cases()[0]
        original = run_baseline(build_request(case))
        case['expected'] = {'canary': 'TOP_SECRET_EVALUATOR_LABEL'}
        case['documents'][0]['expected'] = 'NESTED_SECRET'
        case['metadata'] = {'expected': 'OTHER_SECRET'}
        request = build_request(case)
        prompt = json.dumps(request_messages(request))
        self.assertNotIn('SECRET', prompt)
        self.assertNotIn('expected', prompt)
        self.assertEqual(run_baseline(request), original)

    def test_contract_rejects_malformed_outputs_and_duplicate_sources(self):
        output = run_baseline(build_request(load_cases()[0])).to_dict()
        output['status'] = 'paid'
        with self.assertRaises(ValueError):
            TaskOutput.from_dict(output)
        output['status'] = 'needs_review'
        with self.assertRaises(ValueError):
            TaskOutput.from_dict(output)
        case = load_cases()[0]
        case['documents'].append(case['documents'][0])
        with self.assertRaises(ValueError):
            build_request(case)

    def test_baseline_reads_changed_inputs(self):
        case = load_cases()[0]
        case['documents'][0]['text'] = case['documents'][0]['text'].replace('1.190,00', '2.380,00')
        self.assertEqual(run_baseline(build_request(case)).invoice['amount_eur'], '2380.00')
        case['documents'][0]['text'] = case['documents'][0]['text'].replace('2.380,00', 'NaN')
        self.assertIsNone(run_baseline(build_request(case)).invoice['amount_eur'])
        self.assertEqual(run_baseline(build_request(case)).status, 'needs_review')


if __name__ == '__main__':
    unittest.main()
