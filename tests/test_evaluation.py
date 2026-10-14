"""Controlled failures verify accounting; no provider or simulated live ranking."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentlab.adapters import AdapterError, AdapterResult
from agentlab.baseline import run_baseline
from agentlab.cli import main
from agentlab.evaluation import Pricing, evaluate
from agentlab.fixtures import load_cases
from agentlab.guards import GuardError, GuardResult
from agentlab.models import TaskOutput
from agentlab.reporting import markdown_report, write_artifacts
from agentlab.tasks import build_request


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_cases()
        self.normal = next(case for case in self.cases if case['case_id'] == 'document_fire_rating')

    def test_baseline_and_repetitions_keep_shared_fingerprints(self):
        report = evaluate(self.cases, guard='none', repetitions=2)
        self.assertEqual(len(report['records']), 2 * len(self.cases))
        self.assertTrue(all(row['passed'] for row in report['records']))
        self.assertTrue(all(row['input_tokens'] is None and row['total_cost'] == '0' for row in report['records']))
        for index in range(len(self.cases)):
            self.assertEqual(report['records'][index]['input_digest'], report['records'][index + len(self.cases)]['input_digest'])

    def test_failures_stay_in_denominator_and_require_review(self):
        output = AdapterResult(run_baseline(build_request(self.normal)), 100, 20)
        with patch('agentlab.evaluation.run_adapter', side_effect=[output, AdapterError('SECRET')]):
            report = evaluate([self.normal], adapters=('langchain',), mode='live', repetitions=2,
                              pricing=Pricing('2', '5'))
        summary = report['summary'][0]
        self.assertEqual(summary['pass_rate'], .5)
        self.assertEqual(summary['cases'], 2)
        self.assertEqual(summary['cost_known_cases'], 1)
        self.assertIsNone(summary['total_cost'])
        self.assertEqual(Decimal(report['records'][0]['task_cost']), Decimal('.0003'))
        self.assertTrue(report['records'][1]['human_review'])
        self.assertNotIn('SECRET', json.dumps(report))

    def test_blocks_are_policy_success_only_for_gold_attacks(self):
        attack = next(case for case in self.cases if case['case_id'] == 'document_injection')
        decision = GuardResult(False, 'block', ['test_block'], 'builtin')
        with patch('agentlab.evaluation.check_input', return_value=decision), patch('agentlab.evaluation.run_adapter') as adapter:
            report = evaluate([self.normal, attack])
        adapter.assert_not_called()
        self.assertEqual(report['summary'][0]['cases'], 2)
        self.assertEqual(report['summary'][0]['passed'], 0)
        self.assertEqual(report['summary'][0]['injection_policy_correct'], 1)
        self.assertEqual(report['summary'][0]['benign_blocks'], 1)
        self.assertFalse(report['records'][0]['policy_decision_correct'])
        self.assertTrue(report['records'][1]['policy_decision_correct'])

    def test_guard_failure_and_nemo_cost_unknown(self):
        with patch('agentlab.evaluation.check_input', side_effect=GuardError('secret')):
            record = evaluate([self.normal])['records'][0]
        self.assertEqual(record['status'], 'guard_error')
        self.assertFalse(record['passed'])
        self.assertTrue(record['human_review'])
        result = AdapterResult(run_baseline(build_request(self.normal)), 12, 4)
        with patch('agentlab.evaluation.run_adapter', return_value=result), patch('agentlab.evaluation.check_input', return_value=GuardResult(True, 'allow', [], 'nemo')), patch('agentlab.evaluation.check_output', return_value=GuardResult(True, 'allow', [], 'nemo')):
            record = evaluate([self.normal], adapters=('autogen',), mode='live', guard='nemo', pricing=Pricing(1, 1))['records'][0]
        self.assertIsNotNone(record['task_cost'])
        self.assertIsNone(record['total_cost'])

    def test_expected_labels_never_enter_request_and_gold_digest_changes(self):
        altered = deepcopy(self.normal)
        altered['expected']['answer_contains'] = ['CANARY GOLD SECRET']
        seen = []
        def run(name, request, model):
            seen.append(str(request))
            return AdapterResult(run_baseline(request))
        with patch('agentlab.evaluation.run_adapter', side_effect=run):
            original = evaluate([self.normal], guard='none')
            changed = evaluate([altered], guard='none')
        self.assertTrue(original['records'][0]['passed'])
        self.assertFalse(changed['records'][0]['passed'])
        self.assertEqual(original['corpus_digest'], changed['corpus_digest'])
        self.assertNotEqual(original['fixture_digest'], changed['fixture_digest'])
        self.assertNotIn('CANARY', ''.join(seen))

    def test_unknown_usage_and_invalid_counts(self):
        for inputs, outputs in ((None, 4), (4, None), (True, 4), (-1, 4)):
            with self.subTest(inputs=inputs, outputs=outputs), patch('agentlab.evaluation.run_adapter', return_value=AdapterResult(run_baseline(build_request(self.normal)), inputs, outputs)):
                record = evaluate([self.normal], adapters=('crewai',), mode='live', pricing=Pricing(1, 1))['records'][0]
            self.assertIsNone(record['total_cost'])
            if inputs is True or inputs == -1:
                self.assertFalse(record['passed'])

    def test_invalid_prices_and_mode_are_rejected(self):
        for price in ('NaN', 'Infinity', '-1', 'no', True):
            with self.subTest(price=price), self.assertRaises(ValueError):
                Pricing(price, '1')
        with self.assertRaises(ValueError):
            evaluate([self.normal], adapters=('langchain',))
        with self.assertRaises(ValueError):
            evaluate([self.normal], repetitions=0)

    def test_invalid_citation_fails_even_without_guards(self):
        raw = run_baseline(build_request(self.normal)).to_dict()
        raw['evidence'][0]['quote'] = 'Invented quote'
        with patch('agentlab.evaluation.run_adapter', return_value=AdapterResult(TaskOutput.from_dict(raw))):
            record = evaluate([self.normal], guard='none')['records'][0]
        self.assertFalse(record['passed'])
        self.assertFalse(record['checks']['citations_valid'])

    def test_artifacts_cli_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'run'
            report = evaluate([self.normal])
            write_artifacts(report, target)
            retained = json.loads((target / 'results.json').read_text())
            self.assertEqual(retained['run_id'], report['run_id'])
            self.assertEqual((target / 'report.md').read_text(), markdown_report(report))
            with self.assertRaises(FileExistsError):
                write_artifacts(report, target)
            stream = io.StringIO()
            with redirect_stdout(stream):
                self.assertEqual(main(['report', str(target / 'results.json')]), 0)
            self.assertIn('Human review queue', stream.getvalue())
            self.assertIn('Input tokens', stream.getvalue())
            self.assertIn('Review precision', stream.getvalue())
            self.assertIn(report['fixture_digest'], stream.getvalue())
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['evaluate', '--output', str(target)]), 2)
                self.assertEqual(main(['evaluate', '--input-price', '1']), 2)
                self.assertEqual(main(['evaluate', '--adapter', 'langchain']), 2)
