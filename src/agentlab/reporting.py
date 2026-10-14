"""Readable reports and exclusive artifact directories."""
import json
import os
from pathlib import Path
import tempfile


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def markdown_report(report: dict) -> str:
    lines = ["# AgentLab evaluation", "", f"Run: `{report['run_id']}`", "",
             f"Mode: {report['mode']}; guard: {report['guard_engine']}; fixtures: {report['fixture_count']}; repetitions: {report['repetitions']}.",
             "", f"Public corpus SHA-256: `{report['corpus_digest']}`", "",
             f"Fixtures and gold SHA-256: `{report['fixture_digest']}`", "",
             "| Adapter | Passed / attempts | Errors | Blocks | Mean / median / p95 seconds | Human review | Total cost | Known cost cases |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    currency = report['pricing']['currency'] if report.get('pricing') else 'unspecified currency'
    for row in report['summary']:
        latency = row['latency_seconds']
        cost = 'unknown' if row['total_cost'] is None else f"{row['total_cost']} {currency}"
        lines.append(f"| {_cell(row['adapter'])} | {row['passed']} / {row['cases']} | {row['errors']} | {row['blocked']} | "
                     f"{latency['mean']:.4f} / {latency['median']:.4f} / {latency['p95_nearest_rank']:.4f} | "
                     f"{row['human_review_rate']:.1%} | {_cell(cost)} | {row['cost_known_cases']} / {row['cases']} |")
    lines += ['', '## Usage and review routing', '',
              '| Adapter | Input tokens | Output tokens | Complete usage cases | Review precision | Review recall |',
              '| --- | --- | --- | --- | --- | --- |']
    for row in report['summary']:
        inputs = 'unknown' if row['input_tokens'] is None else str(row['input_tokens'])
        outputs = 'unknown' if row['output_tokens'] is None else str(row['output_tokens'])
        precision = 'not applicable' if row['review_precision'] is None else f"{row['review_precision']:.1%}"
        recall = 'not applicable' if row['review_recall'] is None else f"{row['review_recall']:.1%}"
        lines.append(f"| {_cell(row['adapter'])} | {inputs} | {outputs} | {row['usage_known_cases']} / {row['cases']} | {precision} | {recall} |")
    lines += ['', '## Safety policy decisions', '', '| Adapter | Injection handled / attempts | Benign blocks |', '| --- | --- | --- |']
    for row in report['summary']:
        lines.append(f"| {_cell(row['adapter'])} | {row['injection_policy_correct']} / {row['injection_attempts']} | {row['benign_blocks']} |")
    lines += ['', 'A blocked attack can satisfy the safety policy while its task pass remains false.', '']
    lines += ['', '## Methodology', ''] + [f"- {item}" for item in report['methodology']]
    lines += ['', '## Human review queue', '', '| Case | Adapter / repeat | Task review | Guard review | Reasons |', '| --- | --- | --- | --- | --- |']
    for row in report['records']:
        if row['human_review']:
            lines.append(f"| {_cell(row['case_id'])} | {_cell(row['adapter'])} / {row['repeat']} | {row['task_needs_review']} | "
                         f"{row['guard_review']} | {_cell('; '.join(row['review_reasons']))} |")
    lines += ['', '## Failed task checks', '']
    for row in report['records']:
        if not row['passed']:
            failures = [name for name, passed in row['checks'].items() if not passed]
            detail = ', '.join(failures) or row['status']
            lines.append(f"- {_cell(row['case_id'])} ({_cell(row['adapter'])}, repeat {row['repeat']}): {_cell(detail)}")
    return '\n'.join(lines) + '\n'


def write_text_exclusive(path: str | Path, content: str) -> None:
    """Exclusive creation prevents accidental replacement of retained evidence."""
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(content)


def write_artifacts(report: dict, directory: str | Path) -> Path:
    """Reserve a new directory; atomically publish each complete report file."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    payloads = {'results.json': json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                'report.md': markdown_report(report)}
    for name, content in payloads.items():
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, target / name)
        finally:
            temporary.unlink(missing_ok=True)
    return target
