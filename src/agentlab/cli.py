"""CLI with explicit execution modes and machine-readable stdout."""
import argparse
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys

from .adapters import ADAPTERS
from .evaluation import Pricing, evaluate
from .fixtures import load_cases
from .reporting import markdown_report, write_artifacts, write_text_exclusive


def _execution_options(parser):
    parser.add_argument('--fixtures', help='Synthetic corpus JSON; defaults to packaged cases')
    parser.add_argument('--mode', choices=('offline', 'live'), default='offline')
    parser.add_argument('--adapter', '--runner', choices=ADAPTERS, action='append', dest='adapters',
                        help='Repeat to compare adapters; offline default: baseline')
    parser.add_argument('--guard', choices=('none', 'builtin', 'nemo'), default='builtin')
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--input-price', help='User-supplied price per million input tokens')
    parser.add_argument('--output-price', help='User-supplied price per million output tokens')
    parser.add_argument('--currency', default='USD')
    parser.add_argument('--output', type=Path, help='New directory for results.json and report.md; existing paths rejected')


def _preflight(args, adapters):
    if args.mode != 'live':
        return
    modules = {'langchain': ('langchain', 'langchain_openai'), 'autogen': ('autogen_agentchat', 'autogen_ext'), 'crewai': ('crewai',)}
    for adapter in adapters:
        for module in modules.get(adapter, ()):
            if find_spec(module) is None:
                raise ValueError(f'{adapter} dependency unavailable; install agentlab[{adapter}]')
    if args.guard == 'nemo' and find_spec('nemoguardrails') is None:
        raise ValueError('nemo dependency unavailable; install agentlab[nemo]')
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('live OpenAI execution requires OPENAI_API_KEY in the environment')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='agentlab', description='Synthetic construction-office agent evaluation')
    parser.add_argument('--version', action='version', version='agentlab 0.1.0')
    sub = parser.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list', help='List synthetic cases')
    listing.add_argument('--fixtures')
    run = sub.add_parser('run', help='Execute and grade one synthetic case')
    run.add_argument('case_id')
    _execution_options(run)
    benchmark = sub.add_parser('evaluate', help='Compare adapters over identical cases, serially')
    _execution_options(benchmark)
    report = sub.add_parser('report', help='Render retained results JSON as Markdown')
    report.add_argument('results', type=Path)
    report.add_argument('--output', type=Path, help='New Markdown file; existing file rejected')
    args = parser.parse_args(argv)
    try:
        if args.command == 'report':
            content = markdown_report(json.loads(args.results.read_text(encoding='utf-8')))
            if args.output:
                write_text_exclusive(args.output, content)
            else:
                print(content, end='')
            return 0
        cases = load_cases(args.fixtures)
        if args.command == 'list':
            result = [{'case_id': case['case_id'], 'task': case['task'], 'tags': case.get('tags', [])} for case in cases]
        else:
            adapters = tuple(args.adapters or ('baseline',))
            if args.output and args.output.exists():
                raise ValueError(f'artifact path already exists: {args.output}')
            if (args.input_price is None) != (args.output_price is None):
                raise ValueError('supply both --input-price and --output-price')
            pricing = Pricing(args.input_price, args.output_price, args.currency) if args.input_price is not None else None
            if args.command == 'run':
                cases = [case for case in cases if case['case_id'] == args.case_id]
                if not cases:
                    raise ValueError(f'unknown case: {args.case_id}')
            _preflight(args, adapters)
            result = evaluate(cases, adapters=adapters, mode=args.mode, guard=args.guard,
                              model=args.model, repetitions=args.repetitions, pricing=pricing)
            if args.output:
                write_artifacts(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        # A completed evaluation returns 0 even when cases fail; inspect summary errors/pass rates.
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'agentlab: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
