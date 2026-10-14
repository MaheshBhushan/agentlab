"""Command line interface. All machine-readable results go to stdout."""
import argparse
import json
import sys
from .baseline import run_baseline
from .fixtures import load_cases
from .tasks import build_request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentlab", description="Construction-office agent evaluation")
    parser.add_argument("--version", action="version", version="agentlab 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="List synthetic task cases")
    listing.add_argument("--fixtures")
    run = sub.add_parser("run", help="Run the deterministic offline baseline")
    run.add_argument("case_id")
    run.add_argument("--fixtures")
    run.add_argument("--runner", choices=["baseline"], default="baseline")
    args = parser.parse_args(argv)
    try:
        cases = load_cases(args.fixtures)
        if args.command == "list":
            result = [{"case_id": c["case_id"], "task": c["task"], "tags": c.get("tags", [])} for c in cases]
        else:
            case = next((c for c in cases if c["case_id"] == args.case_id), None)
            if case is None:
                raise ValueError(f"unknown case: {args.case_id}")
            result = run_baseline(build_request(case)).to_dict()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"agentlab: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
