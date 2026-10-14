"""Command line interface. All machine-readable results go to stdout."""
import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentlab", description="Construction-office agent evaluation")
    parser.add_argument("--version", action="version", version="agentlab 0.1.0")
    args = parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
