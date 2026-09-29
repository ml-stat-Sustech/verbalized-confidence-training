#!/usr/bin/env python3
"""Write IDs of questions answered correctly in every rollout."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line in source:
            row = json.loads(line)
            if row["group_success_rate"] == 1.0:
                output.write(f'{row["id"]}\n')


if __name__ == "__main__":
    main()
