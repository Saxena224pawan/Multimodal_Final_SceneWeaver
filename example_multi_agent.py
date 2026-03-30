"""Minimal mock-backed example for the multi-agent refinement pipeline."""

from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from tests.multi_agent_support import MockPipeline


def main() -> None:
    pipeline = MockPipeline()
    with TemporaryDirectory() as tmpdir:
        stats = pipeline.run_integration_test(
            output_dir=tmpdir,
            num_windows=2,
            max_iterations=2,
        )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
