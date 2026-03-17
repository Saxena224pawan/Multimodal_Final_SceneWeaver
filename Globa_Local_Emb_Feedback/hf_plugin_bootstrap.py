from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HF_PLUGIN import HFPlugin


def bootstrap() -> None:
    """Example entrypoint for this case folder."""
    hf = HFPlugin()
    # Update repo_id/local_dir per case before running downloads.
    print("HF plugin ready:", hf.__class__.__name__)


if __name__ == "__main__":
    bootstrap()
