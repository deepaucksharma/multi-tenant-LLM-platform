"""CLI entrypoint for the repo's Hugging Face Jobs launcher."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlops.hf_jobs import main


if __name__ == "__main__":
    raise SystemExit(main())
