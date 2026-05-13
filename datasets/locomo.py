from __future__ import annotations

import sys
from pathlib import Path

# 把 A-mem 目录加入 sys.path，使 load_dataset 可直接 import
_AMEM_DIR = Path(__file__).parent.parent.parent / "A-mem"
if str(_AMEM_DIR) not in sys.path:
    sys.path.insert(0, str(_AMEM_DIR))

from load_dataset import load_locomo_dataset, LoCoMoSample, QA, Turn, Session, Conversation  # noqa: F401

DEFAULT_DATA_PATH = _AMEM_DIR / "data" / "locomo10.json"


def load(path: str | Path | None = None) -> list[LoCoMoSample]:
    return load_locomo_dataset(path or DEFAULT_DATA_PATH)
