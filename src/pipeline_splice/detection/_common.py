import os
import sys
from pathlib import Path

import torch

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).parent
else:
    ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(str(ROOT)), relative_path)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
