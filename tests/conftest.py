import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:  # matplotlib is optional — only some legacy tests use it
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
