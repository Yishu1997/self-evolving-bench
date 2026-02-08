import sys
from pathlib import Path

# adding project root (the folder that contains /bench) to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
