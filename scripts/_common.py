"""Thêm thư mục src/ vào sys.path để chạy script mà không cần cài package."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
