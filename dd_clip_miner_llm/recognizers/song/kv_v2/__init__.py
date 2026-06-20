"""KV v3 song pipeline — isolated optimization package.

Improvements over kv_v2:
- Preserves high-confidence known titles (fixes "唯一" regression)
- Detects opening humming segments
- Skips small-cluster reviews (cost reduction ~40%)
- Uses 0.75 deletion threshold (vs 0.70 in kv_v2)
- Special opening segment handling
"""
from .pipeline import run

__all__ = ["run"]
