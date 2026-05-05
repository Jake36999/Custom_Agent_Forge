import json
import math
from collections import Counter
from typing import Any, List

def stable_json_dumps(value: Any) -> str:
    """Serializes objects deterministically for hashing and deduplication."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _get_stable_key(item: Any) -> str:
    """Returns a deterministic hash for deduplication, using stable_hash if available, else stable_json_dumps."""
    if hasattr(item, 'stable_hash') and callable(getattr(item, 'stable_hash')):
        return item.stable_hash()
    return stable_json_dumps(item)

def dedupe_stable_items(items: List[Any]) -> List[Any]:
    """Removes duplicates from a list of objects based on their stable hash or JSON string."""
    deduped = []
    seen = set()
    for item in items or []:
        item_key = _get_stable_key(item)
        if item_key not in seen:
            deduped.append(item)
            seen.add(item_key)
    return deduped


def dedupe_and_sort_stable_items(items: List[Any]) -> List[Any]:
    """Dedupes and sorts a list of items deterministically by their stable key."""
    return sorted(dedupe_stable_items(items), key=_get_stable_key)


def append_unique_stable_item(items: List[Any], item: Any) -> bool:
    """Appends an item to a list only if it doesn't already exist (checked via stable hash or JSON). Returns True if added."""
    item_key = _get_stable_key(item)
    for existing in items:
        if _get_stable_key(existing) == item_key:
            return False
    items.append(item)
    return True

def shannon_entropy(data: str) -> float:
    """Calculates Shannon Entropy (Byte-Encoded) to detect highly obfuscated garbage."""
    if not data: return 0.0
    counts = Counter(data.encode('utf-8'))
    total = sum(counts.values())
    return -sum((c/total) * math.log2(c/total) for c in counts.values())