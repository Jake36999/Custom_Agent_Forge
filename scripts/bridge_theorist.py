
from ruamel.yaml import YAML
import json
import re

YAML_PATH = "output/knowledge_complexes/KNOWLEDGE_MATRIX_UNIFIED.yaml"
JSONL_PATH = "output/datasets/theorist/extracted_theorist.jsonl"

yaml = YAML(typ="safe")


def load_validated_nodes(yaml_path):
    """Load only validated_nodes from the knowledge matrix YAML.

    Rejected nodes are never propagated — the DAG rejected them for a reason.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        validated = data.get("validated_nodes") or []
        if not isinstance(validated, list):
            validated = []
        if not validated:
            print("[bridge_theorist] WARNING: validated_nodes is empty. Run the DAG pipeline first.")
        else:
            print(f"[bridge_theorist] Using validated_nodes only ({len(validated)} nodes).")
        return validated

    return []


def _sanitize_snippet(raw: str) -> str:
    """Unescape YAML double-serialized newlines and tabs."""
    return raw.replace("\\n", "\n").replace("\\t", "\t")


def _sanitize_name(raw: str) -> str:
    """Strip OCR garbage from a node name.

    Requires at least one real alphabetic word (3+ consecutive letters).
    Falls back to a generic label for pure-numeric strings, OCR fragments
    shorter than 3 chars, or names longer than 40 chars.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    has_real_word = bool(re.search(r"[a-zA-Z]{3,}", cleaned))
    if not cleaned or len(cleaned) < 3 or len(cleaned) > 40 or not has_real_word:
        return "this physical chemistry concept"
    return cleaned


_TABLE_FRAGMENT_RE = re.compile(
    r"(Table\s*\d|Figure\s*\d|see also|listed in|^\s*[\d\.\-\+\s\[\]\(\)%°]+\s*$)",
    re.I,
)

def _is_usable_content(snippet: str) -> bool:
    """Return False for content too short or obviously a table/figure fragment."""
    if len(snippet.strip()) < 80:
        return False
    if _TABLE_FRAGMENT_RE.search(snippet):
        return False
    return True


def main():
    nodes = load_validated_nodes(YAML_PATH)
    count = 0
    skipped = 0
    with open(JSONL_PATH, "w", encoding="utf-8") as out:
        for node in nodes:
            if not isinstance(node, dict):
                continue

            clean_snippet = _sanitize_snippet(str(node.get("code_snippet", "")).strip())

            if not _is_usable_content(clean_snippet):
                skipped += 1
                continue

            clean_name = _sanitize_name(str(node.get("name", "")))

            record = {
                "mode": "theorist",
                "skill_type": "theoretical_reasoning",
                "source_type": "ocr_document",
                "name": clean_name,
                "instruction": (
                    node.get("teaching_layer", {}).get("method_signature")
                    or f"Explain the theoretical framework and chemical properties of {clean_name}."
                ),
                "code_snippet": clean_snippet,
                # _source_doc and _node_id are scrubbed by sft_formatter before SFT output.
                # They carry provenance for OCR grounding and future DPO pair construction.
                "_source_doc": YAML_PATH,
                "_node_id": str(node.get("node_id", "")),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} nodes to {JSONL_PATH} (skipped {skipped} low-quality fragments)")


if __name__ == "__main__":
    main()
