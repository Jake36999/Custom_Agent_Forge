
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class AdvocatePayload:
    theory_text: str
    implementation_code: str
    text_to_code_ratio: float


def _extract_yaml_payloads(data: Dict[str, Any], source_name: str) -> List[AdvocatePayload]:
    """Convert a parsed policy YAML into AdvocatePayload list."""
    payloads: List[AdvocatePayload] = []
    schema = data.get("schema", "unknown")

    # ── core_principles: ACS rules ────────────────────────────────────────────
    if "acs_rules" in data:
        identity = data.get("identity", {})
        role = identity.get("role", "")
        desc = identity.get("description", "")
        preamble = f"Role: {role}\n{desc}".strip() if (role or desc) else ""

        for rule in data["acs_rules"]:
            level   = rule.get("level", "INFO")
            message = rule.get("message", "")
            ctype   = rule.get("condition_type", "")
            matches = ", ".join(rule.get("match", []))
            theory  = (
                f"{preamble}\n\n"
                f"Constraint Level: {level}\n"
                f"Enforcement Message: {message}"
            ).strip()
            impl = (
                f"condition_type: {ctype}\n"
                f"match: [{matches}]\n"
                f"level: {level}"
            )
            _push(payloads, theory, impl)

    # ── reasoning_lenses: lenses + ontology mappings ──────────────────────────
    if "base_atomic_lenses" in data:
        ontology = data.get("ontology_mappings", {})
        all_patterns: List[str] = []
        for section in ontology.values():
            if isinstance(section, dict):
                for pat, op_class in section.items():
                    all_patterns.append(f"{pat!r}  →  {op_class}")

        patterns_block = "\n".join(all_patterns)

        for lens in data["base_atomic_lenses"]:
            name    = lens.get("name", "")
            purpose = lens.get("purpose", "")
            theory  = f"Reasoning Lens: {name}\nPurpose: {purpose}"
            impl    = patterns_block if patterns_block else f"lens: {name}"
            _push(payloads, theory, impl)

    # ── cognitive_bindings: binding matrix ────────────────────────────────────
    if "binding_matrix" in data:
        description = data.get("description", "")
        for binding in data["binding_matrix"]:
            target_sid   = binding.get("target_sid") or binding.get("target_pattern_type", "")
            cue          = binding.get("methodological_cue") or binding.get("hardware_activation_cue", "")
            triggered    = binding.get("triggered_lens", "")
            sem_payload  = str(binding.get("semantic_payload", "")).strip()
            graph_trig   = binding.get("graph_complexity_trigger", "")

            theory = (
                f"{description}\n\n"
                f"Binding Cue: {cue}\n"
                f"Triggered Lens: {triggered}\n\n"
                f"Semantic Payload:\n{sem_payload}"
            ).strip()
            impl_parts = [f"target: {target_sid}", f"triggered_lens: {triggered}"]
            if graph_trig:
                impl_parts.append(f"graph_complexity_trigger: {graph_trig}")
            impl = "\n".join(impl_parts)
            _push(payloads, theory, impl)

    # ── generic fallback: stringify top-level keys ────────────────────────────
    if not payloads:
        for key, value in data.items():
            if key == "schema":
                continue
            theory = f"Policy schema: {schema}\nSection: {key}"
            impl   = json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value
            _push(payloads, theory, impl)

    return payloads


def _push(payloads: List[AdvocatePayload], theory: str, impl: str) -> None:
    theory = theory.strip()
    impl   = impl.strip()
    if not theory:
        return
    text_vol  = len(theory)
    code_vol  = len(impl)
    total_vol = text_vol + code_vol
    ratio     = text_vol / total_vol if total_vol > 0 else 1.0
    payloads.append(AdvocatePayload(
        theory_text=theory,
        implementation_code=impl,
        text_to_code_ratio=round(ratio, 3),
    ))


def _extract_notebook_payloads(target_path: Path, target_ratio: float) -> List[AdvocatePayload]:
    """Fallback: process Jupyter notebooks the original way."""
    try:
        from jupyter_trace_ingestor import load_notebook, stream_cells
    except ImportError:
        return []

    nb_node    = load_notebook(str(target_path))
    cell_stream = stream_cells(nb_node)
    payloads: List[AdvocatePayload] = []
    md_buffer:  List[str] = []

    for cell in cell_stream:
        ctype  = cell.get("cell_type")
        source = cell.get("source", "").strip()
        has_error = bool(cell.get("traceback", ""))
        if not source:
            continue
        if ctype == "markdown":
            md_buffer.append(source)
        elif ctype == "code":
            if has_error:
                md_buffer.clear()
                continue
            if md_buffer:
                theory   = "\n\n".join(md_buffer)
                text_vol = len(theory)
                code_vol = len(source)
                total    = text_vol + code_vol
                if total > 0:
                    ratio = text_vol / total
                    if ratio >= target_ratio:
                        payloads.append(AdvocatePayload(
                            theory_text=theory,
                            implementation_code=source,
                            text_to_code_ratio=round(ratio, 3),
                        ))
            md_buffer.clear()
    return payloads


def process_manifest(manifest_path: Path, output_dir: Path, target_ratio: float) -> None:
    if not manifest_path.exists():
        sys.exit(1)
    with open(manifest_path, "r", encoding="utf-8") as f:
        targets = [line.strip() for line in f if line.strip()]

    output_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        target_path = Path(target)
        if not target_path.exists():
            print(f"[!] Missing: {target_path}")
            continue
        try:
            suffix = target_path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                with open(target_path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                payloads = _extract_yaml_payloads(data or {}, target_path.name)
                # Apply ratio filter
                payloads = [p for p in payloads if p.text_to_code_ratio >= target_ratio]
            else:
                payloads = _extract_notebook_payloads(target_path, target_ratio)

            if payloads:
                out_file = output_dir / f"{target_path.stem}_advocate_extract.json"
                with open(out_file, "w", encoding="utf-8") as fh:
                    json.dump([asdict(p) for p in payloads], fh, indent=2)
                print(f"[+] {target_path.name}: {len(payloads)} payloads")
            else:
                print(f"[~] {target_path.name}: 0 payloads (ratio filter or empty)")

        except Exception as e:
            print(f"[!] Error on {target_path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ratio",      type=float, required=True)
    args = parser.parse_args()
    process_manifest(Path(args.manifest), Path(args.output_dir), args.ratio)
