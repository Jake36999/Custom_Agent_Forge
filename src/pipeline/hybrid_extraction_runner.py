import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import asdict

try:
    from jupyter_trace_ingestor import load_notebook, stream_cells
    from veteran_state_machine import VeteranStateMachine
    from advocate_extraction_runner import AdvocateExtractor
except ImportError:
    print("[!] Critical Import Error: Ensure ingestor and state machines are in PYTHONPATH.")
    sys.exit(1)

def _build_veteran_trajectory(
    execution_pattern: List[str],
    traceback_text: str,
    diff: str,
    resolved_code: str,
) -> Dict[str, Any]:
    """Build a reasoning_trajectory dict with confusion_matrix and response_vector.

    Each row of the confusion_matrix encodes four binary/continuous signals:
        [has_traceback, has_diff, has_resolution, step_confidence]

    The response_vector contains per-step resolution scores:
        1.0 if the step produced a fix, 0.0 otherwise.
    """
    has_tb = 1.0 if traceback_text else 0.0
    has_diff = 1.0 if diff else 0.0
    has_res = 1.0 if resolved_code else 0.0

    confusion_matrix: List[List[float]] = []
    response_vector: List[float] = []

    for step_label in execution_pattern:
        sl = step_label.lower()
        if sl in ("fix", "resolution", "resolved"):
            row = [has_tb, has_diff, has_res, 1.0]
            rv = 1.0
        elif sl in ("error", "traceback", "failure"):
            row = [has_tb, 0.0, 0.0, 0.0]
            rv = 0.0
        elif sl in ("diagnosis", "analysis", "diagnostic"):
            row = [has_tb, 0.0, 0.0, 0.5]
            rv = 0.0
        else:  # attempt, unknown
            row = [0.0, 0.0, 0.0, 0.5]
            rv = 0.0
        confusion_matrix.append(row)
        response_vector.append(rv)

    return {
        "confusion_matrix": confusion_matrix,
        "response_vector": response_vector,
        "labels": execution_pattern,
        "metadata": {
            "has_traceback": bool(traceback_text),
            "has_diff": bool(diff),
            "has_resolution": bool(resolved_code),
        },
    }


def _enrich_veteran_payload(payload: dict) -> dict:
    """Inject reasoning_trajectory into a veteran extraction payload's semantics."""
    traceback_text = payload.get("error_traceback", payload.get("traceback", ""))
    diff = payload.get("successful_diff", payload.get("diff", ""))
    resolved_code = payload.get("resolved_code", payload.get("attempt_code", ""))
    execution_pattern = payload.get("execution_pattern", ["attempt", "error", "diagnosis", "fix"])

    trajectory = _build_veteran_trajectory(execution_pattern, traceback_text, diff, resolved_code)
    semantics = payload.get("semantics", {})
    if not isinstance(semantics, dict):
        semantics = {}
    semantics["reasoning_trajectory"] = trajectory
    payload["semantics"] = semantics
    return payload


def process_manifest(manifest_path: Path, output_dir: Path):
    if not manifest_path.exists(): sys.exit(1)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        targets = [line.strip() for line in f if line.strip()]
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default hybrid tuning ratio
    advocate_ratio = 0.65 

    for target in targets:
        target_path = Path(target)
        if not target_path.exists(): continue
            
        try:
            nb_node = load_notebook(str(target_path))
            
            # Crucial: Cast the generator to a list so both extractors can iterate over it
            cell_stream = list(stream_cells(nb_node))
            
            # 1. Run Veteran Pipeline
            vet_machine = VeteranStateMachine()
            vet_payloads = vet_machine.process_stream(cell_stream)
            
            # 2. Run Advocate Pipeline
            adv_machine = AdvocateExtractor(target_ratio=advocate_ratio)
            adv_payloads = adv_machine.process_stream(cell_stream)
            
            # 3. Combine and Serialize
            combined = []
            if vet_payloads:
                vet_dicts = [asdict(p) for p in vet_payloads]
                for vd in vet_dicts:
                    _enrich_veteran_payload(vd)
                combined.extend(vet_dicts)
            if adv_payloads: combined.extend([asdict(p) for p in adv_payloads])
            
            if combined:
                out_file = output_dir / f"{target_path.stem}_hybrid_extract.json"
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(combined, f, indent=2)
                print(f"    [+] Hybrid extraction: {len(vet_payloads)} Veteran + {len(adv_payloads)} Advocate payloads saved.")
                
        except Exception as e:
            print(f"    [!] Error processing {target_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    process_manifest(Path(args.manifest), Path(args.output_dir))
