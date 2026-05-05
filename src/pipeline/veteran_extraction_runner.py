"""
veteran_extraction_runner.py
The Subprocess Bridge - Connects the DAG orchestrator to the Jupyter ingestion 
and state machine extraction logic.
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import json
from pathlib import Path
from dataclasses import asdict

# Assuming these scripts sit in the same src/pipeline/ directory
# Adjust imports if your structure strictly separates core logic from runners
try:
    from jupyter_trace_ingestor import load_notebook, stream_cells
    from veteran_state_machine import VeteranStateMachine
except ImportError:
    print("[!] Critical Import Error: Ensure jupyter_trace_ingestor.py and veteran_state_machine.py are in the PYTHONPATH.")
    sys.exit(1)

def process_manifest(manifest_path: Path, output_dir: Path):
    """
    Parses the target manifest, streams notebook cells through the state machine,
    and serializes the resulting pedagogical payloads to the output directory.
    """
    if not manifest_path.exists():
        print(f"[!] Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, 'r', encoding='utf-8') as f:
        targets = [line.strip() for line in f if line.strip()]
        
    print(f"[*] Veteran Runner initialized. Processing {len(targets)} targets.")
    
    # Ensure the knowledge_complexes directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    total_payloads = 0
    
    for target in targets:
        target_path = Path(target)
        if not target_path.exists():
            print(f"[!] Target not found on disk: {target}")
            continue
            
        print(f"  -> Scanning: {target_path.name}")
        try:
            # 1. Ingestion Phase: Safely parse the Jupyter AST
            nb_node = load_notebook(str(target_path))
            cell_stream = stream_cells(nb_node)
            
            # 2. Cognitive Extraction Phase: Run the causal loop detector
            machine = VeteranStateMachine()
            payloads = machine.process_stream(cell_stream)
            
            # 3. Serialization Phase: Dump to intermediate knowledge complex
            if payloads:
                # Create a deterministic output filename bound to the source notebook
                out_name = f"{target_path.stem}_veteran_extract.json"
                out_file = output_dir / out_name
                
                # Convert Python Dataclasses to standard dictionaries for JSON writing
                serializable_payloads = [asdict(p) for p in payloads]
                
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_payloads, f, indent=2)
                    
                print(f"    [+] Extracted {len(payloads)} causal debugging loops. Saved to {out_name}")
                total_payloads += len(payloads)
            else:
                print(f"    [-] No complete veteran loops found in {target_path.name}")
                
        except Exception as e:
            print(f"    [!] Error processing {target_path.name}: {e}")
            
    print(f"[*] Veteran Runner complete. Total pedagogical payloads extracted: {total_payloads}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veteran Extraction Runner for Jupyter Logs")
    parser.add_argument("--manifest", required=True, help="Path to the targets.txt manifest file")
    parser.add_argument("--output-dir", required=True, help="Path to the knowledge_complexes/ output directory")
    
    args = parser.parse_args()
    
    process_manifest(Path(args.manifest), Path(args.output_dir))
