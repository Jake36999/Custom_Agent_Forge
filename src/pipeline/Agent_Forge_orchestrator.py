"""
Aletheia Master Orchestrator (v4.9 - Final Production Edition)
---------------------------------------------------------------------
Central control plane for the Knowledge Compiler Engine.
Manages the end-to-end workflow from landing pad ingestion to QLoRA training.

Upgrades in v4.9 (Finishing Touches):
- Active Checkpoint Resumption (Menu and CLI --resume functionality)
- Graceful Interrupt Handling (Ctrl+C safety)
- Subprocess Unicode Resiliency (errors='replace')
- Behavioral Bounding & Strict Statistical Enforcement
- Metrics-Enriched Lineage (manifest.json)
- True Run Isolation (No global paths)
"""

import os
import sys
import glob
import subprocess
import yaml
import json
import uuid
import argparse
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

# --- Configuration Paths ---
LANDING_PAD_DIR = os.path.join("config", "landing_pad")
CONFIG_DIR = "config"
RUNS_DIR = "runs"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 70)
    print(" 🧠 ALETHEIA KNOWLEDGE COMPILER - MASTER ORCHESTRATOR")
    print("=" * 70)

_CORPUS_DENYLIST_FILES = {
    "versioneer.py", "setup.py", "setup.cfg", "conftest.py",
    "_version.py", "version.py", "__version__.py",
}
_CORPUS_DENYLIST_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".tox", "dist", "build", "egg-info",
    "site-packages", ".eggs",
}

def scan_available_files(extensions: List[str], search_dir: str = LANDING_PAD_DIR) -> List[str]:
    """Scans the specified directory for files matching the given extensions."""
    found_files = []
    for ext in extensions:
        for path in glob.glob(f"{search_dir}/**/*{ext}", recursive=True):
            parts = set(Path(path).parts)
            if parts & _CORPUS_DENYLIST_DIRS:
                continue
            if Path(path).name in _CORPUS_DENYLIST_FILES:
                continue
            found_files.append(path)
    return found_files

def select_target_files(mode_name: str, extensions: List[str]) -> List[str]:
    """Interactive prompt for selecting targets."""
    # Map mode to subfolder for targeted scanning
    mode_to_subdir = {
        "Coding Assistant": "coding_assistant",
        "Theorist": "theorist",
        "Advocate": "advocate",
        "Veteran": "veteran"
    }
    subdir = mode_to_subdir.get(mode_name, "")
    search_dir = os.path.join(LANDING_PAD_DIR, subdir) if subdir else LANDING_PAD_DIR
    print(f"\n[*] Scanning for {mode_name} targets in '{search_dir}/'...")
    files = scan_available_files(extensions, search_dir=search_dir)
    
    if not files:
        print(f"[!] No suitable files found for {mode_name} ({', '.join(extensions)}).")
        return []

    print("\n--- Available Targets ---")
    for idx, file in enumerate(files):
        print(f"  [{idx + 1}] {file}")
    print(f"  [{len(files) + 1}] Select ALL files")
    print("-" * 25)
    
    choice = input("Select target(s) (e.g., '1', '1,3', or 'all'): ").strip().lower()
    if choice == 'all' or choice == str(len(files) + 1):
        return files
    
    selected_files = []
    try:
        indices = [int(i.strip()) - 1 for i in choice.split(',')]
        for i in indices:
            if 0 <= i < len(files):
                selected_files.append(files[i])
    except ValueError:
        print("[!] Invalid selection. Aborting.")
    
    return selected_files

def validate_identity_kernel(kernel_path: str) -> bool:
    """Statically validates the Identity Kernel schema before pipeline execution."""
    print(f"[*] Validating Identity Kernel: {kernel_path}")
    try:
        with open(kernel_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        required_fields = ["name", "tone", "core_directives"]
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            print(f"[!] Identity Kernel is missing required fields: {', '.join(missing)}")
            return False
            
        print("  [+] Kernel Schema Valid.")
        return True
    except Exception as e:
        print(f"[!] Failed to parse Identity Kernel: {e}")
        return False

def select_identity_kernel() -> Optional[str]:
    """Interactive prompt for selecting an identity YAML file."""
    print("\n[*] Scanning for Identity Kernels (.yaml) in config directories...")
    kernels = scan_available_files(['.yaml'], search_dir=CONFIG_DIR)
    
    if not kernels:
        print("[!] No Identity Kernels found. Proceeding without identity.")
        return None
        
    print("\n--- Available Identity Kernels ---")
    for idx, k in enumerate(kernels):
        print(f"  [{idx + 1}] {k}")
        
    choice = input("Select an identity kernel: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(kernels):
            kernel_path = kernels[idx]
            if validate_identity_kernel(kernel_path):
                return kernel_path
            else:
                print("[!] Kernel validation failed. Proceeding without identity.")
                return None
    except ValueError:
        pass
    
    print("[!] Invalid selection. Proceeding without identity.")
    return None

def run_cmd(cmd: List[str], log_file: Path, step_name: str) -> bool:
    """Executes a subprocess, captures output for observability, and validates success."""
    print(f"[*] Executing: {' '.join(cmd)}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*40}\n[{datetime.now()}] STEP: {step_name}\nCMD: {' '.join(cmd)}\n{'='*40}\n")
        
        try:
            # No timeout — ingestion of large repos can exceed 30+ min; errors='replace' for unicode safety
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=None, errors="replace")
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n" + result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            f.write(f"\n[CRITICAL FAILURE] Return Code: {e.returncode}\n")
            f.write("STDOUT:\n" + e.stdout)
            f.write("STDERR:\n" + e.stderr)
            print(f"[!] FAILED: {step_name}. Check logs at {log_file}")
            return False

def validate_contract(filepath: Path, step_name: str, expected_format: str = "jsonl", min_items: int = 1) -> bool:
    """Semantically validates that a pipeline step produced valid, structured output."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        print(f"[!] Contract Violation: {step_name} failed to produce {filepath.name} or file is empty.")
        return False
        
    try:
        if expected_format == "jsonl":
            count = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        json.loads(line)  # Enforce valid JSON structure
                        count += 1
            if count < min_items:
                print(f"[!] Contract Violation: {filepath.name} contains {count} items. Expected >= {min_items}.")
                return False
                
        elif expected_format == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)
                
    except json.JSONDecodeError:
        print(f"[!] Contract Violation: {filepath.name} contains invalid {expected_format}.")
        return False
        
    return True

def quality_gate_dataset(dataset_path: Path, mode: str = "coding_assistant") -> Tuple[bool, dict]:
    """
    Polymorphic Quality Gate: Validates both Legacy Alpaca and Phase 3 Conversational schemas.
    Single-mode pipelines (theorist/veteran/advocate) produce intentionally homogeneous datasets
    and are exempt from typology-diversity checks.
    """
    _SINGLE_MODE_EXEMPT = {"theorist", "veteran", "advocate"}
    skip_typology_check = mode in _SINGLE_MODE_EXEMPT

    print(f"\n[3.5/4] Dataset Quality Gate  ({dataset_path.name})")
    print(f"        Mode: {mode.upper()}{' | Typology check: SKIPPED (single-mode)' if skip_typology_check else ''}")
    typologies: Counter[str] = Counter()
    metrics = {
        "total_samples": 0, "valid_samples": 0, "behaviorally_rejected": 0,
        "parrot_failures": 0,
        "typology_distribution": {}, "dominant_typology_ratio": 0.0
    }

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                metrics["total_samples"] += 1

                # --- Schema Polymorphism Check ---
                is_alpaca = "instruction" in item and "output" in item
                is_conversational = "messages" in item and isinstance(item.get("messages"), list)

                if not (is_alpaca or is_conversational):
                    print("[!] Quality Gate Failed: Unrecognized schema. Missing 'instruction/output' or 'messages'.")
                    return False, metrics

                # --- Behavioral Text Extraction ---
                if is_alpaca:
                    instruction_text = str(item.get("instruction", "")).strip()
                    output_text = str(item.get("output", "")).strip()
                else:
                    messages = item["messages"]
                    instruction_text = " ".join(
                        [m.get("content", "") for m in messages if m.get("role") == "user"]
                    ).strip()
                    output_text = " ".join(
                        [m.get("content", "") for m in messages if m.get("role") == "assistant"]
                    ).strip()

                # --- Parrot Failure Detection ---
                # Reject samples where the assistant output is a verbatim match or substring
                # of the full non-assistant context. Threshold of 20 chars avoids false
                # positives on trivially short tokens.
                if output_text and len(output_text) > 20:
                    if is_conversational:
                        non_assistant_text = " ".join([
                            m.get("content", "") for m in item["messages"]
                            if m.get("role") != "assistant"
                        ]).strip()
                    else:
                        non_assistant_text = instruction_text
                    if output_text == non_assistant_text or output_text in non_assistant_text:
                        metrics["parrot_failures"] += 1
                        metrics["behaviorally_rejected"] += 1
                        continue

                # --- Behavioral Constraint Bounding ---
                if len(instruction_text) < 10 or len(output_text) < 5:
                    metrics["behaviorally_rejected"] += 1
                    continue

                typo = item.get("metadata", {}).get("typology", "unknown")
                typologies[typo] += 1
                metrics["valid_samples"] += 1

        if metrics["valid_samples"] == 0:
            print("[!] Quality Gate Failed: Dataset has 0 behaviorally valid samples.")
            return False, metrics

        rejection_rate = metrics["behaviorally_rejected"] / metrics["total_samples"]
        if rejection_rate > 0.10 and metrics["total_samples"] > 10:
            print(f"[!] Quality Gate Failed: High behavioral rejection rate ({rejection_rate*100:.1f}%).")
            return False, metrics

        dominant_count = max(typologies.values()) if typologies else 0
        metrics["dominant_typology_ratio"] = dominant_count / metrics["valid_samples"]

        print(f"        Valid: {metrics['valid_samples']}  |  Rejected: {metrics['behaviorally_rejected']}  |  Parrot: {metrics['parrot_failures']}")
        if typologies:
            print("        Typology Distribution:")

        for typo, count in typologies.items():
            ratio = count / metrics["valid_samples"]
            metrics["typology_distribution"][typo] = {"count": count, "ratio": round(ratio, 3)}
            print(f"          {typo:<25} {ratio*100:5.1f}%  ({count})")

        if metrics["valid_samples"] > 10 and not skip_typology_check:
            if len(typologies) < 2:
                print("[!] Quality Gate Failed: Severe mode collapse. Only 1 typology generated.")
                return False, metrics
            if metrics["dominant_typology_ratio"] > 0.85:
                print(
                    f"[!] Quality Gate Failed: Mode collapse detected. "
                    f"Dominant typology exceeds 85% threshold ({metrics['dominant_typology_ratio']*100:.1f}%)."
                )
                return False, metrics

        if metrics["parrot_failures"] > 0:
            print(f"  [!] {metrics['parrot_failures']} parrot_failure(s) dropped from valid samples.")

        return True, metrics

    except Exception as e:
        print(f"[!] Quality Gate Exception: {e}")
        return False, metrics

def save_manifest(run_dir: Path, manifest: dict):
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def run_pipeline(mode: str, targets: Optional[List[str]] = None, identity_kernel: Optional[str] = None, auto_train: bool = False, run_id: Optional[str] = None):
    """Executes the run-isolated pipeline dynamically based on the Triad Mode."""
    is_resume = bool(run_id)
    if not run_id:
        run_id = uuid.uuid4().hex[:8]
        
    run_dir = Path(RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "pipeline.log"
    
    # Initialize or Load Lineage Manifest
    if is_resume:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            # Inherit values from the previous run if not explicitly overridden
            mode = manifest.get("mode", mode)
            targets = manifest.get("targets", targets) or targets
            identity_kernel = manifest.get("identity_kernel", identity_kernel)
        else:
            print(f"[!] Run manifest not found for {run_id}. Cannot resume.")
            return
    else:
        manifest = {
            "run_id": run_id,
            "mode": mode,
            "targets": targets or [],
            "identity_kernel": identity_kernel,
            "timestamp_start": datetime.now().isoformat(),
            "status": "running",
            "artifacts": {},
            "quality_metrics": {}
        }
    
    # Reset status for current execution
    manifest["status"] = "running"
    save_manifest(run_dir, manifest)
    
    print("\n" + "=" * 70)
    print(f"🚀 INITIATING PIPELINE: {mode.upper()}")
    print(f"🆔 Run ID: {run_id}{' (RESUMING)' if is_resume else ''}")
    print(f"📁 Run Directory: {run_dir}")
    print(f"🎯 Targets: {len(targets) if targets else 0} files selected")
    if identity_kernel:
        print(f"🎭 Identity: {identity_kernel}")
    print("=" * 70 + "\n")

    try:
        # --- STEP 1: TARGET MANIFEST ---
        step_1_flag = run_dir / "step_1_complete.flag"
        target_list_path = run_dir / "targets.txt"
        
        if not step_1_flag.exists() and targets:
            print("[1/4] Preparing Run Targets...")
            with open(target_list_path, 'w', encoding="utf-8") as f:
                for t in targets:
                    f.write(t + "\n")
            manifest["artifacts"]["target_list"] = str(target_list_path)
            step_1_flag.touch()
            save_manifest(run_dir, manifest)
        else:
            print("[1/4] Skipping Run Targets Preparation (Resuming from checkpoint)...")

        # --- STEP 2: ENGINE ROUTING (The Triad Architecture Logic) ---
        step_2_flag = run_dir / "step_2_complete.flag"
        raw_ast = run_dir / "raw_ast_bundle.json"
        complex_dir = run_dir / "knowledge_complexes"
        
        if not step_2_flag.exists():
            print(f"\n[2/4] Engaging Core Extraction Engine for {mode.upper()}...")
            
            # --- NEW VETERAN PIPELINE ROUTING ---
            if mode == "veteran":
                print(f"  [+] Mounting Jupyter Trace Ingestor & Veteran State Machine...")
                # Here we call the script that wraps jupyter_trace_ingestor.py and veteran_state_machine.py
                engine_cmd = [
                    sys.executable, 
                    "src/pipeline/veteran_extraction_runner.py", # Future runner script for the state machine
                    "--manifest", str(target_list_path),
                    "--output-dir", str(complex_dir)
                ]
                if not run_cmd(engine_cmd, log_file, "Veteran Diagnostic Extraction"): 
                    manifest["status"] = "failed_at_veteran_extraction"
                    return

            # --- NEW ADVOCATE PIPELINE ROUTING ---
            elif mode == "advocate":
                # We can dynamically pass the tuning metrics straight from the config to the sub-process
                with open("config/engine_config.yaml", "r") as cfg:
                    _cfg = yaml.safe_load(cfg)
                    ratio = _cfg.get("advocate_tuning", {}).get("text_to_code_ratio", 0.5)
                
                print(f"  [+] Mounting Advocate Extractor (Target Text-to-Code Ratio: {ratio})...")
                engine_cmd = [
                    sys.executable, 
                    "src/pipeline/advocate_extraction_runner.py", 
                    "--manifest", str(target_list_path),
                    "--ratio", str(ratio),
                    "--output-dir", str(complex_dir)
                ]
                if not run_cmd(engine_cmd, log_file, "Advocate Theoretical Extraction"): 
                    manifest["status"] = "failed_at_advocate_extraction"
                    return

            # --- NEW HYBRID PIPELINE ROUTING ---
            elif mode == "hybrid":
                print(f"  [+] Mounting Hybrid Extraction (Veteran + Advocate)...")
                hybrid_cmd = [
                    sys.executable, 
                    "src/pipeline/hybrid_extraction_runner.py", 
                    "--manifest", str(target_list_path),
                    "--output-dir", str(complex_dir)
                ]
                if not run_cmd(hybrid_cmd, log_file, "Hybrid Complete Extraction"): 
                    manifest["status"] = "failed_at_hybrid_extraction"
                    return

            # --- ORIGINAL BASELINE ROUTING ---
            elif mode == "coding_assistant":
                # For Coding Assistant, pass the actual .txt file(s) with GitHub links to the ingestor
                # If multiple targets, process each sequentially (or could be batched if desired)
                for manifest_file in (targets or []):
                    ingestor_cmd = [
                        sys.executable, "scripts/autonomous_ingestor.py",
                        "--manifest", manifest_file,
                        "--output-dir", str(complex_dir)
                    ]
                    if not run_cmd(ingestor_cmd, log_file, f"Autonomous Ingestor ({manifest_file})"): 
                        manifest["status"] = f"failed_at_autonomous_ingestor_{os.path.basename(manifest_file)}"
                        return

            elif mode == "theorist":
                engine_cmd = [
                    sys.executable, 
                    "src/pipeline/semantic_compiler.py",
                    f"--mode={mode}",
                    "--runtime-mode=train",
                    "--manifest", str(target_list_path),
                    "--output-dir", str(complex_dir)
                ]
                if not run_cmd(engine_cmd, log_file, "Advanced Semantic Compiler"): 
                    manifest["status"] = "failed_at_semantic_compiler"
                    return
                
            manifest["artifacts"]["knowledge_complexes_dir"] = str(complex_dir)
            step_2_flag.touch()
            save_manifest(run_dir, manifest)
        else:
            print(f"\n[2/4] Skipping Core Extraction Engine (Resuming from checkpoint)...")

        # --- STEP 2.5: DAG SCORING PASS (SIE + ACS + Unified Confidence) ---
        step_2_5_flag = run_dir / "step_2_5_complete.flag"
        
        if not step_2_5_flag.exists() and mode != "theorist":
            # Theorist mode runs DAG internally via semantic_compiler.py — skip for it
            print(f"\n[2.5/4] Running DAG Scoring Pass (SIE + ACS + Constraint Propagation)...")
            dag_pass_cmd = [
                sys.executable,
                "src/pipeline/dag_scoring_pass.py",
                str(complex_dir),
                "--output-dir", str(complex_dir),
                "--mode", mode,
            ]
            if not run_cmd(dag_pass_cmd, log_file, "DAG Scoring Pass"):
                manifest["status"] = "failed_at_dag_scoring"
                save_manifest(run_dir, manifest)
                return
            manifest["dag_scoring"] = "completed"
            step_2_5_flag.touch()
            save_manifest(run_dir, manifest)
        elif mode == "theorist":
            print(f"\n[2.5/4] Skipping DAG Scoring Pass (DAG runs inside semantic_compiler for theorist mode)...")
        else:
            print(f"\n[2.5/4] Skipping DAG Scoring Pass (Resuming from checkpoint)...")

        # --- STEP 3: DATASET FORMATTER ---
        step_3_flag = run_dir / "step_3_complete.flag"
        dataset_output = run_dir / f"qlora_{mode}_dataset.jsonl"

        if not step_3_flag.exists():
            print("\n[3/4] Formatting Dataset...")

            if mode == "theorist":
                # Theorist pipeline: bridge DAG output → RAL amplification → SIE formatter
                extracted_path = Path("output/datasets/theorist/extracted_theorist.jsonl")
                amplified_dir  = Path("output/datasets/theorist_amplified")

                print("  [3a] Bridging DAG output to extraction JSONL...")
                if not run_cmd([sys.executable, "scripts/bridge_theorist.py"], log_file, "Bridge Theorist"):
                    manifest["status"] = "failed_at_bridge"
                    return

                print("  [3b] RAL Amplification via OpenAI API...")
                ral_cmd = [
                    sys.executable, "scripts/ral_amplifier.py",
                    "--input",  str(extracted_path),
                    "--output", str(amplified_dir / "amplified_theorist.jsonl"),
                ]
                if not run_cmd(ral_cmd, log_file, "RAL Amplifier"):
                    manifest["status"] = "failed_at_ral"
                    return

                print("  [3c] SIE Formatting (ChatML + lens tagging)...")
                sft_cmd = [
                    sys.executable, "-m", "src.pipeline.sft_formatter",
                    str(amplified_dir),
                    "-o", str(dataset_output),
                    "--apply-loss-masking",
                ]
                if not run_cmd(sft_cmd, log_file, "SIE Formatter"):
                    manifest["status"] = "failed_at_sft_formatter"
                    return

            else:
                # Standard pipeline: legacy dataset_formatter.py
                with open("config/engine_config.yaml", "r") as cfg:
                    format_cfg = yaml.safe_load(cfg).get("dataset_formatting", {})

                formatter_cmd = [
                    sys.executable,
                    "src/pipeline/dataset_formatter.py",
                    str(complex_dir),
                    "-o", str(dataset_output),
                    "--mode", mode,
                ]
                if format_cfg.get("apply_loss_masking"):
                    formatter_cmd.append("--apply-loss-masking")
                if format_cfg.get("train_split"):
                    formatter_cmd.extend(["--split", str(format_cfg.get("train_split"))])
                if identity_kernel:
                    formatter_cmd.extend(["--identity", identity_kernel])

                if not run_cmd(formatter_cmd, log_file, "Dataset Formatter"):
                    manifest["status"] = "failed_at_formatter"
                    return

            if not validate_contract(dataset_output, "Dataset Formatter", expected_format="jsonl"):
                manifest["status"] = "failed_contract_formatter"
                return

            manifest["artifacts"]["qlora_dataset"] = str(dataset_output)
            step_3_flag.touch()
            save_manifest(run_dir, manifest)
        else:
            print("\n[3/4] Skipping Dataset Formatting (Resuming from checkpoint)...")

        # --- STEP 3.5: DATASET QUALITY GATE ---
        step_3_5_flag = run_dir / "step_3_5_complete.flag"
        
        if not step_3_5_flag.exists():
            gate_passed, gate_metrics = quality_gate_dataset(dataset_output, mode=mode)
            manifest["quality_metrics"] = gate_metrics  # Inject metrics into lineage
            
            if not gate_passed:
                print("\n[!] Pipeline halted due to Dataset Quality Gate failure.")
                manifest["status"] = "failed_quality_gate"
                return
                
            step_3_5_flag.touch()
            save_manifest(run_dir, manifest)
        else:
             print("\n[3.5/4] Skipping Dataset Quality Gate (Resuming from checkpoint)...")

        # --- STEP 4: MODEL TRAINING HANDOFF ---
        _MODE_METRICS = {
            "theorist":         ("Causal Axioms Extracted",        "valid_samples"),
            "veteran":          ("Trajectories/Recoveries Parsed", "valid_samples"),
            "advocate":         ("Alignment Constraints Enforced", "valid_samples"),
            "coding_assistant": ("API Patterns Compiled",          "valid_samples"),
        }
        _metric_label, _metric_key = _MODE_METRICS.get(mode, ("Knowledge Units Compiled", "valid_samples"))
        _metric_value = manifest.get("quality_metrics", {}).get(_metric_key, "N/A")
        _total        = manifest.get("quality_metrics", {}).get("total_samples", "N/A")
        _rejected     = manifest.get("quality_metrics", {}).get("behaviorally_rejected", 0)

        print("\n" + "=" * 70)
        print(f"  ✅  PIPELINE SUCCESS — MODE: {mode.upper()}")
        print(f"  Run ID : {run_id}")
        print(f"  Output : {dataset_output.name}")
        print("-" * 70)
        print(f"  {_metric_label:<38}: {_metric_value}")
        print(f"  {'Total Nodes Processed':<38}: {_total}")
        print(f"  {'Behaviorally Rejected':<38}: {_rejected}")
        print("=" * 70)
        print(f"\n[4/4] Handoff to Training Engine...")
        print(f"[*] Success! Validated Dataset isolated at: {dataset_output}")
        
        manifest["status"] = "completed"
        save_manifest(run_dir, manifest)
        
        if auto_train:
            train_choice = 'y'
        else:
            try:
                train_choice = input("\nData generation complete. Deploy training phase now? [y/n]: ").strip().lower()
            except EOFError:
                train_choice = 'n'

        if train_choice == 'y':
            train_script = "scripts/colab_theorist_Train.py"
            if os.path.exists(train_script):
                print("\n[TRAINING] Launching training pipeline...")
                expected_model_path = run_dir / "model_export.gguf"
                train_cmd = [
                    sys.executable, train_script,
                    "--dataset", str(dataset_output),
                    "--output",  str(expected_model_path),
                ]
                if not run_cmd(train_cmd, log_file, "Training"):
                    manifest["training_status"] = "failed"
                    save_manifest(run_dir, manifest)
                else:
                    print("\n[TRAINING COMPLETE]")
                    if expected_model_path.exists():
                        manifest["artifacts"]["model"] = str(expected_model_path)
            else:
                print(f"[!] Training script not found locally. Upload {dataset_output.name} to Colab to train.")
        else:
            print(f"\n[*] Training skipped. Dataset ready at: {dataset_output}")

    except Exception as e:
        print(f"\n[!] Unexpected Pipeline Crash: {e}")
        manifest["status"] = "crashed"

    finally:
        # Guarantee manifest finalization regardless of mid-step return or crash
        save_manifest(run_dir, manifest)

def resume_interactive_run():
    """Lists incomplete runs and allows the user to resume one."""
    runs_path = Path(RUNS_DIR)
    if not runs_path.exists():
        print("\n[!] No runs directory found. Nothing to resume.")
        input("Press Enter to return...")
        return
        
    resumable = []
    for run_dir in runs_path.iterdir():
        if run_dir.is_dir():
            manifest_file = run_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        man = json.load(f)
                        if man.get("status") not in ["completed"]:
                            resumable.append(man)
                except Exception:
                    continue
                    
    if not resumable:
        print("\n[*] No failed, incomplete, or crashed runs found.")
        input("Press Enter to return...")
        return
        
    print("\n--- Resumable Runs ---")
    # Sort by timestamp descending
    resumable.sort(key=lambda x: x.get("timestamp_start", ""), reverse=True)
    for idx, man in enumerate(resumable):
        print(f"  [{idx + 1}] ID: {man.get('run_id')} | Mode: {man.get('mode')} | Status: {man.get('status')}")
        
    choice = input("\nSelect a run to resume (or 0 to cancel): ").strip()
    try:
        if choice == '0':
            return
        idx = int(choice) - 1
        if 0 <= idx < len(resumable):
            selected = resumable[idx]
            run_pipeline(
                mode=selected.get('mode', 'coding_assistant'),
                targets=selected.get('targets'), 
                identity_kernel=selected.get('identity_kernel'),
                run_id=selected.get('run_id')
            )
            input("\nPress Enter to return to menu...")
    except ValueError:
        print("[!] Invalid selection.")
        input("Press Enter to return...")

def interactive_mode():
    """Fallback interactive CLI for manual operation."""
    while True:
        clear_screen()
        print_header()
        print("\nPlease select an Orchestrator Mode:\n")
        print("  [1] 💻 Coding Assistant")
        print("      Produces QLoRA flashcards from repos/code via AST Slicer.")
        print("\n  [2] 📚 Theorist")
        print("      Produces flashcards via Semantic Chunker (Train Mode).")
        print("\n  [3] 🎭 Advocate")
        print("      Trained on code + text + Identity Kernel.")
        print("\n  [4] 🔄 Resume Failed/Halted Run")
        print("      Pick up an incomplete pipeline execution from its last checkpoint.")
        print("\n  [0] Exit")
        print("-" * 70)
        
        choice = input("Select Mode (0-4): ").strip()
        
        if choice == '0':
            print("Exiting Orchestrator...")
            break
            
        elif choice == '1':
            # Coding Assistant: scan only coding_assistant/ for .txt manifest files
            targets = select_target_files("Coding Assistant", ['.txt'])
            if targets:
                run_pipeline("coding_assistant", targets)
            input("\nPress Enter to return to menu...")
        
        elif choice == '2':
            # Theorist: scan only theorist/ for text/semantic files
            targets = select_target_files("Theorist", ['.txt', '.md', '.yaml', '.csv', '.pdf'])
            if targets:
                run_pipeline("theorist", targets)
            input("\nPress Enter to return to menu...")
            
        elif choice == '3':
            targets = select_target_files("Advocate", ['.txt', '.md', '.py', '.js', '.yaml'])
            if targets:
                identity = select_identity_kernel()
                run_pipeline("advocate", targets, identity_kernel=identity)
            input("\nPress Enter to return to menu...")
            
        elif choice == '4':
            resume_interactive_run()
            
        else:
            print("Invalid selection.")
            input("Press Enter to continue...")

def main():
    clear_screen()
    print_header()
    
    # --- 1. Headless CLI Automation (Restored) ---
    parser = argparse.ArgumentParser(description="Aletheia Master Orchestrator (v4.9)")
    parser.add_argument("--mode", choices=["coding_assistant", "theorist", "advocate", "veteran", "hybrid"], help="Target orchestration mode")
    parser.add_argument("--targets", help="Comma-separated paths to target files, or 'all'")
    parser.add_argument("--identity", help="Path to Identity Kernel YAML (for advocate mode)")
    parser.add_argument("--auto-train", action="store_true", help="Automatically trigger Unsloth script on success")
    parser.add_argument("--resume", help="Resume a halted pipeline by providing its Run ID")
    
    # If arguments are passed, run in headless mode instead of relying on the bus
    if len(sys.argv) > 1:
        args = parser.parse_args()
        
        # Check if resuming an existing run via CLI
        if args.resume:
            print(f"[*] Resuming Run ID: {args.resume} in headless mode...")
            run_pipeline(mode="unknown", run_id=args.resume, auto_train=args.auto_train)
            sys.exit(0)
            
        if not args.mode:
            print("[!] Headless mode requires --mode to be specified unless --resume is used.")
            sys.exit(1)
            
        if args.identity and not validate_identity_kernel(args.identity):
            sys.exit(1)
            
        targets = []
        if args.targets:
            if args.targets.lower() == 'all':
                extensions = ['.py', '.txt', '.md', '.js', '.ipynb', '.yaml', '.csv']
                targets = scan_available_files(extensions)
            else:
                raw_targets = [t.strip() for t in args.targets.split(',')]
                # Pre-flight check: Ensure all explicitly passed targets exist
                for t in raw_targets:
                    if not Path(t).exists():
                        print(f"[!] Invalid target specified in headless mode: {t}")
                        sys.exit(1)
                    targets.append(t)
        
        if not targets:
            print("[!] No targets specified or found. Use --targets.")
            sys.exit(1)
            
        run_pipeline(args.mode, targets, identity_kernel=args.identity, auto_train=args.auto_train)
        return

    # --- 2. Centralized Config Bus Routing (Fallback if no args) ---
    config_path = Path("config/engine_config.yaml")
    
    if not config_path.exists():
        print(f"[!] Centralized config not found at {config_path}. Defaulting to Interactive TUI.")
        interactive_mode()
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        engine_config = yaml.safe_load(f)

    run_settings = engine_config.get("run_settings", {})
    active_mode = run_settings.get("active_mode", "baseline")
    target_directory = run_settings.get("target_directory", "landing_pad/past_sessions")
    
    print(f"[*] Engine Config Loaded Successfully.")
    print(f"[*] Active Mode: [{active_mode.upper()}]")
    print(f"[*] Target Directory: {target_directory}")

    # Dynamic Target Routing based on Mode
    if active_mode in ['veteran', 'advocate', 'hybrid']:
        extensions = ['.ipynb'] # These modes strictly hunt Jupyter traces
    elif active_mode == 'theorist':
        extensions = ['.pdf', '.txt', '.md', '.yaml', '.csv']
    else:
        extensions = ['.py', '.txt', '.md', '.yaml', '.csv'] # Baseline modes
        
    print(f"[*] Scanning for {extensions} targets in '{target_directory}'...")
    targets = scan_available_files(extensions, search_dir=target_directory)

    if not targets:
        print(f"[!] No valid targets found in {target_directory}. Aborting.")
        sys.exit(1)

    # Fire the pipeline directly from the config parameters
    run_pipeline(mode=active_mode, targets=targets)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Orchestrator interrupted by user (Ctrl+C). Exiting gracefully.")
        sys.exit(0)
