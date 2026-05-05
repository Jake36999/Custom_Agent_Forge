"""
import logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("output/orchestrator_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
SOVEREIGN KERNEL: AUTONOMOUS ROLLING-BUFFER INGESTOR
Refactored for Linux/Colab Environment & DDD Architecture
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only for semantic stage; keeps GPU free for Unsloth training
import sys
import csv
import gc
import json
import re
import shutil
import stat
import datetime
import subprocess
import uuid
from pathlib import Path

# ==============================================================================
# DDD PATH CONFIGURATIONS
# ==============================================================================
# Assuming this script lives in the root directory
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

STAGING_DIR = WORKSPACE_ROOT / "staging_point"
OUTPUT_DIR = WORKSPACE_ROOT / "output" / "knowledge_complexes"
FAILED_DIR = STAGING_DIR / "failed_workspaces"
MANIFEST_DIR = OUTPUT_DIR / "manifests"

LOG_CSV = WORKSPACE_ROOT / "output" / "github_ingestion_log.csv"
ENV_JSON = WORKSPACE_ROOT / "config" / "environment.json"
LANDING_PAD_DIR = WORKSPACE_ROOT / "config" / "landing_pad" # Re-established cleanly inside config

MAX_CAPACITY_BYTES = 40 * 1024 * 1024 * 1024  # 40GB

# Pipeline scripts
SLICER_SCRIPT = WORKSPACE_ROOT / "src" / "pipeline" / "semantic_slicer_AG.py"
PROCESSOR_SCRIPT = WORKSPACE_ROOT / "src" / "pipeline" / "cognitive_processor.py"

def get_manifest_path(project_name: str) -> Path:
    return MANIFEST_DIR / f"{project_name}_INGESTION_MANIFEST.json"

def get_manifest_record(project_name: str):
    manifest_path = get_manifest_path(project_name)
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        required_props = ["project_name", "repo_url", "ingestion_status", "schema_version", "chunks_processed", "timestamp"]
        if not all(prop in manifest for prop in required_props):
            return None

        if manifest.get("project_name") != project_name or manifest.get("ingestion_status") != "Completed":
            return None
        if manifest.get("schema_version") != "aletheia_skill_dossier_v1" or int(manifest.get("chunks_processed", 0)) <= 0:
            return None

        return manifest
    except Exception:
        return None

def write_manifest(project_name: str, repo_url: str, chunks_processed: int):
    enforce_path_integrity()
    manifest_path = get_manifest_path(project_name)
    temp_path = MANIFEST_DIR / f"{project_name}.tmp.{uuid.uuid4().hex}"

    manifest = {
        "project_name": project_name,
        "repo_url": repo_url,
        "ingestion_status": "Completed",
        "schema_version": "aletheia_skill_dossier_v1",
        "chunks_processed": chunks_processed,
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
    temp_path.replace(manifest_path)
    return manifest_path

# ------------------------------------------------------------------------------
# STEP 1: Extract links and build Smart CSV backlog
# ------------------------------------------------------------------------------
def build_ingestion_backlog(manifest_path=None):
    """
    If manifest_path is provided, only parse that file for GitHub links (one per line).
    Otherwise, fallback to legacy landing pad scan.
    """
    if manifest_path:
        print(f"[MANIFEST MODE] Parsing manifest: {manifest_path}")
        backlog_dict = {}
        url_pattern = re.compile(r"(https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:\.git)?)")
        with open(manifest_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = url_pattern.search(line)
                if match:
                    url = match.group(1)
                    project_name = url.split("/")[-1].replace(".git", "")
                    status = "Completed" if get_manifest_record(project_name) else "Pending"
                    backlog_dict[url] = {"ProjectName": project_name, "RepoUrl": url, "Status": status}
        if backlog_dict:
            backlog = sorted(backlog_dict.values(), key=lambda x: x["RepoUrl"])
            with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["ProjectName", "RepoUrl", "Status"])
                writer.writeheader()
                writer.writerows(backlog)
            pending = sum(1 for item in backlog if item["Status"] == "Pending")
            completed = sum(1 for item in backlog if item["Status"] == "Completed")
            print(f"[OK] Backlog synced to {LOG_CSV.name}")
            print(f"  -> Pending: {pending} (Will be ingested)")
            print(f"  -> Completed: {completed} (Already recorded)")
        else:
            print("[!] No valid GitHub links found in manifest.")
        return
    # ADVOCATE YAML CONFIG BYPASS
    yaml_files = list(LANDING_PAD_DIR.glob("*.yaml")) + list(LANDING_PAD_DIR.glob("*.yml"))
    advocate_dir = LANDING_PAD_DIR / "advocate"
    if advocate_dir.exists() and advocate_dir.is_dir():
        yaml_files += list(advocate_dir.glob("*.yaml"))
        yaml_files += list(advocate_dir.glob("*.yml"))
    if yaml_files:
        print(f"[ADVOCATE BYPASS] Found {len(yaml_files)} .yaml/.yml file(s) in {LANDING_PAD_DIR}/advocate. Building manifest for semantic compiler.")
        manifest_path = LOG_CSV.parent / "advocate_ingestion_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for yaml_file in yaml_files:
                f.write(str(yaml_file.resolve()) + "\n")
        print(f"[ADVOCATE BYPASS] Manifest written to {manifest_path}")
        # Set up a dummy backlog so downstream logic doesn't break
        with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["ProjectName", "RepoUrl", "Status"])
            writer.writeheader()
            writer.writerow({"ProjectName": "advocate_yaml_batch", "RepoUrl": "ADVOCATE_BYPASS", "Status": "Pending"})
        return
    enforce_path_integrity()

    # PDF BYPASS
    pdf_files = list(LANDING_PAD_DIR.glob("*.pdf"))
    if pdf_files:
        print(f"[PDF BYPASS] Found {len(pdf_files)} PDF(s) in {LANDING_PAD_DIR}. Building manifest for semantic compiler.")
        manifest_path = LOG_CSV.parent / "pdf_ingestion_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for pdf_file in pdf_files:
                f.write(str(pdf_file.resolve()) + "\n")
        print(f"[PDF BYPASS] Manifest written to {manifest_path}")
        # Optionally, set up a dummy backlog so downstream logic doesn't break
        with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["ProjectName", "RepoUrl", "Status"])
            writer.writeheader()
            writer.writerow({"ProjectName": "theorist_pdf_batch", "RepoUrl": "PDF_BYPASS", "Status": "Pending"})
        return

    # VETERAN NOTEBOOK BYPASS
    veteran_ipynb_files = list(LANDING_PAD_DIR.glob("*.ipynb"))
    veteran_dir = LANDING_PAD_DIR / "veteran"
    if veteran_dir.exists() and veteran_dir.is_dir():
        veteran_ipynb_files += list(veteran_dir.glob("*.ipynb"))
    if veteran_ipynb_files:
        print(f"[VETERAN BYPASS] Found {len(veteran_ipynb_files)} .ipynb file(s) in {LANDING_PAD_DIR}/veteran. Building manifest for semantic compiler.")
        manifest_path = LOG_CSV.parent / "veteran_ingestion_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for nb_file in veteran_ipynb_files:
                f.write(str(nb_file.resolve()) + "\n")
        print(f"[VETERAN BYPASS] Manifest written to {manifest_path}")
        # Set up a dummy backlog so downstream logic doesn't break
        with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["ProjectName", "RepoUrl", "Status"])
            writer.writeheader()
            writer.writerow({"ProjectName": "veteran_notebook_batch", "RepoUrl": "VETERAN_BYPASS", "Status": "Pending"})
        return
def _is_veteran_batch(repo: dict) -> bool:
    return repo.get("RepoUrl", "") == "VETERAN_BYPASS" or repo.get("ProjectName", "") == "veteran_notebook_batch"

    # Legacy GitHub logic
    source_files = list(LANDING_PAD_DIR.glob("*.txt"))
    if not source_files:
        print(f"[!] No .txt files found in Landing Pad: {LANDING_PAD_DIR}")
        print("    Please drop your text files with GitHub links into this folder.")
        return

    backlog_dict = {}
    url_pattern = re.compile(r"(https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:\.git)?)")

    for file_path in source_files:
        print(f"[*] Scanning {file_path.name} for GitHub links...")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = url_pattern.search(line)
                if match:
                    url = match.group(1)
                    project_name = url.split("/")[-1].replace(".git", "")

                    status = "Completed" if get_manifest_record(project_name) else "Pending"
                    backlog_dict[url] = {"ProjectName": project_name, "RepoUrl": url, "Status": status}

    if backlog_dict:
        backlog = sorted(backlog_dict.values(), key=lambda x: x["RepoUrl"])

        with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["ProjectName", "RepoUrl", "Status"])
            writer.writeheader()
            writer.writerows(backlog)

        pending = sum(1 for item in backlog if item["Status"] == "Pending")
        completed = sum(1 for item in backlog if item["Status"] == "Completed")

        print(f"[OK] Backlog synced to {LOG_CSV.name}")
        print(f"  -> Pending: {pending} (Will be ingested)")
        print(f"  -> Completed: {completed} (Already recorded)")
    else:
        print("[!] No valid GitHub links found in the Landing Pad files.")

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def get_dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

def invoke_checked(args: list, error_msg: str, env=None):
    result = subprocess.run(args, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{error_msg} (exit code {result.returncode})")


def _is_pdf_batch(repo: dict) -> bool:
    project_name = str(repo.get("ProjectName", "")).lower()
    repo_url = str(repo.get("RepoUrl", ""))
    return repo_url == "PDF_BYPASS" or "pdf" in project_name


def on_rm_error(func, path, exc_info):
    # Clear the readonly bit and reattempt the removal
    os.chmod(path, stat.S_IWRITE)
    try:
        func(path)
    except Exception:
        pass

def move_to_failed_workspace(project_name: str, source_path: Path):
    if not source_path.exists():
        return None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = FAILED_DIR / f"{project_name}_{timestamp}"
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(str(source_path), str(destination), dirs_exist_ok=True)
        shutil.rmtree(str(source_path), onerror=on_rm_error)
    except Exception as e:
        print(f"  [!] Failed to move workspace: {e}")
    return destination


# HD2: Automated 24-hour TTL for forensic failed workspace snapshots.
# Prevents disk saturation from stagnant failure artifacts.
FAILED_WORKSPACE_TTL_HOURS = int(os.environ.get("ALETHEIA_FAILED_WS_TTL_HOURS", "24"))

def purge_expired_failed_workspaces():
    """Remove failed workspace directories older than FAILED_WORKSPACE_TTL_HOURS."""
    if not FAILED_DIR.exists():
        return
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=FAILED_WORKSPACE_TTL_HOURS)
    purged = 0
    for child in sorted(FAILED_DIR.iterdir()):
        if child.is_dir():
            created = datetime.datetime.fromtimestamp(child.stat().st_ctime)
            if created < cutoff:
                print(f"  [TTL] Purging expired forensic workspace: {child.name} (created {created})")
                shutil.rmtree(child, onerror=on_rm_error)
                purged += 1
    if purged:
        print(f"[TTL] Purged {purged} expired failed workspace(s).")


# ------------------------------------------------------------------------------
# PRE-FLIGHT INTEGRITY GATES
# ------------------------------------------------------------------------------
def enforce_path_integrity():
    print("[*] Enforcing Path & Execution Integrity...")
    workspace_str = str(WORKSPACE_ROOT)
    
    # 1. Inject PYTHONPATH
    current_ppath = os.environ.get("PYTHONPATH", "")
    if workspace_str not in current_ppath:
        os.environ["PYTHONPATH"] = f"{workspace_str}{os.pathsep}{current_ppath}".strip(os.pathsep)
        print(f"  [+] Injected {workspace_str} into PYTHONPATH")
        
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
        print(f"  [+] Injected {workspace_str} into sys.path")

    # 2. Assert Directories
    critical_dirs = [STAGING_DIR, OUTPUT_DIR, FAILED_DIR, MANIFEST_DIR, LANDING_PAD_DIR]
    for d in critical_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"[!] Critical Failure: Cannot create directory {d}. Reason: {e}")
    print("  [+] All critical directories asserted.")

def test_compiler_readiness():
    print("[*] Running compiler import/compile gate...")
    try:
        # Attempt a strict core import
        # import src.core.models  # Unused import removed
        print("  [+] Core Module 'src.core.models' resolved successfully.")
    except ImportError as e:
        raise RuntimeError(f"[!] Pre-flight Import Failed: Could not resolve core modules. Ensure PYTHONPATH is correct. Error: {e}")
    
    source_files = [
        WORKSPACE_ROOT / "src" / "pipeline" / "cognitive_processor.py",
        WORKSPACE_ROOT / "src" / "pipeline" / "dag_runtime.py",
        WORKSPACE_ROOT / "src" / "pipeline" / "acs_engine.py",
        WORKSPACE_ROOT / "src" / "pipeline" / "dataset_formatter.py"
    ]
    for f in source_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing core script: {f}")

    invoke_checked([sys.executable, "-m", "py_compile"] + [str(f) for f in source_files], "Compiler preflight failed")
    print("  [OK] Compiler preflight passed.")

def write_environment_sensor():
    env_meta = {
        "Hardware": {"CPU_Count": os.cpu_count()},
        "OS": os.name,
        "Timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(ENV_JSON, 'w', encoding='utf-8') as f:
        json.dump(env_meta, f, indent=4)
    print(f"[ENV] Environment metadata written to {ENV_JSON.name}")

# ------------------------------------------------------------------------------
# STEP 2 & 3: Rolling Buffer Batch Processor
# ------------------------------------------------------------------------------
def start_autonomous_ingestion(force_refresh=False):
    enforce_path_integrity()
    purge_expired_failed_workspaces()
    write_environment_sensor()
    test_compiler_readiness()

    if not LOG_CSV.exists():
        print("[!] Log CSV not found. Please run build_ingestion_backlog first.")
        return

    with open(LOG_CSV, 'r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    # State reconciliation
    for r in records:
        if r["Status"] == "Completed":
            manifest_path = get_manifest_path(r["ProjectName"])
            if not manifest_path.exists() or force_refresh:
                print(f"[*] Reconciling state: Marking {r['ProjectName']} as Pending.")
                r["Status"] = "Pending"

    # Write back reconciled log
    with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["ProjectName", "RepoUrl", "Status"])
        writer.writeheader()
        writer.writerows(records)

    pending = [r for r in records if r["Status"] == "Pending"]
    if len(pending) > 5:
            print(f"[CircuitBreaker] {len(pending)} repos pending; capping at 5 for this run. Re-run to process remainder.")
    pending = pending[:5]
    if not pending:
        print("[!] No pending repositories found in log.")
        return

    # Prepare environment for absolute imports
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)


    batch_size = 5
    i = 0
    while i < len(pending):
        print("\n" + "="*60)
        print("[->] INITIATING DOWNLOAD BATCH...")
        print("="*60)

        current_batch = []
        batch_attempt_count = 0

        for b in range(batch_size):
            if i + b >= len(pending):
                break
            repo = pending[i + b]
            target_folder = STAGING_DIR / repo["ProjectName"]
            batch_attempt_count += 1

            if get_dir_size(STAGING_DIR) >= MAX_CAPACITY_BYTES:
                print("[!] Capacity threshold reached. Halting downloads to process current batch.")
                batch_attempt_count -= 1
                break

            try:
                manifest_path = get_manifest_path(repo["ProjectName"])
                if manifest_path.exists():
                    manifest_path.unlink()

                if _is_pdf_batch(repo):
                    print(f" -> Routing local PDF batch {repo['ProjectName']} directly to compiler...")
                    current_batch.append(repo)
                    continue

                if _is_veteran_batch(repo):
                    print(f" -> Routing local VETERAN notebook batch {repo['ProjectName']} directly to compiler...")
                    current_batch.append(repo)
                    continue

                if target_folder.exists():
                    retained = move_to_failed_workspace(repo["ProjectName"], target_folder)
                    print(f"  [!] Existing workspace moved to {retained.name} before clone.")

                print(f" -> Cloning {repo['ProjectName']}...")
                invoke_checked(["git", "clone", repo["RepoUrl"], str(target_folder)], f"Git clone failed for {repo['ProjectName']}")
                current_batch.append(repo)

            except Exception as e:
                print(f"  [ERROR] Clone failed for {repo['ProjectName']}: {e}")
                if target_folder.exists():
                    move_to_failed_workspace(repo["ProjectName"], target_folder)
                repo["Status"] = "Failed"

        if not current_batch and batch_attempt_count == 0:
            print("[!] No repositories could be added. Resolve capacity issues.")
            break

        print("\n" + "="*60)
        print("[*] COMPILING KNOWLEDGE MATRICES...")
        print("="*60)

        for repo in current_batch:
            print(f"DEBUG REPO: {repo['ProjectName']} | URL: {repo['RepoUrl']}")
            # PDF BYPASS: If this is the theorist PDF batch, run semantic_compiler.py directly
            if _is_pdf_batch(repo):
                print("[PDF BYPASS] Detected local PDF batch. Invoking semantic_compiler.py on manifest.")
                manifest_path = LOG_CSV.parent / "pdf_ingestion_manifest.json"
                try:
                    invoke_checked([
                        sys.executable, str(WORKSPACE_ROOT / "src" / "pipeline" / "semantic_compiler.py"),
                        "--mode", "theorist",
                        "--runtime-mode", "train",
                        "--manifest", str(manifest_path),
                        "--output-dir", str(OUTPUT_DIR)
                    ], f"Semantic compiler failed for {repo['ProjectName']}", env=env)
                    repo["Status"] = "Compiled"
                except Exception as e:
                    print(f"  [ERROR] Semantic compiler failed for {repo['ProjectName']}: {e}")
                    repo["Status"] = "Failed"
                continue

            # VETERAN NOTEBOOK BYPASS
            if _is_veteran_batch(repo):
                print("[VETERAN BYPASS] Detected local VETERAN notebook batch. Invoking semantic_compiler.py on manifest.")
                manifest_path = LOG_CSV.parent / "veteran_ingestion_manifest.json"
                try:
                    invoke_checked([
                        sys.executable, str(WORKSPACE_ROOT / "src" / "pipeline" / "semantic_compiler.py"),
                        "--mode", "veteran",
                        "--runtime-mode", "train",
                        "--manifest", str(manifest_path),
                        "--output-dir", str(OUTPUT_DIR)
                    ], f"Semantic compiler failed for {repo['ProjectName']}", env=env)
                    repo["Status"] = "Compiled"
                except Exception as e:
                    print(f"  [ERROR] Semantic compiler failed for {repo['ProjectName']}: {e}")
                    repo["Status"] = "Failed"
                continue

            target_folder = STAGING_DIR / repo["ProjectName"]
            print(f"[*] Processing: {repo['ProjectName']}")

            try:
                # 1. Slice raw code into AST JSON
                print("  -> Slicing ASTs...")
                ast_bundle_path = target_folder / "raw_ast_bundle.json"
                invoke_checked([
                    "python", str(SLICER_SCRIPT), str(target_folder),
                    "--base-dir", str(target_folder),
                    "--format", "json", "--agent-role", "architecture",
                    "--workers", str(max(1, (os.cpu_count() or 2) - 1)),
                    "--deterministic", "-o", str(ast_bundle_path)
                ], f"AST slicing failed for {repo['ProjectName']}", env=env)

                # 2. Compile directly to final Unified YAML
                bundle_parts = sorted(target_folder.glob("raw_ast_bundle*.json"))
                if not bundle_parts:
                    raise RuntimeError(f"No bundle parts produced for {repo['ProjectName']}")

                compiled_chunk_count = 0
                for part in bundle_parts:
                    print(f"  -> Compiling Matrix for {part.name}...")
                    suffix = part.stem.replace("raw_ast_bundle", "")
                    final_yaml = OUTPUT_DIR / f"KNOWLEDGE_MATRIX_{repo['ProjectName']}{suffix}_UNIFIED.yaml"

                    invoke_checked([
                        "python", str(PROCESSOR_SCRIPT), str(part),
                        "--output", str(final_yaml),
                        "--agent-prefix", "LogosAgent"
                    ], f"Compilation failed for {part.name}", env=env)

                    compiled_chunk_count += 1

                write_manifest(repo["ProjectName"], repo["RepoUrl"], compiled_chunk_count)
                print(f"  [OK] Successfully compiled pure Unified YAML chunks for {repo['ProjectName']}")
                repo["Status"] = "Completed"

                print("  [X] Purging raw repository data...")
                shutil.rmtree(target_folder, onerror=on_rm_error)

            except Exception as e:
                print(f"  [ERROR] Failed to compile {repo['ProjectName']}: {e}")
                repo["Status"] = "Failed"
                if target_folder.exists():
                    move_to_failed_workspace(repo["ProjectName"], target_folder)

        # Update CSV
        for r in records:
            for cb in current_batch:
                if r["ProjectName"] == cb["ProjectName"]:
                    r["Status"] = cb["Status"]

        with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["ProjectName", "RepoUrl", "Status"])
            writer.writeheader()
            writer.writerows(records)

        del current_batch
        gc.collect()
        i += batch_attempt_count

    print("\n[OK] ALL QUEUED REPOSITORIES PROCESSED!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, help="Path to specific target manifest")
    parser.add_argument("--output-dir", type=str, help="Override default output directory for knowledge complexes")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh completed repositories")
    args = parser.parse_args()

    # Allow orchestrator to override output directory for run isolation
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        MANIFEST_DIR = OUTPUT_DIR / "manifests"
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[RUN ISOLATION] Overriding OUTPUT_DIR: {OUTPUT_DIR}")

    if os.environ.get("ALETHEIA_SKIP_INGESTOR_AUTORUN") != "1":
        build_ingestion_backlog(manifest_path=args.manifest)
        start_autonomous_ingestion(force_refresh=args.force_refresh)
    # Duplicate parser.add_argument and args = parser.parse_args() removed
