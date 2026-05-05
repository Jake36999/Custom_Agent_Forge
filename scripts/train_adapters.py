import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
config_files = sorted(CONFIG_DIR.glob("axolotl_*.yml"))

for config in config_files:
    print(f"\n[TRAINING] Starting adapter: {config.name}\n{'='*60}")
    result = subprocess.run([
        "accelerate", "launch", "-m", "axolotl.cli.train", str(config)
    ])
    if result.returncode != 0:
        print(f"[ERROR] Training failed for {config.name}. Halting pipeline.")
        sys.exit(result.returncode)
    print(f"[SUCCESS] Completed adapter: {config.name}\n{'-'*60}")
print("\n[ALL DONE] All adapters trained successfully.")
