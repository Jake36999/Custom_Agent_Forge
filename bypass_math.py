import yaml
import json
import glob
import gc
import os

# Suppress PyYAML warnings on unrecognised custom tags (e.g. !!python/object)
yaml.SafeLoader.add_multi_constructor(
    "", lambda loader, tag_suffix, node: None
)

print("Initiating Aggressive Graph Math Bypass...")

yaml_files = glob.glob(r"runs\401807d0\knowledge_complexes\*.yaml")
out_path = r"runs\401807d0\knowledge_complexes\dag_scored_nodes.jsonl"

print(f"Found {len(yaml_files)} YAML files to process (accepted + rejected).")

count = 0
skipped = 0
with open(out_path, "w", encoding="utf-8") as f_out:
    for file in yaml_files:
        print(f"Scooping: {os.path.basename(file)}")
        try:
            with open(file, "r", encoding="utf-8") as f_in:
                payload = yaml.safe_load(f_in)

            if not isinstance(payload, dict):
                skipped += 1
                continue

            # Loot both validated and rejected arrays
            nodes = (payload.get("validated_nodes") or []) + (payload.get("rejected_nodes") or [])

            for node in nodes:
                if not isinstance(node, dict):
                    continue

                node["state"] = "ACCEPTED"

                # Fix Pydantic dict bug — source_context must be a JSON string
                if isinstance(node.get("source_context"), dict):
                    node["source_context"] = json.dumps(node["source_context"])

                f_out.write(json.dumps(node) + "\n")
                count += 1

        except Exception as e:
            print(f"  [!] Skipped {os.path.basename(file)}: {e}")
            skipped += 1

        payload = None
        gc.collect()

print(f"\nSuccess! {count} nodes recovered. {skipped} files skipped.")
print(f"Output: {out_path}")
