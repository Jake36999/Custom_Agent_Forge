from ruamel.yaml import YAML
import json
import os

# --- CONFIG ---
YAML_PATH = os.environ.get("VETERAN_YAML", "output/knowledge_complexes/KNOWLEDGE_MATRIX_UNIFIED.yaml")
JSONL_PATH = os.environ.get("VETERAN_JSONL", "output/datasets/veteran/extracted_veteran.jsonl")

yaml = YAML(typ="safe")

def stream_validated_nodes(yaml_path):
    """
    Extracts the validated_nodes block as a list and yields each node dict.
    Reads the YAML file line-by-line; peak memory is O(one node).
    """
    import re
    with open(yaml_path, "r", encoding="utf-8") as f:
        in_nodes = False
        node_lines = []
        indent = None
        for line in f:
            if not in_nodes:
                if line.strip() == "validated_nodes:":
                    in_nodes = True
                continue
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if re.match(r"^[a-zA-Z0-9_]+:", line):
                if node_lines:
                    node_yaml = ''.join(node_lines)
                    if not node_yaml.lstrip().startswith('- '):
                        node_yaml = '- ' + node_yaml.lstrip()
                    node_list = None
                    try:
                        node_list = yaml.load(node_yaml)
                        if isinstance(node_list, list) and node_list and isinstance(node_list[0], dict):
                            yield node_list[0]
                    except Exception as e:
                        print(f"[ERROR] Failed to parse node: {e}")
                    finally:
                        del node_yaml, node_list
                break
            if indent is None and line.lstrip().startswith('- '):
                indent = len(line) - len(line.lstrip())
            if indent is not None and line.startswith(' ' * indent + '- '):
                if node_lines:
                    node_yaml = ''.join(node_lines)
                    if not node_yaml.lstrip().startswith('- '):
                        node_yaml = '- ' + node_yaml.lstrip()
                    node_list = None
                    try:
                        node_list = yaml.load(node_yaml)
                        if isinstance(node_list, list) and node_list and isinstance(node_list[0], dict):
                            yield node_list[0]
                    except Exception as e:
                        print(f"[ERROR] Failed to parse node: {e}")
                    finally:
                        del node_yaml, node_list
                node_lines = [line[indent:]]
            else:
                node_lines.append(line[indent:] if indent is not None and line.startswith(' ' * indent) else line)
        if node_lines:
            node_yaml = ''.join(node_lines)
            if not node_yaml.lstrip().startswith('- '):
                node_yaml = '- ' + node_yaml.lstrip()
            node_list = None
            try:
                node_list = yaml.load(node_yaml)
                if isinstance(node_list, list) and node_list and isinstance(node_list[0], dict):
                    yield node_list[0]
            except Exception as e:
                print(f"[ERROR] Failed to parse node: {e}")
            finally:
                del node_yaml, node_list

def main():
    count = 0
    os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
    with open(JSONL_PATH, "w", encoding="utf-8") as out:
        for node in stream_validated_nodes(YAML_PATH):
            if isinstance(node, dict):
                code_snippet = node.get("code_snippet", "")
                # Advocate YAML configuration bridge
                if node.get("mode") == "advocate" or node.get("skill_type") == "architectural_advocacy" or node.get("source_type") == "yaml_configuration":
                    record = {
                        "mode": "advocate",
                        "skill_type": "architectural_advocacy",
                        "source_type": "yaml_configuration",
                        "name": node.get("name", f"Config_{node.get('node_id', 'Unknown')}") ,
                        "instruction": node.get("instruction", "Analyze the following configuration:"),
                        "code_snippet": node.get("code_snippet", ""),
                        "output": node.get("output", "")
                    }
                else:
                    record = {
                        "mode": "veteran",
                        "skill_type": "trajectory_correction",
                        "source_type": "jupyter_notebook",
                        "name": node.get("name", f"Trajectory_{node.get('node_id', 'Unknown')}") ,
                        "instruction": node.get("instruction", "Analyze the reasoning path, identify logical drift, and output a corrective trajectory."),
                        "code_snippet": code_snippet,
                        "output": code_snippet
                    }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"Wrote {count} nodes to {JSONL_PATH}")

if __name__ == "__main__":
    main()
