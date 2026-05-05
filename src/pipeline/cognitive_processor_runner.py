
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import json
from pathlib import Path
from src.pipeline.cognitive_processor import CognitiveProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processor = CognitiveProcessor(data)
    nodes = processor.compile()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'compiled_nodes.json'
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump([nodes], f, indent=2)
    print(f'[OK] Compiled nodes saved to {out_file}')
