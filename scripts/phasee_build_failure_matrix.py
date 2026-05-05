import hashlib
import json
from pathlib import Path

src = Path("output/qlora_rejected_skills.jsonl")
out = Path("output/phasee_failure_matrix.jsonl")

rows = []
if src.exists():
    for line in src.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        node_id = str(obj.get("node_id", "UNKNOWN_NODE"))
        reason = str(obj.get("reason", "unknown_rejection"))
        failure_id = hashlib.sha256(f"{node_id}|{reason}".encode("utf-8")).hexdigest()[:16]

        rows.append(
            {
                "failure_id": failure_id,
                "node_id": node_id,
                "failure_type": "legacy_rejection",
                "rejection_reason": reason,
                "sie_snapshot": {"rho": 0.0, "s_sie": 0.0, "J_info": 0.0},
                "advocate_audit": {
                    "root_cause": "legacy_rejection",
                    "recommendation": "manual_review",
                },
                "trajectory": {
                    "trajectory_id": f"traj_{failure_id}",
                    "steps": [
                        {
                            "attempt_code": "",
                            "error": reason,
                            "diagnosis": "legacy_rejection",
                            "fix": "manual_review",
                        }
                    ],
                },
                "remediation_status": "abandoned",
            }
        )

rows = sorted(rows, key=lambda r: (r["node_id"], r["rejection_reason"], r["failure_id"]))
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")

print(f"FailureMatrixRows={len(rows)}")
print(f"FailureMatrixPath={out}")
