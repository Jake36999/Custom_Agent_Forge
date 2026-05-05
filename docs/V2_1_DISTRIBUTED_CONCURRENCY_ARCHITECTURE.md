# V2.1 Distributed Concurrency Architecture

## Decision

Use per-worker ledger shards plus a deterministic post-run merge. Do not put a
Redis mutex around every state transition.

The ledger is on the hot path: `DAGRuntime._log_transition()` can append
thousands of records per run, and a global distributed lock would serialize the
fastest part of the Celery worker fleet. V2.1 should keep transition logging
local inside each runtime and use Redis only for coarse-grained coordination:
shard claims, idempotency, and the single final merge commit.

## Why This Fits The Five Layers

Layer 3 remains the semantic projection layer. Workers may extract and score
signals by mode, but they do not reinterpret ledger integrity.

Layer 4 remains strict. Each worker validates its own local state-transition
chain with `verify_transition_ledger()` before returning a result to the chord
callback.

Layer 5 performs controlled artifact synthesis. The callback merges accepted
shards into a byte-stable run ledger, and rejected or weak projections remain
available for failure and Veteran datasets. Failed shard validation should mark
the shard `execution_eligible=False` and route it into the failure matrix rather
than silently dropping its data.

## Worker Contract

Each mode task returns the current payload plus a ledger shard:

```json
{
  "run_id": "PIPE-123",
  "mode": "veteran",
  "shard_id": "PIPE-123:veteran:0",
  "worker_id": "celery-host-3",
  "ledger_valid": true,
  "ledger_root": "sha256:last_transition_hash",
  "transition_ledger": [],
  "scored_nodes": [],
  "rejected_traces": []
}
```

Shard identity should be deterministic where possible:

```text
shard_id = sha256(run_id | mode | sorted_input_node_ids | task_index)[:16]
```

## Redis Usage

Use Redis for control-plane keys, not transition-plane writes:

```text
SET run:{run_id}:shard:{shard_id}:claim {worker_id} NX EX 3600
SET run:{run_id}:merge:claim {callback_id} NX EX 3600
HSET run:{run_id}:shards {shard_id} {ledger_root}
SET run:{run_id}:final_ledger_root {global_root} NX
```

The shard claim prevents duplicate work when a task is retried. The merge claim
ensures only one chord callback publishes the final ledger artifact. TTLs should
be long enough for the maximum expected worker runtime and refreshed by a
heartbeat if V2.1 introduces very large shards.

## Merge Algorithm

1. Collect all shard payloads from the Celery chord callback.
2. Reject no data. For any shard with `ledger_valid=false`, mark all records from
   that shard with `execution_eligible=False`, record the shard failure, and keep
   the rows for the failure matrix.
3. Sort valid shards by `(mode_order, shard_id, ledger_root)` where mode order is
   `theorist`, `coding_assistant`, `advocate`, `veteran`.
4. Build a super-ledger where each shard receives a merge record:

```text
global_hash_n = sha256(global_hash_n-1 | shard_id | ledger_root | row_count)
```

5. Write `output/ledger_{project_name}_{run_id}.jsonl` with stable JSON lines.
6. Persist `global_root` and merge metadata once via Redis `SET ... NX`.

## Why Not A Global Mutex

A Redis mutex around every transition gives a simple total order, but it makes
the ledger the throughput bottleneck. It also couples task liveness to the
network on every state mutation. A single lost lock renewal could make an
otherwise healthy worker look corrupt.

Shard-then-merge preserves tamper evidence without flattening the worker fleet.
Each local chain proves intra-worker ordering, and the super-ledger proves the
merge order of shard roots. That gives the cryptographic property V2.0 needs
while keeping V2.1 concurrency open.

## Failure Semantics

The V2 containment rule still applies:

- Do not silently drop a shard because a lock expired or a ledger root failed.
- Do not raise an unrecoverable global failure for one weak projection.
- Mark affected rows `execution_eligible=False`.
- Route the shard diagnostics to the failure matrix and Veteran anti-pattern
  stream.
- Keep accepted rows from valid shards eligible for the primary training matrix.

## Open Implementation Notes

- Add `ledger_shard.py` or a small serializer in `src/pipeline/dag_runtime.py`
  only after V2.0 is frozen.
- Teach `src/tasks/mode_tasks.py` to include `transition_ledger`,
  `ledger_valid`, and `ledger_root` in task results.
- Teach `src/tasks/pipeline_workflow.py::collect_and_format()` to perform the
  deterministic merge and write the global ledger artifact.
- Add retry tests for duplicate shard claims and a callback replay test for the
  merge lock.
