# Mode-Balance Downsampling Algorithm

## Goal

Cap Veteran failure rows during SFT compilation without deleting the source
failure matrix or hiding weak projections from Veteran training analysis.

This is distribution control at the dataset-export boundary. It is not semantic
filtering, and it does not change Layer 3 projection or Layer 4 validation.

## Drop-In Logic

The strict ChatML path now exposes this as:

```python
from src.pipeline.sft_formatter import downsample_mode_records

balanced = downsample_mode_records(
    records,
    target_mode="veteran",
    max_target_ratio=1.0,  # 1:1 veteran to non-veteran
    seed=42,
)
```

For a stricter 0.5:1 cap:

```python
balanced = downsample_mode_records(
    records,
    target_mode="veteran",
    max_target_ratio=0.5,
    seed=42,
)
```

The CLI form is:

```powershell
python src\pipeline\sft_formatter.py output -o output\qlora_skill_dataset_balanced.jsonl --max-veteran-ratio 1.0 --balance-seed 42
```

## Algorithm

```python
def downsample_mode_records(records, target_mode="veteran", max_target_ratio=1.0, seed=42):
    target_records = [r for r in records if r.get("_mode") == target_mode]
    non_target_records = [r for r in records if r.get("_mode") != target_mode]

    if not target_records or not non_target_records:
        return list(records)

    allowed = floor(len(non_target_records) * max_target_ratio)
    if len(target_records) <= allowed:
        return list(records)

    ordered_targets = sorted(target_records, key=stable_json_blob)
    selected_targets = random.Random(seed).sample(ordered_targets, allowed)
    return non_target_records + selected_targets
```

The caller should sort the returned records with the existing stable JSON key
before writing JSONL. `sft_formatter.format_complexes()` already does this.

## Placement Options

For `sft_formatter.py`, use the new optional `max_veteran_ratio` argument. The
default is `None`, so V2.0 byte-stable output is unchanged.

For legacy `dataset_formatter.py`, the equivalent insertion point is after
`apply_mode_balance_weights(all_accepted_items, process_id)` and before final
sampling. If the existing equal-count `balance_dataset()` path remains active,
the Veteran cap should replace that final sampler or run only when equal-count
balancing is disabled; otherwise the two sampling policies will stack.

## Recommended Defaults

- Use `max_veteran_ratio=1.0` for the first V2.1 pass.
- Use `max_veteran_ratio=0.5` only if token sizing shows failure traces are
  both dominant and long.
- Keep the failure matrix unmodified so no anti-pattern evidence is lost.
