# SPDX-FileCopyrightText: 2026 Xintong Yang
# SPDX-License-Identifier: Apache-2.0

"""Recompute the sequence-length sweep from complete, unmodified JSONL logs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SIZES = ("4b", "8b")
LENGTHS = (8192, 16384, 32768, 65536, 131072, 262144)
BATCHES = (1, 4, 16, 32)
METHODS = ("full", "indexmem")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12)


def parse_result(path, key, raw, model_config):
    size, length, batch, method = key
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert events, path
    assert events[-1]["event"] in ("complete", "failure") or not any(e["event"] == "complete" for e in events)
    for e in events:
        assert (e["size"], e["length"], e["batch"], e["method"]) == key, path
        assert e["steps"] == 256, path
    uuids = {e["gpu_uuid"] for e in events}
    assert len(uuids) == 1
    completed = [e for e in events if e["event"] == "complete"]
    failures = [e for e in events if e["event"] == "failure"]
    measured = [e for e in events if e["event"] == "measurement"]
    warmups = [e for e in events if e["event"] == "warmup"]
    assert not (completed and failures), path
    status = "success" if completed else failures[-1]["status"] if failures else "incomplete"
    result = dict(
        size=size,
        length=length,
        batch=batch,
        method=method,
        status=status,
        provenance="measured",
        trials=len(measured),
        warmup_trials=len(warmups),
        gpu_uuid=next(iter(uuids)),
        source=str(path.relative_to(raw)),
        source_sha256=sha(path),
        first_event_at=events[0]["at"],
        last_event_at=events[-1]["at"],
        beyond_configured_positions=length > 40960,
    )
    if status != "success":
        failure = failures[-1] if failures else {}
        result.update(
            failure_phase=failure.get("phase", "unknown"),
            error_type=failure.get("error_type", "IncompleteLog"),
            error=failure.get("error", "No complete/failure event"),
        )
        return result, measured
    assert len(completed) == 1 and completed[0]["measured_trials"] == 15
    assert len(measured) == 15 and len(warmups) == 6
    assert {(e["document"], e["repeat"]) for e in measured} == {(d, r) for d in range(3) for r in range(5)}
    assert {(e["document"], e["repeat"]) for e in warmups} == {(d, r) for d in range(3) for r in (-2, -1)}
    validations = [e for e in events if e["event"] == "state_validated"]
    assert len(validations) == 3 and {e["document"] for e in validations} == {0, 1, 2}
    layers = model_config["num_hidden_layers"]
    heads = model_config["num_key_value_heads"]
    for v in validations:
        if method == "indexmem":
            assert v["disjoint_pages"] is True
            assert len(v["budgets"]) == layers
            assert all(
                len(b) == heads and sum(b) + heads * 64 == heads * 2048 and min(b) + 64 >= 512 for b in v["budgets"]
            )
        else:
            assert v["retained_tokens"] == length
    for e in measured + warmups:
        assert math.isfinite(e["seconds"]) and e["seconds"] > 0
        assert e["nonfinite_rows"] == 0
        assert close(e["batch_tokens_s"], batch * 256 / e["seconds"])
        assert close(e["sequence_tokens_s"], 256 / e["seconds"])
        assert close(e["ms_step"], 1000 * e["seconds"] / 256)
        if method == "indexmem":
            assert e["cmp_population_delta"] == 256 * layers * heads * batch
    for d in range(3):
        rows = [e for e in measured + warmups if e["document"] == d]
        assert len({(e["id"], e["output_sha256"]) for e in rows}) == 1, (path, d)
    seconds = sum(e["seconds"] for e in measured)
    tokens = sum(e["steps"] * e["batch"] for e in measured)
    latencies = [e["ms_step"] for e in measured]
    mean_latency = seconds * 1000 / (256 * len(measured))
    result.update(
        batch_tokens_s=tokens / seconds,
        sequence_tokens_s=tokens / seconds / batch,
        ms_step=mean_latency,
        ms_step_sd=statistics.stdev(latencies),
        latency_cv_pct=100 * statistics.stdev(latencies) / mean_latency,
        ms_step_min=min(latencies),
        ms_step_max=max(latencies),
        initial_allocated_gib=max(e["initial_allocated_bytes"] for e in measured) / 2**30,
        decode_peak_allocated_gib=max(e["peak_allocated_bytes"] for e in measured) / 2**30,
        decode_peak_reserved_gib=max(e["peak_reserved_bytes"] for e in measured) / 2**30,
        total_measured_seconds=seconds,
        total_generated_tokens=tokens,
    )
    return result, measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Run directory from prepare.py/run_matrix.py")
    parser.add_argument("--output", type=Path, help="Defaults to the run directory")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path(__file__).with_name("reference.json").read_text())
    pattern = re.compile(r"(4b|8b)_l(\d+)_b(\d+)_(full|indexmem)")
    names = set(json.loads((root / "jobs.json").read_text())) if (root / "jobs.json").exists() else set()
    names.update(path.name.split(".attempt")[0] for path in (root / "results").glob("*.attempt*.jsonl"))
    if not names:
        parser.error("No jobs or result JSONL files found")
    rows, attempts = [], []
    for name in sorted(names):
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"Unexpected job name: {name}")
        size, length, batch, method = match.groups()
        key = (size, int(length), int(batch), method)
        paths = sorted(
            (root / "results").glob(f"{name}.attempt*.jsonl"), key=lambda path: int(path.stem.rsplit("attempt", 1)[1])
        )
        candidates = []
        for path in paths:
            try:
                row, _ = parse_result(path, key, root, reference["models"][size]["config"])
            except (AssertionError, ValueError) as exc:
                row = dict(
                    size=size,
                    length=int(length),
                    batch=int(batch),
                    method=method,
                    status="invalid",
                    source=str(path.relative_to(root)),
                    source_sha256=sha(path),
                    error=f"{type(exc).__name__}: {exc}",
                )
            candidates.append(row)
            attempts.append(row.copy())
        successful = [row for row in candidates if row["status"] == "success"]
        if len(successful) > 1:
            raise ValueError(f"Multiple completed attempts for {name}; select a single run explicitly")
        row = (
            successful
            or candidates
            or [dict(size=size, length=int(length), batch=int(batch), method=method, status="pending")]
        )[-1].copy()
        row["attempts"] = len(paths)
        rows.append(row)
    index = {(row["size"], row["length"], row["batch"], row["method"]): row for row in rows}
    for row in rows:
        full = index.get((row["size"], row["length"], row["batch"], "full"))
        if full and full["status"] == row["status"] == "success":
            row["speedup_vs_full"] = row["batch_tokens_s"] / full["batch_tokens_s"]
            row["same_gpu_as_full"] = row["gpu_uuid"] == full["gpu_uuid"]
    leading = ["size", "length", "batch", "method", "status", "batch_tokens_s", "sequence_tokens_s", "speedup_vs_full"]
    fields = leading + sorted(set().union(*(row.keys() for row in rows)) - set(leading))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dump(output / "summary.json", rows)
    dump(output / "attempts.json", attempts)
    print(
        json.dumps(
            {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}
        )
    )


if __name__ == "__main__":
    main()
