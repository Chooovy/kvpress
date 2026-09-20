# SPDX-FileCopyrightText: 2026 Xintong Yang
# SPDX-License-Identifier: Apache-2.0

"""Run FA2 configuration pairs on explicitly selected, idle GPU UUIDs.

Each worker processes both methods of a configuration in order on its GPU.
Completed outcomes are skipped; failed/interrupted attempts need --only to retry.
This launcher never starts or stops GPU reservation scripts.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import itertools
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

from prepare import CODE, verify_prepared, write_json
from summarize import BATCHES, LENGTHS, METHODS, SIZES, parse_result


def job_name(key):
    size, length, batch, method = key
    return f"{size}_l{length}_b{batch}_{method}"


def compute_processes():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True, timeout=15
    )
    return [(uuid.strip(), int(pid)) for uuid, pid in (line.split(",") for line in output.splitlines() if line.strip())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", help="Full GPU UUIDs from nvidia-smi -L")
    parser.add_argument("--sizes", choices=SIZES, nargs="+", default=list(SIZES))
    parser.add_argument("--lengths", type=int, choices=LENGTHS, nargs="+", default=list(LENGTHS))
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--only", nargs="+", help="Exact method job names for selected runs or explicit retries")
    parser.add_argument("--dry-run", action="store_true", help="Print the queue without checking assets or using GPUs")
    args = parser.parse_args()
    if any(batch < 1 for batch in args.batches):
        parser.error("batches must be positive")
    pairs = [
        [(size, length, batch, method) for method in METHODS if method in args.methods]
        for length, size, batch in itertools.product(
            sorted(set(args.lengths)), sorted(set(args.sizes)), sorted(set(args.batches))
        )
    ]
    names = {job_name(key) for pair in pairs for key in pair}
    if args.only and not set(args.only) <= names:
        parser.error(f"Unknown --only names: {sorted(set(args.only) - names)}")
    selected = set(args.only or names)
    pairs = [[key for key in pair if job_name(key) in selected] for pair in pairs]
    pairs = [pair for pair in pairs if pair]
    if args.dry_run:
        print(json.dumps([[job_name(key) for key in pair] for pair in pairs], indent=2))
        return 0
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(not uuid.startswith("GPU-") for uuid in args.gpus):
        parser.error("--gpus requires distinct full GPU UUIDs")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"):
        parser.error("Unset allocator overrides to match the published FA2 environment")

    root = args.root.resolve()
    with (root / "matrix.lock").open("a") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_prepared(root)
        available = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=15
        ).splitlines()
        if not set(args.gpus) <= set(available):
            raise ValueError("Unknown GPU UUID")
        occupied = [row for row in compute_processes() if row[0] in args.gpus]
        if occupied:
            raise RuntimeError(f"Selected GPUs are occupied: {occupied}")
        results, logs = root / "results", root / "logs"
        results.mkdir(exist_ok=True)
        logs.mkdir(exist_ok=True)
        manifest = root / "jobs.json"
        existing = json.loads(manifest.read_text()) if manifest.exists() else []
        write_json(manifest, sorted(set(existing) | selected))
        reference = json.loads((CODE / "reference.json").read_text())
        tasks = queue.Queue()
        for pair in pairs:
            tasks.put(pair)
        stop = threading.Event()
        children, failures = {}, []
        guard = threading.Lock()

        def terminate(signum=None, frame=None):
            stop.set()
            with guard:
                for proc in children.values():
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass

        signal.signal(signal.SIGINT, terminate)
        signal.signal(signal.SIGTERM, terminate)

        def run(uuid, key):
            name = job_name(key)
            paths = sorted(results.glob(f"{name}.attempt*.jsonl"), key=lambda p: int(p.stem.rsplit("attempt", 1)[1]))
            config = reference["models"][key[0]]["config"]
            for path in paths:
                try:
                    row, _ = parse_result(path, key, root, config)
                except (AssertionError, ValueError):
                    continue
                if row["status"] == "success":
                    print(f"Skip completed: {name}", flush=True)
                    return
            prior_logs = list(logs.glob(f"{name}.attempt*.log"))
            if (paths or prior_logs) and not args.only:
                print(f"Preserve prior attempt: {name}; use --only to retry", flush=True)
                return
            if any(gpu == uuid for gpu, _ in compute_processes()):
                raise RuntimeError(f"GPU occupied before launch: {uuid}")
            attempt = 1
            while (results / f"{name}.attempt{attempt}.jsonl").exists() or (
                logs / f"{name}.attempt{attempt}.log"
            ).exists():
                attempt += 1
            output = results / f"{name}.attempt{attempt}.jsonl"
            command = [
                sys.executable,
                str(CODE / "benchmark.py"),
                "--root",
                str(root),
                "--size",
                key[0],
                "--length",
                str(key[1]),
                "--batch",
                str(key[2]),
                "--method",
                key[3],
                "--output",
                str(output),
            ]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=uuid,
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                NUMEXPR_NUM_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
                PYTHONUNBUFFERED="1",
            )
            with (logs / f"{name}.attempt{attempt}.log").open("x") as log:
                with guard:
                    if stop.is_set():
                        return
                    proc = subprocess.Popen(
                        command,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    children[uuid] = proc
                print(f"Start {name} on {uuid} (PID {proc.pid})", flush=True)
                try:
                    while proc.poll() is None and not stop.wait(5):
                        foreign = [pid for gpu, pid in compute_processes() if gpu == uuid and pid != proc.pid]
                        if foreign:
                            raise RuntimeError(f"Foreign GPU process on {uuid}: {foreign}")
                finally:
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                    with guard:
                        children.pop(uuid, None)
                if stop.is_set():
                    return
                try:
                    row, _ = parse_result(output, key, root, config)
                    status = row["status"]
                    if (proc.returncode == 0) != (status == "success"):
                        status = "error"
                except (OSError, AssertionError, ValueError):
                    status = "error"
                print(f"Finish {name}: {status}", flush=True)
                if status not in ("success", "oom"):
                    failures.append(name)

        def worker(uuid):
            try:
                while not stop.is_set():
                    try:
                        pair = tasks.get_nowait()
                    except queue.Empty:
                        return
                    for key in pair:
                        if stop.is_set():
                            return
                        run(uuid, key)
            except BaseException:
                terminate()
                raise

        try:
            with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
                futures = [executor.submit(worker, uuid) for uuid in args.gpus]
                for future in futures:
                    future.result()
        finally:
            interrupted = stop.is_set()
            terminate()
        return 1 if failures or interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
