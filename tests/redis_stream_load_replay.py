"""
Replay-based load test writer for Redis Streams.

Purpose:
- Snapshot recent entries from an existing stream (for example face_inference:requests)
- Re-publish duplicated rows back into the same stream at a controlled rate
- Help find service throughput limits under rising input load

Example:
    python tests/redis_stream_load_replay.py \
        --redis-url redis://localhost:6379 \
        --stream face_inference:requests \
        --sample-size 100 \
        --rate 120 \
        --duration 120

Notes:
- This script writes NEW entries into the target stream (it does not mutate existing ones).
- event_id and timestamp are refreshed per emitted row to avoid ID collisions and keep data live-like.
- If your payload has deadline_ts_ms, this script can refresh it with --deadline-ms.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass

import redis


@dataclass
class Counters:
    sent: int = 0
    start_ts: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay recent Redis stream rows at controlled rate")
    p.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    p.add_argument("--stream", default="face_inference:requests", help="Target stream to write into")
    p.add_argument("--sample-size", type=int, default=100, help="How many recent rows to snapshot")
    p.add_argument("--rate", type=float, default=100.0, help="Target writes per second")
    p.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    p.add_argument(
        "--burst-size",
        type=int,
        default=1,
        help="Rows written per scheduling tick using one Redis pipeline exec [1]",
    )
    p.add_argument(
        "--deadline-ms",
        type=int,
        default=0,
        help="If >0 and deadline_ts_ms field exists, set it to now+deadline-ms",
    )
    p.add_argument(
        "--refresh-event-id",
        action="store_true",
        default=True,
        help="Refresh event_id for each replayed row (default: on)",
    )
    p.add_argument(
        "--no-refresh-event-id",
        dest="refresh_event_id",
        action="store_false",
        help="Keep original event_id (not recommended)",
    )
    p.add_argument(
        "--print-every",
        type=float,
        default=1.0,
        help="Stats print interval (seconds)",
    )
    return p.parse_args()


def b(s: str) -> bytes:
    return s.encode("utf-8")


def snapshot_rows(r: redis.Redis, stream: str, sample_size: int) -> list[dict[bytes, bytes]]:
    # Newest first from XREVRANGE, then reverse for natural order.
    rows = r.xrevrange(stream, max=b"+", min=b"-", count=sample_size)
    if not rows:
        return []
    rows.reverse()
    return [fields for _msg_id, fields in rows]


def mutate_fields(
    fields: dict[bytes, bytes],
    seq: int,
    refresh_event_id: bool,
    deadline_ms: int,
) -> dict[bytes, bytes]:
    out = dict(fields)
    now_ms = int(time.time() * 1000)

    # Keep timestamp fresh for latency-sensitive consumers.
    out[b"timestamp"] = str(now_ms).encode("utf-8")

    if refresh_event_id and b"event_id" in out:
        base = out[b"event_id"].decode("utf-8", errors="replace")
        out[b"event_id"] = f"{base}_lt_{now_ms}_{seq}".encode("utf-8")

    if deadline_ms > 0 and b"deadline_ts_ms" in out:
        out[b"deadline_ts_ms"] = str(now_ms + deadline_ms).encode("utf-8")

    return out


def run_load_test(args: argparse.Namespace) -> int:
    try:
        r = redis.from_url(args.redis_url, decode_responses=False)
        r.ping()
    except Exception as exc:
        print(f"[ERROR] Redis connect failed: {exc}", file=sys.stderr)
        return 2

    source_rows = snapshot_rows(r, args.stream, args.sample_size)
    if not source_rows:
        print(f"[ERROR] No rows found in stream '{args.stream}'.")
        return 3

    print("=" * 72)
    print("Redis Stream Replay Load Test")
    print("=" * 72)
    print(f"redis_url      : {args.redis_url}")
    print(f"stream         : {args.stream}")
    print(f"sample_size    : {len(source_rows)}")
    print(f"target_rate    : {args.rate:.2f} rows/s")
    print(f"burst_size     : {max(args.burst_size, 1)}")
    print(f"duration       : {args.duration:.1f}s")
    print(f"deadline_ms    : {args.deadline_ms}")
    print(f"refresh_event_id: {args.refresh_event_id}")
    print("=" * 72)

    counters = Counters(sent=0, start_ts=time.perf_counter())
    end_ts = counters.start_ts + args.duration
    next_print = counters.start_ts + max(args.print_every, 0.1)

    burst_size = max(args.burst_size, 1)
    # Pace writes by bursts while preserving target average rows/s.
    interval = burst_size / max(args.rate, 1e-6)
    next_emit = counters.start_ts

    rows_iter = itertools.cycle(source_rows)

    while True:
        now = time.perf_counter()
        if now >= end_ts:
            break

        if now < next_emit:
            time.sleep(min(next_emit - now, 0.002))
            continue

        try:
            pipe = r.pipeline(transaction=False)
            for _ in range(burst_size):
                src = next(rows_iter)
                payload = mutate_fields(
                    src,
                    seq=counters.sent,
                    refresh_event_id=args.refresh_event_id,
                    deadline_ms=args.deadline_ms,
                )
                pipe.xadd(args.stream, payload)
                counters.sent += 1
            pipe.execute()
        except Exception as exc:
            print(f"[WARN] XADD failed at seq={counters.sent}: {exc}")

        # Catch up if we fell behind to keep average rate near target.
        next_emit += interval
        if now - next_emit > 1.0:
            next_emit = now + interval

        if now >= next_print:
            elapsed = max(now - counters.start_ts, 1e-9)
            achieved = counters.sent / elapsed
            print(f"sent={counters.sent:7d} elapsed={elapsed:7.2f}s achieved={achieved:8.2f} rows/s")
            next_print = now + max(args.print_every, 0.1)

    total_elapsed = max(time.perf_counter() - counters.start_ts, 1e-9)
    achieved = counters.sent / total_elapsed
    print("-" * 72)
    print(f"DONE sent={counters.sent} in {total_elapsed:.2f}s | achieved={achieved:.2f} rows/s")
    print("-" * 72)
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run_load_test(args))


if __name__ == "__main__":
    main()
