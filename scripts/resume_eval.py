#!/usr/bin/env python3
"""续跑被中断的评测：从已有 results.jsonl 读出算过的 id，只跑剩下的。

这台机器上 ollama 会被并行的其它任务重启，评测会中途夭折；
每次夭折后用这个脚本接着跑，不用从头来。
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path.home() / "WorkBuddy/jev-rsi"))
from eval_local import chat_local, summarize  # noqa: E402

TEV1 = pathlib.Path(os.environ.get("TEV1_REPO", pathlib.Path.home() / "WorkBuddy/tev1"))
sys.path.insert(0, str(TEV1))
from examples.decide import payload, select  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {}
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    rows = [json.loads(l) for l in args.records.read_text().splitlines() if l.strip()]
    todo = [r for r in rows if r["id"] not in done]
    print(f"  已有 {len(done)} 条，还差 {len(todo)} 条", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    def one(row):
        o = {k: row.get(k) for k in ["id", "source", "group_id"]}
        o.update(expected=row["answer"], expected_key=row["answer_key"], status=0,
                 valid=False, correct=False, predicted_key=None)
        body = payload(row, args.model)
        t0 = time.perf_counter()
        try:
            data = chat_local(body)
            o["status"] = 200
            o["logprobs"] = data["choices"][0].get("logprobs")
            chosen = select(data, row["options"])
            o.update(valid=True, correct=chosen["label"] == row["answer"],
                     predicted_key=chosen["key"], resolved_model=data.get("model"))
        except Exception as e:
            o["error"] = type(e).__name__
            o["error_msg"] = str(e)[:200]
        o["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return o

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        with res_path.open("a") as fh:
            for i, r in enumerate(pool.map(one, todo), 1):
                done[r["id"]] = r
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)

    results = [done[r["id"]] for r in rows if r["id"] in done]
    raw = args.records.read_bytes()
    import hashlib
    from datetime import datetime, timezone
    report = {
        "model": args.model, "provider": "local-ollama",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "concurrency": args.concurrency,
        "decoding": {"type": "client-side label validation", "logprobs": True,
                     "top_logprobs": 5, "max_tokens": 8, "temperature": 0,
                     "enable_thinking": False},
        **summarize(results),
        "by_source": {s: summarize([r for r in results if r["source"] == s])
                      for s in sorted({r["source"] for r in results})},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ["accuracy", "correct", "requests", "valid", "http_errors"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
