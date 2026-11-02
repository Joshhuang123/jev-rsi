#!/usr/bin/env python3
"""tev1 决策协议的本地 ollama 评测器。

为什么自己写而不是用 tev1/scripts/evaluate.py：
  那个脚本 --provider 只支持 together，会打计费 API。

这里直接 import tev1 仓库自己的 examples/decide.py 的 payload()/select()，
保证协议逐字一致（system prompt、temperature=0、max_tokens=8、logprobs、
enable_thinking=False），只把传输层从 Together 换成本机 ollama。

唯一差别：Together 端用 response_format.regex 在服务端约束输出只能是某个字母，
ollama 没有 regex 约束（只有 JSON schema），所以这里改成「本地按允许字母校验」，
统计口径与 tev1 的 valid 字段保持一致（返回不在允许集合内的字母算 invalid）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

TEV1 = Path.home() / "WorkBuddy" / "tev1"
sys.path.insert(0, str(TEV1))
from examples.decide import payload, select  # noqa: E402

LOCAL_CHAT = "http://localhost:11434/api/chat"


def chat_local(body: dict, timeout: float = 120.0, retries: int = 4) -> dict:
    """把 tev1 的 OpenAI 风格 body 转成 ollama /api/chat。

    带重试：这台机器上 ollama 会被并行的其它任务重启（PID 变过），
    重启瞬间的连接错误不该算成模型答错。
    """
    msgs = body["messages"]
    opts = dict(body.get("options") or {})
    opts["num_predict"] = body.get("max_tokens", 8)
    opts["temperature"] = body.get("temperature", 0)
    if body.get("logprobs"):
        opts["logprobs"] = 5
    payload_o = {
        "model": body["model"],
        "messages": msgs,
        "stream": False,
        "think": False,
        "options": opts,
    }
    # tev1 用 chat_template_kwargs 关思考；ollama 用顶层 think 字段
    ck = body.get("chat_template_kwargs") or {}
    if "enable_thinking" in ck:
        payload_o["think"] = bool(ck["enable_thinking"])

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                LOCAL_CHAT, data=json.dumps(payload_o).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            text = d.get("message", {}).get("content", "")
            return {
                "model": d.get("model"),
                "choices": [{
                    "message": {"content": text, "logprobs": d.get("logprobs")},
                }],
            }
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def summarize(rows):
    return {"requests": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "accuracy": (sum(r["correct"] for r in rows) / len(rows)) if rows else None,
            "valid": sum(r["valid"] for r in rows),
            "http_errors": sum(r["status"] != 200 for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if not 1 <= args.concurrency <= 8:
        ap.error("并发 1–8（ollama 本机，别压太狠）")

    raw = args.records.read_bytes()
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        ap.error("记录非空且 id 唯一")
    for row in rows:
        payload(row, args.model)          # 协议校验（选项数、label 连续、key 唯一）
        sel = next((o for o in row["options"] if o["label"] == row["answer"]), None)
        if sel is None or sel["key"] != row["answer_key"]:
            ap.error(f"{row['id']}: 金标 label 与 key 不一致")

    args.output.mkdir(parents=True, exist_ok=True)

    def one(row):
        out = {k: row.get(k) for k in ["id", "source", "group_id"]}
        out.update(expected=row["answer"], expected_key=row["answer_key"], status=0,
                   valid=False, correct=False, predicted_key=None)
        body = payload(row, args.model)
        t0 = time.perf_counter()
        try:
            data = chat_local(body)
            out["status"] = 200
            out["logprobs"] = data["choices"][0].get("logprobs")
            chosen = select(data, row["options"])
            out.update(valid=True, correct=chosen["label"] == row["answer"],
                       predicted_key=chosen["key"], resolved_model=data.get("model"))
        except urllib.error.HTTPError as e:
            out.update(status=e.code, error="HTTPError")
        except Exception as e:
            out["error"] = type(e).__name__
            out["error_msg"] = str(e)[:200]
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        with (args.output / "results.jsonl").open("w") as fh:
            for r in pool.map(one, rows):
                results.append(r)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                if len(results) % 50 == 0:
                    print(f"  … {len(results)}/{len(rows)}  当前准确率 "
                          f"{sum(x['correct'] for x in results)/len(results):.3f}", flush=True)

    report = {
        "model": args.model, "provider": "local-ollama",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "limit": args.limit, "concurrency": args.concurrency,
        "decoding": {"type": "client-side label validation", "logprobs": True,
                     "top_logprobs": 5, "max_tokens": 8, "temperature": 0,
                     "enable_thinking": False},
        **summarize(results),
        "by_source": {s: summarize([r for r in results if r["source"] == s])
                      for s in sorted({r["source"] for r in results})},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
