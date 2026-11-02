#!/usr/bin/env python3
"""把 ollama 里 MLX 模型的「每张量一个 blob」重组回单个 safetensors 文件。

为什么需要：ollama 0.40 导入 safetensors 时，把每个张量（或一组）拆成独立的 blob 存。
但每个 blob 本身仍是**合法的 safetensors 文件**（自带 header + 数据）。
所以不用重新下载 3GB 权重，直接把 730 个 blob 拼回去即可。

原理：
  safetensors = [8 字节 header 长度][header JSON][data buffer]
  header 里每个张量记录 dtype/shape/在 data buffer 里的 offset/长度

步骤：
  1. 读 manifest，按 mediaType 分出 tensor 层与 json 层
  2. 逐个 tensor blob 读 header，收集 (name, dtype, shape, 该 blob 内 offset, 数据长度)
  3. 写新文件：先算每个张量在总 buffer 里的新 offset，再按序落数据
  4. json 层（config/tokenizer…）原样落盘
"""
import json
import struct
import sys
from pathlib import Path

OLLAMA = Path.home() / ".ollama/models"
BLOBS = OLLAMA / "blobs"


def read_safetensors_header(fh):
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
    meta = hdr.pop("__metadata__", None)
    return hdr, meta, n


def main():
    manifest_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    m = json.loads(manifest_path.read_text())
    layers = m.get("layers", [])
    tensor_layers, json_layers = [], []
    for l in layers:
        d = l["digest"].replace(":", "-")
        mt = l.get("mediaType", "")
        if "tensor" in mt:
            tensor_layers.append((l, d))
        elif mt.endswith("json"):
            json_layers.append((l, d))

    print(f"  tensor 层 {len(tensor_layers)}  json 层 {len(json_layers)}")

    # ── 1. 收集所有张量元信息 ────────────────────────────────────────
    entries = []          # (name, dtype, shape, blob_path, offset_in_blob, nbytes)
    total_data = 0
    for l, d in tensor_layers:
        fp = BLOBS / d
        if not fp.exists():
            print(f"  ⚠ 缺 blob {d[:20]}…  跳过")
            continue
        with open(fp, "rb") as fh:
            hdr, meta, _ = read_safetensors_header(fh)
            for name, info in hdr.items():
                off = int(info["data_offsets"][0])
                n = int(info["data_offsets"][1]) - off
                entries.append({
                    "name": name, "dtype": info["dtype"], "shape": info["shape"],
                    "blob": fp, "off": off, "n": n,
                })
                total_data += n
    print(f"  张量总数 {len(entries)}  数据总量 {total_data/1e9:.2f} GB")

    # ── 2. 写新的 safetensors ────────────────────────────────────────
    header = {}
    cursor = 0
    for e in entries:
        header[e["name"]] = {
            "dtype": e["dtype"], "shape": e["shape"],
            "data_offsets": [cursor, cursor + e["n"]],
        }
        e["new_off"] = cursor
        cursor += e["n"]

    hdr_bytes = json.dumps(header, separators=(",", ":")).encode()
    # safetensors 要求 header 长度 8 字节对齐（实际只要求能被 u64 读出）
    pad = (8 - len(hdr_bytes) % 8) % 8
    hdr_bytes += b" " * pad

    out = out_dir / "model.safetensors"
    with open(out, "wb") as fo:
        fo.write(struct.pack("<Q", len(hdr_bytes)))
        fo.write(hdr_bytes)
        # 按 blob 分组读，逐块写出（同一 blob 的张量在文件里可能不连续 → 按 new_off 排序）
        by_blob = {}
        for e in entries:
            by_blob.setdefault(e["blob"], []).append(e)
        for fp, group in by_blob.items():
            group.sort(key=lambda x: x["new_off"])
            with open(fp, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                fh.read(n)
                base = fh.tell()
                for e in group:
                    fh.seek(base + e["off"])
                    remain = e["n"]
                    while remain > 0:
                        chunk = fh.read(min(remain, 1 << 24))
                        if not chunk:
                            raise RuntimeError(f"读短了: {e['name']}")
                        fo.write(chunk)
                        remain -= len(chunk)
    print(f"  ✓ 写出 {out}  ({out.stat().st_size/1e9:.2f} GB)")

    # ── 3. json 层原样落盘 ───────────────────────────────────────────
    for l, d in json_layers:
        fp = BLOBS / d
        if not fp.exists():
            continue
        name = l.get("mediaType", "x").split(".")[-1]
        dst = out_dir / {
            "application/vnd.docker.container.image.v1+json": "config.json",
            "application/vnd.ollama.image.model.template": "chat_template.jinja",
        }.get(l.get("mediaType", ""), f"layer_{d[:12]}.json")
        # tokenizer 之类按内容猜个名
        head = fp.read_bytes()[:200]
        if b'"tokenizer' in head and dst.name.startswith("layer_"):
            dst = out_dir / "tokenizer.json"
        dst.write_bytes(fp.read_bytes())
        print(f"  ✓ {dst.name}  ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
