#!/usr/bin/env python3
"""RSI 循环：围绕冻结的 Qwen3.5-4B，反复提出实验 → 跑 → 按固定规则留/汰。

这是 openjev 那种自我改进循环的最小可复现版本。按 OrcaRouter 那篇 RSJ 解读里的定义：
  「一个系统提出自己的下一个实验，运行它，并依据**事先固定**的规则保留或淘汰结果，
    而结果又为下一轮提供输入。……它的质量完全取决于这个评分函数被应用得有多诚实。」

所以这里三件事必须事先钉死，本脚本写死在代码里、不可在运行中改：
  ① 评分函数 = **dev 集**的逐问题 argmax 准确率（test 集全程不参与选择，只在最后报一次）
  ② 留汰规则 = 候选分数 > 当前最优分 → 采纳（成为下一轮的起点配置）；否则丢弃
  ③ 实验空间 = 下面 PROPOSALS 里写死的候选序列（头维度 / 学习率 / 轮数 / eps / L2）

每一轮的产物都写进 run_record.jsonl：这是「能读这个循环」而不是「只能相信它」的关键。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from train_head import load_npz, make_xy, normalize, train_head

# ── 事先固定的实验提案序列 ────────────────────────────────────────────
PROPOSALS = [
    # (说明,                hidden, lr,     epochs, eps,   l2,     seed)
    ("线性头 / lr3e-3",        0,   3e-3,   40,     0.02,  0.0,   0),
    ("线性头 / lr1e-2",        0,   1e-2,   40,     0.02,  0.0,   0),
    ("MLP128 / lr3e-3",      128,  3e-3,   40,     0.02,  0.0,   0),
    ("MLP128 / lr1e-2",      128,  1e-2,   40,     0.02,  0.0,   0),
    ("MLP256 / lr3e-3",      256,  3e-3,   40,     0.02,  0.0,   0),
    ("MLP256 / lr1e-2",      256,  1e-2,   40,     0.02,  0.0,   0),
    ("MLP256 / eps0.05",     256,  1e-2,   40,     0.05,  0.0,   0),
    ("MLP256 / eps0.10",     256,  1e-2,   40,     0.10,  0.0,   0),
    ("MLP256 / 80 轮",       256,  1e-2,   80,     0.02,  0.0,   0),
    ("MLP256 / L2=1e-4",     256,  1e-2,   40,     0.02,  1e-4,  0),
    ("MLP512 / lr1e-2",      512,  1e-2,   40,     0.02,  0.0,   0),
    ("MLP512 / 80 轮",       512,  1e-2,   80,     0.02,  0.0,   0),
]


def pad_to(X, Y, kmax):
    if X.shape[1] < kmax:
        X = np.concatenate([X, np.zeros((len(X), kmax - X.shape[1], X.shape[2]), np.float32)], 1)
        Y = np.concatenate([Y, np.zeros((len(Y), kmax - Y.shape[1]), np.float32)], 1)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=pathlib.Path, required=True)
    ap.add_argument("--test", type=pathlib.Path, required=True)
    ap.add_argument("--dev-frac", type=float, default=0.22, help="从 train 里切多少比例当 dev")
    ap.add_argument("--record", type=pathlib.Path, default=pathlib.Path("/tmp/rsi_run_record.jsonl"))
    args = ap.parse_args()

    ltr, otr, gtr, ytr = load_npz(args.train)
    lte, ote, gte, yte = load_npz(args.test)
    Xtr, Ytr, ktr, _ = make_xy(ltr, otr, gtr, ytr)
    Xte, Yte, kte, _ = make_xy(lte, ote, gte, yte)
    kmax = max(ktr.max(), kte.max())
    Xtr, Ytr = pad_to(Xtr, Ytr, kmax)
    Xte, Yte = pad_to(Xte, Yte, kmax)

    # 从 train 里切 dev（切分固定 seed，事先钉死）
    rng = np.random.default_rng(1234)
    idx = rng.permutation(len(Xtr))
    n_dev = int(len(Xtr) * args.dev_frac)
    di, ti = idx[:n_dev], idx[n_dev:]
    Xdev, Ydev, kdev = Xtr[di], Ytr[di], ktr[di]
    Xfit, Yfit, kfit = Xtr[ti], Ytr[ti], ktr[ti]

    # 归一化参数只用 fit 部分算（避免 dev 泄露进特征统计）
    Xfit, mu, sd = normalize(Xfit)
    Xdev, _, _ = normalize(Xdev, mu, sd)
    Xte, _, _ = normalize(Xte, mu, sd)

    rnd_dev = float(np.mean([1.0 / k for k in kdev]))
    print(f"  fit {len(Xfit)}  dev {len(Xdev)}  test {len(Xte)}  随机基线 {rnd_dev:.3f}")

    best = {"score": -1.0, "cfg": None, "desc": None}
    history = []
    with args.record.open("w") as rec:
        for i, (desc, hidden, lr, epochs, eps, l2, seed) in enumerate(PROPOSALS, 1):
            t0 = time.time()
            acc, _ = train_head(Xfit, Yfit, kfit, Xdev, Ydev, kdev,
                                hidden=hidden, epochs=epochs, lr=lr,
                                eps=eps, seed=seed, l2=l2)
            kept = acc > best["score"]          # ← 事先固定的留汰规则
            if kept:
                best = {"score": acc, "cfg": (hidden, lr, epochs, eps, l2, seed), "desc": desc}
            row = {"round": i, "desc": desc, "hidden": hidden, "lr": lr, "epochs": epochs,
                   "eps": eps, "l2": l2, "seed": seed, "dev_acc": acc,
                   "kept": kept, "best_dev_acc": best["score"], "best_desc": best["desc"],
                   "secs": round(time.time() - t0, 1)}
            history.append(row)
            rec.write(json.dumps(row, ensure_ascii=False) + "\n")
            rec.flush()
            print(f"  [{i:>2}/{len(PROPOSALS)}] {desc:<24} dev={acc:.4f} "
                  f"{'✅采纳' if kept else '  丢弃'}  (当前最优 {best['score']:.4f} {best['desc']})", flush=True)

    # 胜出的配置在 test 上跑一次（test 全程没参与选择）
    hidden, lr, epochs, eps, l2, seed = best["cfg"]
    Xf2, Yf2, kf2 = Xfit, Yfit, kfit
    acc_te, _ = train_head(Xf2, Yf2, kf2, Xte, Yte, kte,
                           hidden=hidden, epochs=epochs, lr=lr, eps=eps, seed=seed, l2=l2)
    rnd_te = float(np.mean([1.0 / k for k in kte]))
    final = {
        "scoring_function": "dev 集逐问题 argmax 准确率（事先固定，test 全程不参与选择）",
        "keep_rule": "候选 dev 分数 > 当前最优 → 采纳为新基线，否则丢弃",
        "rounds": len(PROPOSALS),
        "best_round": best["desc"],
        "best_dev_acc": best["score"],
        "test_accuracy": acc_te,
        "random_baseline_test": rnd_te,
        "train_questions": int(len(Xfit)), "dev_questions": int(len(Xdev)),
        "test_questions": int(len(Xte)),
    }
    print("\n  " + json.dumps(final, ensure_ascii=False, indent=2))
    pathlib.Path("/tmp/rsi_final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
