#!/usr/bin/env python3
"""head-only 决策头：冻结 Qwen3.5-4B，在 last-token latent 上训一个 MLP。

照搬 openjev 的 `code/latent_mlp.py` 配方：
  · 每个 (question, option) 对一个特征（我们用的是**基础模型**的 last-token hidden state，
    不是 openjev 那个 NLI 微调过的模型 —— 那是它拿 0.814 分的原因，我们这里测的是
    「不微调主干，只加个头」能做到多少」）
  · soft-BCE 损失：gold 标签给 1-eps，其余给 eps
  · 逐问题 argmax 准确率（不是逐对二分类准确率）

对比基线：
  1) 随机猜
  2) LLM 直接选字母（轨道 A 的基线）
  3) head-only（这里）
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def load_npz(p: pathlib.Path):
    d = np.load(p, allow_pickle=True)
    return d["latents"], d["offsets"], d["groups"], d["gold"]


def make_xy(lat, offs, groups, gold):
    """展开成 (样本, 选项, 特征) + 逐问题标签。

    各题选项数不同（4/6/8），所以要 pad 到最大选项数，不能直接 np.stack。
    """
    n = len(offs) - 1
    ks = np.array([offs[i + 1] - offs[i] for i in range(n)])
    kmax = int(ks.max())
    dim = lat.shape[1]
    X = np.zeros((n, kmax, dim), dtype=np.float32)
    y = np.zeros((n, kmax), dtype=np.float32)
    for i in range(n):
        k = ks[i]
        X[i, :k] = lat[offs[i]:offs[i + 1]]
        y[i, gold[i]] = 1.0
    return X, y, ks, groups


def normalize(X, mu=None, sd=None):
    flat = X.reshape(-1, X.shape[-1])
    if mu is None:
        mu = flat.mean(0)
        sd = flat.std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32), mu, sd


def train_head(Xtr, ytr, ktr, Xte, yte, kte, *, hidden=0, epochs=40, lr=3e-3,
               eps=0.02, seed=0, l2=0.0):
    """纯 numpy 的 MLP + soft-BCE。

    形态（照 openjev 的 latent-MLP）：**每个选项打一个标量分**，同一套权重在所有选项间共享；
    逐问题做 listwise 的 soft-BCE —— gold 给 1-eps，其余给 eps；预测取该题内 argmax。
    """
    rng = np.random.default_rng(seed)
    n, kmax, dim = Xtr.shape
    mask = np.zeros((n, kmax), dtype=np.float32)
    for i in range(n):
        mask[i, : ktr[i]] = 1.0
    target = ytr * (1 - eps) + (1 - ytr) * eps

    d1 = dim if hidden == 0 else hidden
    if hidden:
        W1 = rng.normal(0, 1 / np.sqrt(dim), (dim, hidden)).astype(np.float32)
        b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.normal(0, 1 / np.sqrt(d1), (d1, 1)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)

    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
    if hidden:
        mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
        mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)

    def forward(Xb):
        if hidden:
            H = np.tanh(Xb @ W1 + b1)
            return (H @ W2 + b2)[:, :, 0]     # (n, k)
        return (Xb @ W2 + b2)[:, :, 0]

    def adam(param, grad, m, v, lr_, t):
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        mh = m / (1 - 0.9 ** t)
        vh = v / (1 - 0.999 ** t)
        return param - lr_ * mh / (np.sqrt(vh) + 1e-8), m, v

    for ep in range(epochs):
        t = ep + 1
        logits = forward(Xtr)
        p = 1.0 / (1.0 + np.exp(-logits))
        p = p * mask + (1 - mask) * eps
        g = (p - target) * mask / max(n, 1)        # (n, k)
        if hidden:
            H = np.tanh(Xtr @ W1 + b1)
            gW2 = np.einsum("nkd,nk->d", H, g)[:, None] + l2 * W2
            gb2 = np.array([g.sum()], dtype=np.float32)
            gH = g[:, :, None] * W2.T              # (n, k, hidden)
            gW1 = np.einsum("nkd,nkh->dh", Xtr, gH * (1 - H ** 2)) + l2 * W1
            gb1 = (gH * (1 - H ** 2)).sum((0, 1))
        else:
            gW2 = np.einsum("nkd,nk->d", Xtr, g)[:, None] + l2 * W2
            gb2 = np.array([g.sum()], dtype=np.float32)
        W2, mW2, vW2 = adam(W2, gW2, mW2, vW2, lr, t)
        b2, mb2, vb2 = adam(b2, gb2, mb2, vb2, lr, t)
        if hidden:
            W1, mW1, vW1 = adam(W1, gW1, mW1, vW1, lr, t)
            b1, mb1, vb1 = adam(b1, gb1, mb1, vb1, lr, t)

    logits = forward(Xte)
    pred = logits.argmax(1)
    gold = np.array([int(np.argmax(yte[i, : kte[i]])) for i in range(len(kte))])
    acc = float((pred == gold).mean())
    return acc, (W1 if hidden else None, b1 if hidden else None, W2, b2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=pathlib.Path, required=True)
    ap.add_argument("--test", type=pathlib.Path, required=True)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    ltr, otr, gtr, ytr = load_npz(args.train)
    lte, ote, gte, yte = load_npz(args.test)
    Xtr, Ytr, ktr, _ = make_xy(ltr, otr, gtr, ytr)
    Xte, Yte, kte, _ = make_xy(lte, ote, gte, yte)
    # 只保留选项数一致的维度上限
    kmax = max(ktr.max(), kte.max())
    if Xtr.shape[1] < kmax:
        pad = np.zeros((Xtr.shape[0], kmax - Xtr.shape[1], Xtr.shape[2]), np.float32)
        Xtr = np.concatenate([Xtr, pad], 1); Ytr = np.concatenate([Ytr, np.zeros((len(Ytr), kmax - Ytr.shape[1]), np.float32)], 1)
    if Xte.shape[1] < kmax:
        pad = np.zeros((Xte.shape[0], kmax - Xte.shape[1], Xte.shape[2]), np.float32)
        Xte = np.concatenate([Xte, pad], 1); Yte = np.concatenate([Yte, np.zeros((len(Yte), kmax - Yte.shape[1]), np.float32)], 1)

    Xtr, mu, sd = normalize(Xtr)
    Xte, _, _ = normalize(Xte, mu, sd)

    rnd = float(np.mean([1.0 / k for k in kte]))   # 随机基线（按选项数加权）
    acc, _ = train_head(Xtr, Ytr, ktr, Xte, Yte, kte,
                        hidden=args.hidden, epochs=args.epochs, lr=args.lr,
                        eps=args.eps, seed=args.seed, l2=args.l2)
    print(json.dumps({
        "train_questions": int(len(Xtr)), "test_questions": int(len(Xte)),
        "dim": int(Xtr.shape[2]), "hidden": args.hidden,
        "epochs": args.epochs, "lr": args.lr, "eps": args.eps, "l2": args.l2,
        "seed": args.seed,
        "random_baseline": rnd, "head_only_accuracy": acc,
    }, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps({"acc": acc, "random": rnd}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
