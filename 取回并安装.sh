#!/usr/bin/env bash
# jev-rsi · 在「这台 Mac mini」上执行一次，即可完成同步
#
#   前提：T7 移动硬盘插在本机上
#   用法：bash /Volumes/T7/智海论衡/工作档案/jev-rsi/取回并安装.sh
#
# 做的事：
#   1. 把仓库从 T7 拷到 ~/WorkBuddy/jev-rsi
#   2. 建 venv、装依赖
#   3. 拉 tev1 官方评测集
#   4. 打印跑实验的命令
#
# 想改安装位置：JEVRSI_HOME=/your/path bash 取回并安装.sh

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${JEVRSI_HOME:-$HOME/WorkBuddy/jev-rsi}"
TEV1_DIR="${TEV1_DIR:-$HOME/WorkBuddy/tev1}"
VENV_DIR="$DEST/.venv"

say()  { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "$SRC" ] || die "源码目录异常：$SRC"
command -v python3 >/dev/null 2>&1 || die "找不到 python3"
command -v git >/dev/null 2>&1 || die "找不到 git"

# ---- 1. 拷仓库（备份旧版，不删） ----
say "拷贝仓库 → $DEST"
mkdir -p "$(dirname "$DEST")"
if [ -e "$DEST" ]; then
  BAK="${DEST}.old.$(date +%Y%m%d-%H%M%S)"
  say "已有旧版本，改名保留 → $(basename "$BAK")"
  mv "$DEST" "$BAK"
fi
rsync -a --exclude='._*' --exclude='.DS_Store' "$SRC/" "$DEST/"
ok "仓库就位"

# ---- 2. tev1 官方评测集 ----
if [ -d "$TEV1_DIR/.git" ]; then
  say "tev1 已存在，pull…"
  git -C "$TEV1_DIR" pull --ff-only || warn_pull=1
else
  say "克隆 tev1 官方评测集…"
  git clone https://github.com/togethercomputer/tev1.git "$TEV1_DIR" || warn_clone=1
fi
if [ -f "$TEV1_DIR/data/v1/records/test.jsonl" ]; then
  ok "评测集就位（test $(wc -l < "$TEV1_DIR/data/v1/records/test.jsonl" | tr -d ' ') 条）"
else
  printf '\033[1;33m!\033[0m 评测集缺失，实验一跑不了。手动执行：\n'
  printf '    cd %s && git clone https://github.com/togethercomputer/tev1.git .\n' "$TEV1_DIR"
fi

# ---- 3. venv + 依赖 ----
say "建 venv → $VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -q --upgrade pip

IS_AS=0
[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] && IS_AS=1

if [ "$IS_AS" = "1" ]; then
  say "装依赖（mlx / mlx-lm / transformers / numpy）…"
  "$VENV_DIR/bin/pip" install -q numpy transformers mlx mlx-lm
else
  say "装依赖（numpy / transformers）…"
  "$VENV_DIR/bin/pip" install -q numpy transformers
  printf '\033[1;33m!\033[0m 非 Apple Silicon：轨道 B（latent/RSI）需要 Apple Silicon\n'
fi
ok "依赖装好"

# ---- 4. ollama 检查 ----
if command -v ollama >/dev/null 2>&1; then
  ok "ollama 在位"
  ollama list 2>/dev/null | grep -q "qwen3.5:4b" || \
    printf '\033[1;33m!\033[0m 缺模型：ollama pull qwen3.5:4b\n'
else
  printf '\033[1;33m!\033[0m 缺 ollama（实验一要用）：brew install ollama && ollama pull qwen3.5:4b\n'
fi

cat <<EOF

$(printf '\033[1;32m✓\033[0m 全部就绪')

跑实验（复制即用）：

  # 实验一 · 零训练基线，目标复现 67.3%
  cd "$TEV1_DIR"
  TEV1_REPO="$TEV1_DIR" "$VENV_DIR/bin/python" "$DEST/scripts/eval_local.py" \\
    --model qwen3.5:4b --records data/v1/records/test.jsonl --output /tmp/out --concurrency 3

  # 轨道 B · 头训练 + RSI 循环
  "$VENV_DIR/bin/python" "$DEST/scripts/latent_extract.py" \\
    --records "$TEV1_DIR/data/v1/records/train.jsonl" --out /tmp/lt.npz --limit 500 --max-len 256
  "$VENV_DIR/bin/python" "$DEST/scripts/rsi_loop.py" \\
    --train /tmp/lt.npz --test /tmp/lv.npz --record /tmp/rsi.jsonl

本地说明：README.md · known-issues.md（11 个真踩过的坑）

EOF
