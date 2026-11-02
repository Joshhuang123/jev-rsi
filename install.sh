#!/usr/bin/env bash
# jev-rsi · 在新机器上一键复现实验环境
#
#   bash install.sh              # 默认装到 ~/WorkBuddy/jev-rsi
#   bash install.sh --dir /path  # 指定安装目录
#
# 装完会打印下一步命令。不会自动跑评测（跑评测很吃资源，先看 README）。

set -euo pipefail

REPO_URL="https://github.com/Joshhuang123/jev-rsi.git"
TEV1_URL="https://github.com/togethercomputer/tev1.git"
INSTALL_DIR="${1:-$HOME/WorkBuddy/jev-rsi}"
VENV_DIR="$INSTALL_DIR/.venv"

say()  { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. 前置检查
say "检查前置依赖…"
command -v python3 >/dev/null 2>&1 || die "找不到 python3，请先装 Python 3.10+"
command -v git     >/dev/null 2>&1 || die "找不到 git"

PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VER" in
  3.[1-9]) : ;;
  3.10)    : ;;
  *)       warn "python3 = $PY_VER，建议 3.10+（本项目在 3.14 上跑通过）" ;;
esac
say "python3 $PY_VER · git $(git --version | awk '{print $3}')"

# mlx 只在 Apple Silicon macOS 上有
IS_APPLE_SILICON=0
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  IS_APPLE_SILICON=1
fi

# ---------------------------------------------------------------- 2. 主仓
if [ -d "$INSTALL_DIR/.git" ]; then
  say "仓库已存在，pull 最新…"
  git -C "$INSTALL_DIR" pull --ff-only
else
  say "克隆主仓 → $INSTALL_DIR"
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

# ---------------------------------------------------------------- 3. tev1 数据集
TEV1_DIR="${TEV1_DIR:-$HOME/WorkBuddy/tev1}"
if [ -d "$TEV1_DIR/.git" ]; then
  say "tev1 数据集已存在，pull 最新…"
  git -C "$TEV1_DIR" pull --ff-only || warn "tev1 pull 失败，用现有版本继续"
else
  say "克隆 tev1（官方评测集在 data/v1/records/）→ $TEV1_DIR"
  git clone "$TEV1_URL" "$TEV1_DIR" || warn "tev1 克隆失败：评测集缺失则实验一跑不了"
fi

if [ -f "$TEV1_DIR/data/v1/records/test.jsonl" ]; then
  say "评测集就位：test=$(wc -l < "$TEV1_DIR/data/v1/records/test.jsonl" | tr -d ' ') 条"
else
  warn "评测集缺失：$TEV1_DIR/data/v1/records/test.jsonl"
fi

# ---------------------------------------------------------------- 4. venv
say "创建 venv → $VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -q --upgrade pip

say "安装依赖…"
if [ "$IS_APPLE_SILICON" = "1" ]; then
  "$VENV_DIR/bin/pip" install -q numpy transformers mlx mlx-lm
  say "已装 mlx / mlx-lm（Apple Silicon 加速）"
else
  "$VENV_DIR/bin/pip" install -q numpy transformers
  warn "非 Apple Silicon：没装 mlx。轨道 B（latent 抽取 / RSI）需要 Apple Silicon 或 CUDA"
fi

# ollama 是轨道 A 的传输层
if command -v ollama >/dev/null 2>&1; then
  say "检测到 ollama $(ollama --version 2>/dev/null | head -1)"
  if ollama list 2>/dev/null | grep -q "qwen3.5:4b"; then
    say "模型 qwen3.5:4b 已在位"
  else
    warn "缺模型：ollama pull qwen3.5:4b"
  fi
else
  warn "缺 ollama（轨道 A 要用）：brew install ollama && ollama pull qwen3.5:4b"
fi

# ---------------------------------------------------------------- 5. 收尾
cat <<EOF

$(printf '\033[1;32m✓\033[0m 环境就绪')

下一步：

  # 实验一 · 零训练基线（复现 67.3%）
  cd "$TEV1_DIR"
  TEV1_REPO="$TEV1_DIR" "$VENV_DIR/bin/python" "$INSTALL_DIR/scripts/eval_local.py" \\
    --model qwen3.5:4b --records data/v1/records/test.jsonl --output /tmp/out --concurrency 3

  # 轨道 B · 头训练 + RSI 循环
  "$VENV_DIR/bin/python" "$INSTALL_DIR/scripts/latent_extract.py" \\
    --records "$TEV1_DIR/data/v1/records/train.jsonl" --out /tmp/lt.npz --limit 500 --max-len 256
  "$VENV_DIR/bin/python" "$INSTALL_DIR/scripts/rsi_loop.py" \\
    --train /tmp/lt.npz --test /tmp/lv.npz --record /tmp/rsi.jsonl

先读 README.md 和 known-issues.md —— 里面记了 11 个真踩过的坑，能省几个小时。

EOF
