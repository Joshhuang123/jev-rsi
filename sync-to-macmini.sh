#!/usr/bin/env bash
# jev-rsi · 同步到另一台设备（Mac mini / 任意 Mac）
#
#   bash sync-to-macmini.sh              # 通过 GitHub（推荐，需要那台设备能上网）
#   bash sync-to-macmini.sh --via-t7     # 通过 T7 移动硬盘（离线，插上盘直接拷）
#
# 用法二需要：T7 已挂载在 /Volumes/T7

set -euo pipefail

REPO_URL="https://github.com/Joshhuang123/jev-rsi.git"
DEST="${JEVRSI_DEST:-$HOME/WorkBuddy/jev-rsi}"
VIA_T7=0
[ "${1:-}" = "--via-t7" ] && VIA_T7=1

say()  { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

if [ "$VIA_T7" = "1" ]; then
  SRC="/Volumes/T7/智海论衡/工作档案/jev-rsi"
  [ -d "$SRC" ] || die "T7 上没找到 jev-rsi 目录：$SRC（盘挂上了吗？）"
  say "从 T7 拷 → $DEST"
  mkdir -p "$(dirname "$DEST")"
  if [ -d "$DEST" ]; then
    say "目标已存在，先备份…"
    mv "$DEST" "${DEST}.bak.$(date +%Y%m%d-%H%M%S)"
  fi
  rsync -a "$SRC/" "$DEST/"
  ok "拷贝完成 → $DEST"
else
  if [ -d "$DEST/.git" ]; then
    say "已存在，pull 最新…"
    git -C "$DEST" pull --ff-only
  else
    say "克隆 → $DEST"
    mkdir -p "$(dirname "$DEST")"
    git clone "$REPO_URL" "$DEST"
  fi
  ok "GitHub 同步完成 → $DEST"
fi

cat <<EOF

下一步（装环境 + 跑实验）：

  bash "$DEST/install.sh"

EOF
