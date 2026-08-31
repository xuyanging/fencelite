#!/usr/bin/env bash
# 从零把 fence_lite 装到一台 Ubuntu 机器上。幂等，可重复跑。
#
# 当前 systemd profile 按生产机 8 vCPU / 15 GiB / Ubuntu 22.04 固化。
# 换机器规格时先调整 deploy/fence-parallel-jobs.conf，不能直接套用并发值。
#
#   bash deploy/setup_ubuntu.sh
#
# **不搬任何数据**。data/ projects/ _jobs/ 都是空的开始。
# 要迁移已算好的缓存，见本文件末尾的说明 —— 不能直接 rsync 完就完事。
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/fence_lite}"
NODE_VERSION="${NODE_VERSION:-v24.14.1}"
NODE_HOME="${NODE_HOME:-$HOME/.local/opt}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1/6 系统依赖"
# python3.10-venv 单独装：Ubuntu 的 python3 自带 venv 模块常常是残的
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip curl xz-utils

say "2/6 node $NODE_VERSION"
# 边车用了 ??= 和 .at(-1)，需要 node >= 16.6。Ubuntu 22.04 仓库里的是 v12，跑不了。
# 版本必须与开发机一致：arrows_signature 不含 node 版本，换了解释器结果变了
# 缓存也不会失效 —— 差异会静默地进结果。
if [ ! -x "$NODE_HOME/node-$NODE_VERSION/bin/node" ]; then
  mkdir -p "$NODE_HOME"
  curl -fsSL -o /tmp/node.tar.xz \
    "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz"
  rm -rf "/tmp/node-$NODE_VERSION-linux-x64"
  tar xf /tmp/node.tar.xz -C /tmp
  mv "/tmp/node-$NODE_VERSION-linux-x64" "$NODE_HOME/node-$NODE_VERSION"
  rm -f /tmp/node.tar.xz
fi
# 已有的真目录挪开保留，不覆盖（可能是别的服务在用）
if [ -d "$NODE_HOME/node" ] && [ ! -L "$NODE_HOME/node" ]; then
  mv "$NODE_HOME/node" "$NODE_HOME/node-previous-$(date +%s)"
fi
ln -sfn "$NODE_HOME/node-$NODE_VERSION" "$NODE_HOME/node"
"$NODE_HOME/node/bin/node" --version

say "3/6 宿主 venv"
cd "$APP_DIR"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt
# PyMuPDF 必须是钉死的那个版本：page.get_drawings() 的输出跨版本会变，
# 而 sig_of 不含任何库版本 —— 差异既不失效缓存也不报 stale，只会静默移位。
./venv/bin/python -c "import fitz; print(fitz.__doc__.splitlines()[0])"

say "4/6 线型边车 venv（独立环境，绝对不要和宿主合并）"
# 宿主钉 PyMuPDF 1.27.x，边车要 1.28.x —— 两个版本的 get_drawings() 输出不同。
cd "$APP_DIR/tools/linetype_sidecar"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt
cd "$APP_DIR"

say "5/6 .env"
if [ ! -f "$APP_DIR/.env" ]; then
  cat >&2 <<'MSG'
缺 .env。建一个（权限 600），至少要有：
    GEMINI_API_KEY=...
    ANTHROPIC_API_KEY=...      # 只在切 Claude provider 时需要
core/config.py 只从 .env 补充当前进程里缺少的变量；systemd Environment
和 EnvironmentFile 已经提供的同名值优先，别在任一处放占位密钥。
MSG
  exit 1
fi
chmod 600 "$APP_DIR/.env"

say "6/6 systemd"
sudo cp "$APP_DIR/deploy/fence_lite.service" /etc/systemd/system/fence_lite.service
sudo install -d /etc/systemd/system/fence_lite.service.d
sudo cp "$APP_DIR/deploy/fence-lite-web-threads.conf" \
  /etc/systemd/system/fence_lite.service.d/web-threads.conf
sudo cp "$APP_DIR/deploy/fence-parallel-jobs.conf" \
  /etc/systemd/system/fence_lite.service.d/parallel-jobs.conf
sudo cp "$APP_DIR/deploy/fence-upload-limit.conf" \
  /etc/systemd/system/fence_lite.service.d/upload-limit.conf
sudo cp "$APP_DIR/deploy/fence-linetype-refresh.service" \
  /etc/systemd/system/fence-linetype-refresh.service
sudo cp "$APP_DIR/deploy/fence-linetype-refresh.timer" \
  /etc/systemd/system/fence-linetype-refresh.timer
sudo systemctl daemon-reload
sudo systemctl enable --now fence_lite
sudo systemctl enable --now fence-linetype-refresh.timer

say "自检"
./venv/bin/python - <<'PY'
import hashlib
from steps.linetypes import sidecar
from steps import arrows
from core import hw
print("机器规格      :", hw.describe())
print("线型边车解释器:", sidecar._PYTHON, sidecar._PYTHON.is_file())
print("边车依赖版本  :", sidecar.dep_versions())
print("  → 出现 'missing' 就说明 site-packages 布局没找对，"
      "engine_digest 会和开发机不一致、全站线型缓存作废")
d = sidecar.engine_digest()
print("engine_digest :", d)
print("缓存签名分量  : e" + hashlib.sha1(str(d).encode()).hexdigest()[:12])
print("  → 必须与本次发布验证记录的 digest 相同；不要跨 digest 复用线型缓存")
print("箭头边车 node :", arrows._NODE, arrows.sidecar_available())
PY

cat <<'MSG'

装完了。冒烟验证：
    systemctl is-active fence_lite
    curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:5051/
    curl -s -F "pdf=@某份.pdf" http://127.0.0.1:5051/api/upload
    # 跑完一页后确认两个接缝阶段真的落盘了：
    ls data/<slug>/arrows.json data/<slug>/linetypes/
    # 停服后不应有孤儿边车：
    sudo systemctl stop fence_lite && pgrep -af 'linetype_sidecar|arrow_sidecar'

迁移已算好的缓存（可选，别直接 rsync 完就完事）：
  pdf_revision = 字节数 + mtime_ns，是全部签名的根。tar/docker COPY/对象存储
  同步都会改 mtime → vlm_identity 全部对不上 → **已付费的 VLM raw 全部作废**。
  正确做法：rsync data/ 和 projects/ 之后，在**目标机**上逐个项目跑
      venv/bin/python tools/resign_local_snapshot.py <slug>
  base_P*.jpg 不用搬（按新 revision 自动重生）。
  data/<slug>/linetypes/ 那一层重签不了，会 stale —— 纯本地 CPU、零模型费用，
  重算即可。
MSG
