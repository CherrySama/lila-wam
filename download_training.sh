#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
project_root="$PWD"
if ! command -v gh >/dev/null 2>&1; then
    echo "请先安装 GitHub CLI，并运行 gh auth login 登录可访问此私有仓库的账号。" >&2
    exit 1
fi
gh auth status --hostname github.com >/dev/null
bundle_dir="$project_root/.cache/training-download"
mkdir -p "$bundle_dir"
gh release download panthera-training-data-v1 \
    --repo CherrySama/lila-wam --dir "$bundle_dir" --skip-existing \
    --pattern 'LiLa-WAM-panthera-training.tar.gz.part-*' \
    --pattern SHA256SUMS \
    --pattern 'LiLa-WAM-panthera-training.tar.gz.sha256'
cd "$bundle_dir"
sha256sum -c SHA256SUMS
cat LiLa-WAM-panthera-training.tar.gz.part-?? > LiLa-WAM-panthera-training.tar.gz.partial
mv LiLa-WAM-panthera-training.tar.gz.partial LiLa-WAM-panthera-training.tar.gz
sha256sum -c LiLa-WAM-panthera-training.tar.gz.sha256
# Keep the current Git checkout's code; extract only data and frozen vision weights.
tar -xzf LiLa-WAM-panthera-training.tar.gz --strip-components=1 \
    -C "$project_root" --wildcards 'LiLa-WAM/data/*' 'LiLa-WAM/dinov3_pretrain/*'
echo "数据和 DINO 权重已下载、校验并解压到项目中。"
