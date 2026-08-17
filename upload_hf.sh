#!/bin/bash
# 上传 IndexMem checkpoint 到 HuggingFace，并加入 indexmem collection
#
# 用法:
#   1. 在下方填好 checkpoint 路径（不需要的保持注释），然后: bash upload_hf.sh
#   2. 或直接传参（repo名=本地路径，可为文件或目录）:
#        bash upload_hf.sh Qwen-3-8B-gqa_indexer_scalar-stage1=/path/to/final.pt
#   3. 可选环境变量: HF_USER / COLLECTION_TITLE / PRIVATE=1（私有 repo）

set -euo pipefail

# ===== 待上传的 checkpoint（不需要的保持注释）=====
# e2e_ckpt=
# distill_ckpt=

scalar_ckpt=/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/stage1/final.pt

# ===== HuggingFace 配置 =====
export HF_TOKEN="${HF_TOKEN:-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}"
HF_USER="${HF_USER:-}"                        # 留空则按 token 自动识别
COLLECTION_TITLE="${COLLECTION_TITLE:-indexmem}"
PRIVATE="${PRIVATE:-0}"

# ===== repo 名 -> 本地路径 映射 =====
declare -A CKPTS=()

# 从文件路径推导默认 repo 名:
#   目录 -> 目录名
#   文件 -> <祖父目录>-<父目录>, 如 .../Qwen-3-8B-gqa_indexer_scalar/stage1/final.pt -> Qwen-3-8B-gqa_indexer_scalar-stage1
derive_name() {
  local p="$1" parent grandparent
  if [[ -d "$p" ]]; then
    basename "$p"
  else
    parent=$(basename "$(dirname "$p")")
    grandparent=$(basename "$(dirname "$(dirname "$p")")")
    if [[ -n "$grandparent" && "$grandparent" != "/" ]]; then
      echo "${grandparent}-${parent}"
    else
      basename "$p" | sed 's/\.[^.]*$//'
    fi
  fi
}

# HF repo 名只允许 [A-Za-z0-9._-]，且不能以 . 或 - 开头
sanitize_name() {
  echo "$1" | tr ' ' '-' | sed 's/[^A-Za-z0-9._-]/-/g; s/^[.-]*//'
}

add_ckpt() {
  local path="$1" name="$2"
  [[ -z "$path" ]] && return 0
  if [[ ! -e "$path" ]]; then
    echo "[skip] 路径不存在: $path" >&2
    return 0
  fi
  name=$(sanitize_name "$name")
  CKPTS["$name"]="$path"
}

# 兼容旧的变量写法
add_ckpt "${e2e_ckpt:-}"     "$(derive_name "${e2e_ckpt:-/x}" 2>/dev/null || true)"
add_ckpt "${distill_ckpt:-}" "$(derive_name "${distill_ckpt:-/x}" 2>/dev/null || true)"
add_ckpt "${scalar_ckpt:-}"  "$(derive_name "${scalar_ckpt:-/x}" 2>/dev/null || true)"

# 命令行参数: name=path（优先级最高，可覆盖同名条目）
for arg in "$@"; do
  [[ "$arg" == *=* ]] || { echo "[error] 参数需为 name=path 形式: $arg" >&2; exit 1; }
  add_ckpt "${arg#*=}" "${arg%%=*}"
done

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "没有待上传的 checkpoint，请在脚本中填写路径或用 name=path 传参。" >&2
  exit 1
fi

# ===== 识别 HF 用户名 =====
if [[ -z "$HF_USER" ]]; then
  HF_USER=$(hf auth whoami --format json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['user'])")
fi
echo "HF 用户: $HF_USER"

# ===== 找到或创建 indexmem collection =====
COLLECTION_SLUG=$(python3 - "$COLLECTION_TITLE" "$HF_USER" <<'EOF'
import sys
from huggingface_hub import HfApi
title, owner = sys.argv[1], sys.argv[2]
api = HfApi()
for c in api.list_collections(owner=owner):
    if c.title.lower() == title.lower():
        print(c.slug)
        break
else:
    print(api.create_collection(title=title).slug)
EOF
)
echo "Collection: https://huggingface.co/collections/$COLLECTION_SLUG"

# ===== 逐个上传并加入 collection =====
upload_extra=()
[[ "$PRIVATE" == "1" ]] && upload_extra+=(--private)

for name in "${!CKPTS[@]}"; do
  path="${CKPTS[$name]}"
  repo_id="$HF_USER/$name"
  echo "==== 上传 $path -> $repo_id ===="
  if [[ -f "$path" ]]; then
    hf upload "$repo_id" "$path" "$(basename "$path")" \
      --commit-message "Upload $name" "${upload_extra[@]}"
  else
    hf upload "$repo_id" "$path" \
      --commit-message "Upload $name" "${upload_extra[@]}"
  fi
  python3 - "$COLLECTION_SLUG" "$repo_id" <<'EOF'
import sys
from huggingface_hub import HfApi
slug, item = sys.argv[1], sys.argv[2]
HfApi().add_collection_item(slug, item, item_type="model", exists_ok=True)
print(f"已加入 collection: {item}")
EOF
done

echo "全部完成: https://huggingface.co/collections/$COLLECTION_SLUG"
