# NOTE: make sure CEPH_ROOT exists and you have prepared env under $CEPH_ROOT/envs

# Check required environment variables
REQUIRED_VARS=(TAIJI_BASIC_CODE_PATH CEPH_ROOT TAIJI_BASIC_OUTPUT_PATH TJ_TASK_NAME)
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: Required environment variable $var is not set." >&2
        return 1 2>/dev/null || exit 1
    fi
done

# enter the code directory
cd $TAIJI_BASIC_CODE_PATH

# reuse envs from ceph by writing .pixi/config.toml
# Set pixi detached-environments to ENV_PATH if defined
export ENV_PATH=$CEPH_ROOT/envs
echo "Setting pixi detached-environments to: $ENV_PATH"
if command -v pixi >/dev/null 2>&1; then
    pixi config set detached-environments "$ENV_PATH"
else
    echo "WARNING: pixi is not installed; skipping detached-environments configuration." >&2
fi

# Persist pixi's package/wheel cache on ceph too. The default (~/.cache) lives on
# the container overlay, which is wiped on restart -> the flash-attn source build
# (no torch-2.9 wheel exists) would recompile every time. Pinning the cache to
# ceph means it compiles ONCE and every later pod reuses the built wheel.
# PIXI_DISABLE_NETFS_REDIRECT=1 is REQUIRED: ceph is a FUSE netfs, and pixi would
# otherwise redirect the cache to a local (ephemeral) disk, defeating persistence.
export PIXI_CACHE_DIR=$ENV_PATH/pixi-cache
export PIXI_DISABLE_NETFS_REDIRECT=1
echo "Setting pixi cache dir to: $PIXI_CACHE_DIR"

# The env under $ENV_PATH is prebuilt. In containers with no external network,
# `pixi run` would re-solve/revalidate the lockfile (hitting conda channel + pypi)
# and fail. Freeze mode uses the lockfile as-is and skips install to stay offline.
# Default: OFF (let pixi re-solve, needed after dependency changes).
# Enable by passing `--use-frozen-env` when sourcing this script, e.g.
#   source scripts/taiji/init_shim.sh --use-frozen-env && pixi run -e dev ...
# or by exporting SPARSEX_PIXI_FREEZE=1 beforehand.
for arg in "$@"; do
    case "$arg" in
        --use-frozen-env) SPARSEX_PIXI_FREEZE=1 ;;
    esac
done
if [ "${SPARSEX_PIXI_FREEZE:-0}" = "1" ]; then
    echo "Pixi freeze mode ON: using lockfile as-is, skipping re-solve/install (offline)."
    export PIXI_FROZEN=true
    export PIXI_NO_INSTALL=true
fi

# Proxy settings. The taiji containers have no direct external network, so the
# proxy is always on. Override the address via SPARSEX_PROXY_ADDR.
PROXY_ADDR="${SPARSEX_PROXY_ADDR:-http://star-proxy.oa.com:3128}"
echo "Proxy ON: $PROXY_ADDR"
export http_proxy="$PROXY_ADDR"
export https_proxy="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export HTTPS_PROXY="$PROXY_ADDR"
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,mirrors.cloud.tencent.com"
export NO_PROXY=$no_proxy

# use these envs in your job script 
export OUTPUT_PATH=$TAIJI_BASIC_OUTPUT_PATH
export LOG_PATH=$TAIJI_BASIC_OUTPUT_PATH/log
export CKPT_SAVE_PATH=$TAIJI_BASIC_OUTPUT_PATH/ckpt
export EVENT_PATH=$TAIJI_BASIC_OUTPUT_PATH/events
export EXP_NAME=$TJ_TASK_NAME

# soft links for readability of experiments
# NOTE: use -sfn instead of -s to avoid creating a self-referencing symlink inside $OUTPUT_PATH
# when the link already exists. -n prevents ln from dereferencing an existing symlink (treating
# it as a directory), and -f allows overwriting the existing link.
mkdir -p exps && ln -sfn $OUTPUT_PATH  exps/$EXP_NAME # for login debugging

LONGTERM_EXP_DIR=$CEPH_ROOT/user/$RTX_NAME/experiments
mkdir -p $LONGTERM_EXP_DIR && ln -sfn $OUTPUT_PATH $LONGTERM_EXP_DIR/$EXP_NAME # for long-term maintainence

# fix the stupid bash env set by taiji, it breaks the pixi env setup and make the "pixi run" use system env
unset BASH_ENV

# Raise the open-file descriptor limit. Taiji containers default to a low soft
# limit; DCP checkpoint save opens many *.distcp shards concurrently across ranks
# and trips "OSError: [Errno 24] Too many open files" mid-save at the first
# save_steps (killing the job). Sourced here, so the higher limit is inherited by
# pixi run -> train.sh -> torchrun -> python. Best-effort; ignore if unpermitted.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
