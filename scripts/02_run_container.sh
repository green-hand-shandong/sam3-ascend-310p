#!/usr/bin/env bash
# 启动 SAM3 310P 适配测试容器（长驻，exec 进入）
#
# 前置：宿主机需安装 Ascend Docker Runtime（默认 runtime=ascend），
#       容器通过 ASCEND_VISIBLE_DEVICES 选择卡。
# 注意：驱动按"容器命名空间"独占 NPU —— 一张卡只能属于一个容器，
#       新容器对已占用卡会报 aclInit 507899 (Resource Busy)。详见 docs/tutorial.md FAQ。
#
# 用法: bash 02_run_container.sh            # 创建/启动并进入
#       bash 02_run_container.sh exec       # 进入已存在的容器
#       bash 02_run_container.sh run <cmd>  # 在容器内执行一次性命令
set -euo pipefail

NAME=${CONTAINER_NAME:-sam3-310p}
IMAGE=${CONTAINER_IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0-310p}
NPU_DEV=${NPU_DEV:-9}           # 用 npu-smi info 确认空闲卡: Hugepages 0/0 且无进程的卡可被新容器认领
MODEL_HOST_DIR=${MODEL_DEST:-/data1/models}

case "${1:-up}" in
  up)
    docker inspect "$NAME" >/dev/null 2>&1 && docker start "$NAME" || \
    docker run -itd \
      --name "$NAME" \
      -e ASCEND_VISIBLE_DEVICES="$NPU_DEV" \
      -v "$MODEL_HOST_DIR":/data1/models \
      -v "$(cd "$(dirname "$0")/.." && pwd)":/workspace \
      --shm-size=16g \
      "$IMAGE" sleep infinity
    exec docker exec -it "$NAME" bash
    ;;
  exec)
    exec docker exec -it "$NAME" bash
    ;;
  run)
    shift
    docker exec "$NAME" bash -c "$*"
    ;;
esac
