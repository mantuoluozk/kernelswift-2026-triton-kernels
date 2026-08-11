#!/usr/bin/env bash
set -euo pipefail

# 当前 BW1000 验证环境。可在调用时通过同名环境变量覆盖。
CONTAINER_NAME="${CONTAINER_NAME:-zk-triton-0810}"
IMAGE_NAME="${IMAGE_NAME:-harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10}"
PROJECT_DIR="${PROJECT_DIR:-/data/zk/kernelswift-2026-triton-kernels}"
PLATFORM_DIR="$PROJECT_DIR/platforms/bw1000"

# 2026-08-10 在服务器上核对到的镜像摘要：
# harbor.sourcefind.cn:5443/dcu/admin/base/pytorch@sha256:07c285c51837d76fbcc73b771b1a95ecf2d8b71203e5feadb3cb9cb12b5d3f4d

container_status() {
    docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null
}

if status="$(container_status)"; then
    if [[ "$status" != "running" ]]; then
        docker start "$CONTAINER_NAME" >/dev/null
    fi
else
    docker run -dit \
        --name "$CONTAINER_NAME" \
        --network host \
        --privileged \
        --device /dev/kfd \
        --device /dev/dri \
        --device /dev/mkfd \
        --ipc host \
        --shm-size 256G \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --ulimit stack=-1:-1 \
        --ulimit memlock=-1:-1 \
        --user root \
        --workdir "$PLATFORM_DIR" \
        --volume /dev/infiniband:/dev/infiniband \
        --volume /opt/hyhal:/opt/hyhal:ro \
        --volume /data1:/data1 \
        --volume /data:/data \
        "$IMAGE_NAME" /bin/bash >/dev/null
fi

exec docker exec -it --workdir "$PLATFORM_DIR" "$CONTAINER_NAME" /bin/bash
