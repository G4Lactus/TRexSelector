#!/bin/bash
# Stage A driver (host): export the committed HEAD, run stage_ubuntu.sh in an
# ubuntu:24.04 container (native arm64 on Apple silicon), snapshot the
# provisioned container as trex-orb-ubuntu:deps for reruns.
#
#   tools/orbstack/run_ubuntu.sh            fresh container, apt included
#   tools/orbstack/run_ubuntu.sh --reuse    from the snapshot, skips apt
#
# Work dir: $TREX_ORB_WORKDIR (default ${TMPDIR:-/tmp}/trex-orbstack);
# logs in <workdir>/logs, summary in <workdir>/logs/summary.log.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${TREX_ORB_WORKDIR:-${TMPDIR:-/tmp}/trex-orbstack}
NAME=trex-orb-ubuntu
PLATFORM=${TREX_ORB_PLATFORM:-linux/arm64}

mkdir -p "$WORK/logs"
rm -rf "$WORK/repo" && mkdir -p "$WORK/repo"
( cd "$ROOT" && git archive HEAD | tar -x -C "$WORK/repo" )
echo "exported $(git -C "$ROOT" rev-parse --short HEAD) to $WORK/repo"

docker rm -f "$NAME" > /dev/null 2>&1 || true
if [ "${1:-}" = "--reuse" ]; then
    IMAGE="$NAME:deps"; ARGS="--no-apt"
else
    IMAGE="ubuntu:24.04"; ARGS=""
fi
docker run --platform "$PLATFORM" --name "$NAME" \
    -v "$WORK/repo:/work/repo" -v "$WORK/logs:/work/logs" \
    -v "$HERE/stage_ubuntu.sh:/work/stage_ubuntu.sh:ro" \
    "$IMAGE" bash /work/stage_ubuntu.sh $ARGS
rc=$?
if [ "${1:-}" != "--reuse" ]; then
    docker commit "$NAME" "$NAME:deps" > /dev/null && echo "snapshot: $NAME:deps"
fi
echo "--- summary ($WORK/logs/summary.log) ---"
cat "$WORK/logs/summary.log"
exit $rc
