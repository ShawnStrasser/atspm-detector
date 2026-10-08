#!/usr/bin/env bash
# note 69: ./run69.sh [kxN] TAG FOLD SEED [tcn69_func train args...]  -> train + infer (K=4); kxN also infers with N long pieces
set -uo pipefail
cd "$(dirname "$0")"; export DC_WORK="$PWD/work"
W=${WORKERS:-6}; S=code/research/code/neural/tcn69_func.py
mkdir -p work/tcn53/logs work/tcn53/runs
KX=""; case "$1" in kx*) KX=${1#kx}; shift;; esac
TAG=$1; FOLD=$2; SEED=$3; shift 3
python $S train --tag $TAG --fold $FOLD --seed $SEED --workers $W "$@" >> work/tcn53/logs/${TAG}_f${FOLD}_train.log 2>&1 || exit 1
python $S infer --tag $TAG --fold $FOLD --workers $W >> work/tcn53/logs/${TAG}_f${FOLD}_infer.log 2>&1 || exit 1
if [ -n "$KX" ]; then python $S infer --tag $TAG --fold $FOLD --klong $KX --otag ${TAG}k$KX --workers $W >> work/tcn53/logs/${TAG}k${KX}_f${FOLD}_infer.log 2>&1; fi
