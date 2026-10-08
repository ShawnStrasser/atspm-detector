#!/usr/bin/env bash
# pops job lines from /root/queue.txt (flock) and runs run69.sh; results copied to /workspace/out69
export PYTHONPATH=/workspace/pylib
cd /root/pkg
ID=$1
O=/workspace/out69; mkdir -p $O/fpreds $O/ppreds $O/logs $O/models
while true; do
  JOB=$(flock /root/queue.lock bash -c 'l=$(head -n1 /root/queue.txt); [ -n "$l" ] && sed -i 1d /root/queue.txt; echo "$l"')
  [ -z "$JOB" ] && break
  echo "$(date +%T) w$ID start $JOB" >> /root/jobs.log
  if ./run69.sh $JOB; then st=ok; else st=FAIL; fi
  set -- $JOB; case "$1" in kx*) shift;; esac; T=${1}_f${2}
  cp work/tcn53/fpreds/${1}*_f${2}.parquet $O/fpreds/ 2>/dev/null; cp work/tcn53/ppreds/${1}*_f${2}.parquet $O/ppreds/ 2>/dev/null
  cp work/tcn53/logs/${1}*_f${2}_* work/tcn53/runs/$T.done.json $O/logs/ 2>/dev/null; cp work/tcn53/models/$T.pt $O/models/ 2>/dev/null
  echo "$(date +%T) w$ID $st $JOB" >> /root/jobs.log
  cp /root/jobs.log $O/jobs_$(hostname).log
done
echo "$(date +%T) w$ID exit" >> /root/jobs.log
cp /root/jobs.log $O/jobs_$(hostname).log
