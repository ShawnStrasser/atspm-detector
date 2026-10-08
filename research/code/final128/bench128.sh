#!/bin/bash
# note 128 bench: bench124.py (fresh process, cold + 3 warm, peak working set), old vs new interleaved
U=$(cygpath -u "$USERPROFILE")
PY=${PY:-$U/venvs/detector-classifier/Scripts/python.exe}
B=$(cd "$(dirname "$0")/../final124" && pwd)/bench124.py
OUT=$1
for L in m30 h3 h24; do for s in typical_r8 typical_r11 busiest_ev busiest_ch; do for pk in final_v7_prod final_v7_next; do
  echo "load $(powershell -c '(Get-CimInstance Win32_Processor).LoadPercentage')" >> $OUT.load
  $PY $B ../$pk/src ../bench71/ev_${s}_${L}.parquet ${pk}_${s}_${L} >> $OUT
done; done; done
for pk in final_v7_prod final_v7_next; do
  echo "load $(powershell -c '(Get-CimInstance Win32_Processor).LoadPercentage')" >> $OUT.load
  $PY $B ../$pk/src ../s124v/rob/week_n08.parquet ${pk}_n08_d7 1 >> $OUT
done
