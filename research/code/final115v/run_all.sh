PY=${PY:-python}
VP=${DC_WORK:-$HOME/dc_work}/s115v/venv/Scripts/python.exe
export OMP_NUM_THREADS=4
cd ${DC_WORK:-$HOME/dc_work}/s115v
for prof in full le2h; do
  $PY runv.py ref114 ${DC_WORK:-$HOME/dc_work}/s115/ref114 $prof > log_ref114_$prof.txt 2>&1
  $PY runv.py v6b ${DC_WORK:-$HOME/dc_work}/final_v3_candidate_v6b $prof > log_v6b_$prof.txt 2>&1
  $PY runv.py v7src src:${DC_WORK:-$HOME/dc_work}/final_v7_prod/src $prof > log_v7src_$prof.txt 2>&1
  (cd ${DC_WORK:-$HOME/dc_work}/s115v/out && $VP ../runv.py v7whl installed $prof > ../log_v7whl_$prof.txt 2>&1)
done
echo ALLDONE
