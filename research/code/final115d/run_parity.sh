PY=${PY:-python}
VP=$DC_WORK/s115d/smoke/venv/Scripts/python.exe
SRC=$DC_WORK/final_v7_prod/src
export OMP_NUM_THREADS=4
cd $DC_WORK/s115d
$PY par/par115d.py run v7d src:$SRC full > par/log_v7d_full.txt 2>&1 &
$PY par/par115d.py run v7d src:$SRC le2h > par/log_v7d_le2h.txt 2>&1 &
$PY v/runv115d.py v7d src:$SRC full > v/log_v7d_full.txt 2>&1 &
$PY v/runv115d.py v7d src:$SRC le2h > v/log_v7d_le2h.txt 2>&1 &
wait
wait
echo ALLDONE
(cd v/out && $VP ../runv115d.py v7dwhl installed full > ../log_v7dwhl_full.txt 2>&1) &
(cd v/out && $VP ../runv115d.py v7dwhl installed le2h > ../log_v7dwhl_le2h.txt 2>&1) &
wait
echo WHLDONE
