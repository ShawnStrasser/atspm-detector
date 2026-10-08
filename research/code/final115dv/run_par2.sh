PY=${PY:-python}
VN=$DC_WORK/s115dv/sm_new/venv/Scripts/python.exe
V1=$DC_WORK/s115dv/sm_121/venv/Scripts/python.exe
export OMP_NUM_THREADS=4 DC_TREE_THREADS=4 DC_NET_THREADS=4
cd $DC_WORK/s115dv
$PY runn.py v7src src:$DC_WORK/final_v7_prod/src le2h > log_v7src_le2h.txt 2>&1 &
(cd sm_new/run && $VN ../../runn.py v7whl installed full > ../../log_v7whl_full.txt 2>&1; $VN ../../runn.py v7whl installed le2h > ../../log_v7whl_le2h.txt 2>&1) &
(cd sm_121/run && $V1 ../../runn.py v7ort121 installed full > ../../log_v7ort121_full.txt 2>&1; $V1 ../../runn.py v7ort121 installed le2h > ../../log_v7ort121_le2h.txt 2>&1) &
wait
echo PARDONE
