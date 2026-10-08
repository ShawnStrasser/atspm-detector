PY=${PY:-python}
export OMP_NUM_THREADS=4 DC_TREE_THREADS=4 DC_NET_THREADS=4
cd $DC_WORK/s115dv
$PY runn.py ref114 $DC_WORK/s115/ref114 full > log_ref114_full.txt 2>&1 &
$PY runn.py v7src src:$DC_WORK/final_v7_prod/src full > log_v7src_full.txt 2>&1 &
wait
$PY runn.py ref114 $DC_WORK/s115/ref114 le2h > log_ref114_le2h.txt 2>&1 &
$PY runn.py v7src src:$DC_WORK/final_v7_prod/src le2h > log_v7src_le2h.txt 2>&1 &
wait
echo ALLDONE
