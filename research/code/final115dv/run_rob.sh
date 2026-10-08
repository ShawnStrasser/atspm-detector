VP=$DC_WORK/s115dv/sm_new/venv/Scripts/python.exe
export OMP_NUM_THREADS=4 DC_TREE_THREADS=4 DC_NET_THREADS=4
cd $DC_WORK/s115dv/sm_new/run
TAG=whl_v08 $VP ../../robust_a.py > ../../log_rob_a_v08.txt 2>&1
TAG=whl_n05 EVF=$DC_WORK/s115dv/ev/n05.parquet EVG=$DC_WORK/s115dv/ev/n07.parquet WIN_S="2026-09-20 15:00:00" WIN_E="2026-09-20 18:00:00" $VP ../../robust_a.py > ../../log_rob_a_n05.txt 2>&1
$VP ../../robust_b.py whl > ../../log_rob_b.txt 2>&1
$VP ../../robust_c.py whl > ../../log_rob_c.txt 2>&1
echo ROBDONE
