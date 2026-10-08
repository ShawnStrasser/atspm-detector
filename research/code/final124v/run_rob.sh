D=$(cd "$(dirname "$0")" && pwd)
U=$(cygpath -u "$USERPROFILE")
W=$(cygpath -u "${DC_WORK:-$U/dc_work}")
VP=$W/s124v/venv/Scripts/python.exe
export OMP_NUM_THREADS=4 DC_TREE_THREADS=4 DC_NET_THREADS=4
mkdir -p $W/s124v/run && cd $W/s124v/run
TAG=whl_v08 $VP $D/robust_a.py > ../log_rob_a_v08.txt 2>&1
TAG=whl_v05new EVF=$W/s124v/ev/v05.parquet EVG=$W/s124v/ev/v07.parquet WIN_S="2026-09-20 10:00:00" WIN_E="2026-09-20 13:00:00" $VP $D/robust_a.py > ../log_rob_a_v05.txt 2>&1
$VP $D/robust_b.py whl > ../log_rob_b.txt 2>&1
$VP $D/robust_c.py whl > ../log_rob_c.txt 2>&1
$VP $D/robust_d.py > ../log_rob_d.txt 2>&1
OMP_NUM_THREADS=1 DC_TREE_THREADS=1 DC_NET_THREADS=1 CASE=n08 TAG=whl $VP $D/weekn.py > ../log_week_n08.txt 2>&1
OMP_NUM_THREADS=1 DC_TREE_THREADS=1 DC_NET_THREADS=1 EVDIR=$W/s124v/ev CASE=v09 TAG=whl $VP $D/weekn.py > ../log_week_v09.txt 2>&1
echo ROBDONE
