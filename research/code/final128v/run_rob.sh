U=$(cygpath -u "$USERPROFILE")
W=$(cygpath -u "${DC_WORK:-$U/dc_work}")
PY=${PY:-$U/venvs/detector-classifier/Scripts/python.exe}
C=$W/s128v/code
export OMP_NUM_THREADS=4 DC_TREE_THREADS=4 DC_NET_THREADS=4 PYTHONDONTWRITEBYTECODE=1
mkdir -p $W/s128v/robrun && cd $W/s128v/robrun
for pk in prod next; do
  S=$W/s128v/pkg/${pk}_src
  PKG=$S TAG=$pk $PY $C/robust_a.py > a_$pk.log 2>&1
  SRC=$S $PY $C/robust_b.py $pk > b_$pk.log 2>&1
  SRC=$S $PY $C/robust_c.py $pk > c_$pk.log 2>&1
  SRC=$S $PY $C/robust_d.py > d_$pk.log 2>&1
done
echo ROBDONE
