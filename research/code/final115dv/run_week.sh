PY=${PY:-python}
VN=$DC_WORK/s115dv/sm_new/venv/Scripts/python.exe
export OMP_NUM_THREADS=4 DC_TREE_THREADS=4 DC_NET_THREADS=4 CASE=n08
cd $DC_WORK/s115dv/sm_new/run
for p in full le2h; do
  TAG=ref114 PKG=$DC_WORK/s115/ref114 $PY ../../weekn.py $p
  TAG=v7whl $VN ../../weekn.py $p
done
