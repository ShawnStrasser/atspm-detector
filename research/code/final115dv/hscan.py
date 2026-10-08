import re, duckdb, glob, os
for f in sorted(glob.glob('out/*_det.parquet')):
    d = duckdb.sql(f"select \"case\", Detector, health_status, health_reason, health_watch from '{f}'").df()
    txt = d.health_reason.fillna('') + ' ' + d.health_watch.fillna('')
    nums = txt.map(lambda s: max([float(x) for x in re.findall(r'(\d+(?:\.\d+)?(?:e[+-]?\d+)?)x\b', s.replace(',', ''))] or [0]))
    print(os.path.basename(f), 'max ratio %.3g' % nums.max(), 'n>100x', int((nums > 100).sum()), d.health_status.value_counts().to_dict())
    if nums.max() > 100: print(d.loc[nums.idxmax(), ['case', 'Detector', 'health_reason']].to_dict())
