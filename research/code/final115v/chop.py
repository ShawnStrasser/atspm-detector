import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import duckdb, sys
from pathlib import Path
O = Path(DCW + r"\s115v\out")
for t in ["ref114_full", "v6b_full", "v7whl_full", "ref114_le2h", "v7whl_le2h"]:
    f = (O / f"{t}_det.parquet").as_posix()
    print(t, duckdb.sql(f"""select count(*) n, list(distinct "case")[:6] cases,
        list(regexp_extract(health_reason, 'jump around ([0-9]+)x', 1))[:4] ratio, list(health_status)[:4] st
        from '{f}' where regexp_matches(health_reason, 'jump around [0-9]{{5,}}x')""").fetchall())
