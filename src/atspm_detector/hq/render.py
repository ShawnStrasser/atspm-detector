"""The Jinja environment of the SQL templates (sql/*.sql) and the literal filters.

Templates hold plain SQL split into steps by marker lines `-- @step <name> [mat]`; Python decides which template runs
with which options (the plan), the templates only switch whole pieces of SQL on options and repeat pieces in loops.
Literals reach the SQL only through the filters below, each the helper the modules used before the move:
  dbl     a float as a DOUBLE literal (NaN too)          str      a quoted string literal
  inlist  ('a', 'b') of quoted strings                    intlist  (1, 2) of integers
  values  a VALUES table from rows of Python values (str -> quoted, float -> DOUBLE, int, bool, None -> NULL)
  ts      TIMESTAMP 'yyyy-mm-dd hh:mm:ss[.ffffff]' (time zone dropped)   ns  epoch nanoseconds (time zone dropped)
  tstext  str(pd.Timestamp(x)) (the ped modules' window literal)        sqlbool  true / false
  epoch_s epoch seconds of pd.Timestamp(x) as a DOUBLE literal (repr; UTC for a time with a zone, as before)
"""
from __future__ import annotations

from pathlib import Path

import jinja2
import pandas as pd

SQL_DIR = Path(__file__).resolve().parent / "sql"


def dbl(x) -> str:
    """a float as a DOUBLE literal."""
    return f"CAST({float(x)!r} AS DOUBLE)" if x == x else "CAST('nan' AS DOUBLE)"


def sqlstr(x) -> str:
    return "'" + str(x).replace("'", "''") + "'"


def inlist(xs) -> str:
    return "(" + ", ".join(sqlstr(x) for x in xs) + ")"


def intlist(xs) -> str:
    return "(" + ", ".join(str(int(x)) for x in xs) + ")"


def lit(v) -> str:
    """a Python value as an SQL literal: None -> NULL, bool -> true / false, int, float -> DOUBLE, str -> quoted."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return dbl(v)
    return sqlstr(v)


def values(rows, cols: str) -> str:
    """a VALUES table (cols: 'a, b, c') from rows of Python values; no rows: an empty table of NULL columns."""
    if not rows:
        n = len(cols.split(","))
        return f"SELECT * FROM (VALUES ({', '.join(['NULL'] * n)})) t({cols}) WHERE false"
    return f"SELECT * FROM (VALUES {', '.join('(' + ', '.join(lit(v) for v in r) + ')' for r in rows)}) t({cols})"


def _naive(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize(None) if t.tzinfo is not None else t


def ns(x) -> int:
    return int(_naive(x).value)


def ts(x) -> str:
    return f"TIMESTAMP '{_naive(x).isoformat(sep=' ')}'"


def tstext(x) -> str:
    return str(pd.Timestamp(x))


def epoch_s(x) -> str:
    return repr(pd.Timestamp(x).value / 1e9)


def sqlbool(x) -> str:
    return str(bool(x)).lower()


ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(str(SQL_DIR)), undefined=jinja2.StrictUndefined,
                         autoescape=False, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
                         auto_reload=False)
ENV.filters.update(dbl=dbl, str=sqlstr, inlist=inlist, intlist=intlist, values=values, ts=ts, ns=ns, tstext=tstext,
                   epoch_s=epoch_s, sqlbool=sqlbool)


def render(name: str, **ctx) -> str:
    """the rendered text of template `name` (a path under sql/)."""
    return ENV.get_template(name).render(**ctx)
