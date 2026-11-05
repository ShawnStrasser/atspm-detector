"""The runner of the SQL templates (successor of health_duckdb._Q): an ordered list of named steps -> one WITH query
(text), or a script of temp tables (script), with today's rules (note 154ab): a step used by more than one later step
becomes a temp table, a step used once is written into the step that uses it, each table is dropped once nothing needs
it.  Step names are prefixed when rendered, so they never meet a table of the caller's; sources (the caller's tables)
are bound by placeholder (@EVENTS@ ...) after the renaming.

Templates mark their steps with a line `-- @step <name>` (`-- @step <name> mat`: a materialised CTE in text()); the
SQL between two markers is the step.  A step whose rows go to a store table also has a line `-- @emit <store table>` right
after its marker: script() always writes it as a temp table."""
from __future__ import annotations

import re

from . import render as R

MARK = re.compile(r"^[ \t]*-- @step (\w+)( mat)?[ \t]*$", re.M)
EMIT = re.compile(r"-- @emit (\w+)[ \t]*\r?\n")
SOURCE = re.compile(r"@[A-Z]+@")
TOKEN = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|\w+|[^\s\w]")


def steps(text: str) -> list:
    """(name, sql, mat) of each step of a rendered template, in order."""
    parts = MARK.split(text)
    if parts[0].strip():
        raise ValueError(f"SQL before the first step marker: {parts[0].strip()[:80]!r}")
    return [(parts[i], parts[i + 2].strip(), bool(parts[i + 1])) for i in range(1, len(parts), 3)]


def tokens(sql: str) -> list:
    """the SQL as tokens: whitespace between tokens ignored, a comment's inner whitespace collapsed (two texts with the
    same tokens are the same query)."""
    return [" ".join(t.split()) if t.startswith("--") else t for t in TOKEN.findall(sql)]


class Plan:
    """an ordered list of named steps (CTEs) -> one WITH query, or a script of temp tables."""

    def __init__(self, prefix: str, opt: dict | None = None):
        self.prefix = prefix
        self.ctes: list = []
        self.subs: dict = {}          # @EVENTS@ etc. -> the caller's table names, put in after the renaming
        self.opt: dict = dict(opt or {})
        self.emits: dict = {}         # step -> the store table its rows belong to

    def add(self, name: str, sql: str, mat: bool = False):
        self.ctes.append((name, sql.strip(), mat))

    def render(self, template: str, **ctx):
        """render a template and add its steps."""
        for n, s, m in steps(R.render(template, **ctx)):
            e = EMIT.match(s)
            if e:
                self.emits[n] = e.group(1)
                s = s[e.end():]
            self.add(n, s, m)

    def _renamer(self):
        names = sorted((n for n, _, _ in self.ctes), key=len, reverse=True)
        rx = re.compile(r"\b(" + "|".join(names) + r")\b")
        P = self.prefix

        def rn(sql):
            sql = rx.sub(lambda m: P + m.group(1), sql)
            for k, v in self.subs.items():
                sql = sql.replace(k, v)
            return sql
        return rn

    def _body(self) -> str:
        rn, P = self._renamer(), self.prefix
        return ",\n".join(f"{P}{n} AS {'MATERIALIZED ' if m else ''}(\n{rn(s)}\n)" for n, s, m in self.ctes)

    def text(self, final: str, first: "Plan | None" = None) -> str:
        """one WITH query ending in `final`; first: a plan whose steps come before these (a different prefix)."""
        head = f"{first._body()},\n" if first is not None else ""
        return f"WITH {head}{self._body()}\n{self._renamer()(final)}"

    def script(self, finals, keep: tuple = ()) -> tuple:
        """(statements, drops): a step used by more than one later step becomes a temp table, a step used once is
        written into the statement that uses it; drops[i] = the temp tables no longer needed after statement i.
        `finals`: the final query, or a list of them (the last statements, in that order); `keep`: steps the caller
        reads after the script (never in drops: the caller drops them)."""
        finals = [finals] if isinstance(finals, str) else list(finals)
        P = self.prefix
        fk = [f"__final{i}" for i in range(len(finals))]
        rn = self._renamer()
        names = [n for n, _, _ in self.ctes]
        body = {n: s for n, s, _ in self.ctes}
        words = {n: set(re.findall(r"\w+", s)) for n, s in body.items()}
        for k, f in zip(fk, finals):
            words[k] = set(re.findall(r"\w+", f))
        uses = {n: [] for n in names}
        for n in names + fk:
            for o in names:
                if o != n and o in words[n]:
                    uses[o].append(n)
        table = {n for n in names if len(uses[n]) != 1 or n in self.emits}

        def deps(n, seen):
            out = []
            for o in names:
                if o in table or o in seen or o not in words[n]:
                    continue
                seen.add(o)
                out += deps(o, seen) + [o]
            return out
        def main(sql, w):       # a step with a WITH of its own, under the WITH of the steps written into it: a subquery
            return f"SELECT * FROM (\n{sql}\n)" if w and re.match(r"\s*WITH\b", sql, re.I) else sql
        stmts, made = [], []
        for n in names:
            if n not in table:
                continue
            inl = sorted(set(deps(n, set())), key=names.index)
            w = ("WITH " + ",\n".join(f"{P}{o} AS (\n{rn(body[o])}\n)" for o in inl) + "\n") if inl else ""
            stmts.append(f"CREATE OR REPLACE TEMP TABLE {P}{n} AS {w}{main(rn(body[n]), w)}")
            made.append(n)
        for k, f in zip(fk, finals):
            inl = sorted(set(deps(k, set())), key=names.index)
            w = ("WITH " + ",\n".join(f"{P}{o} AS (\n{rn(body[o])}\n)" for o in inl) + "\n") if inl else ""
            stmts.append(w + rn(f))
        # last statement that reads each table (directly, or through a step written into it)
        last = {}
        for i, st in enumerate(stmts):
            w = set(re.findall(r"\w+", st.split(" AS ", 1)[1] if i < len(made) else st))
            for n in made:
                if P + n in w:
                    last[n] = i
        drops = [[P + n for n in made if n not in keep and last.get(n, made.index(n)) == i] for i in range(len(stmts))]
        return stmts, drops

    def lint(self, allow_alias: tuple = ()) -> list:
        """problems of the plan's SQL (design 2.5), [] when none: an arrow lambda (DuckDB 2 removed `->`), an engine
        object (CREATE MACRO / VIEW, query_table), a placeholder no source is bound to, a step read before it is made
        or never made, a step name used as a column (`x.name`), as an alias (`AS name`) or inside a string literal -- the
        renamer would rename it too (allow_alias: today's names that do so consistently)."""
        out = []
        names = [n for n, _, _ in self.ctes]
        names_rx = re.compile(r"\b(" + "|".join(names) + r")\b")
        made = set()
        for n, s, _ in self.ctes:
            low = s.lower()
            if "->" in s:
                out.append(f"{n}: arrow lambda")
            for bad in ("create macro", "create view", "create or replace view", "create temp view", "query_table"):
                if bad in low:
                    out.append(f"{n}: {bad}")
            for src in SOURCE.findall(s):
                if src not in self.subs:
                    out.append(f"{n}: source {src} not bound")
            w = set(re.findall(r"\w+", s))
            for o in names:
                if o != n and o in w and o not in made:
                    out.append(f"{n}: reads {o} before it is made")
            for o in names:
                if o in allow_alias:
                    continue
                if re.search(rf"\.\s*{o}\b|\b(?i:AS)\s+{o}\b(?!\s*\()", s):     # case-sensitive, as the renamer
                    out.append(f"{n}: step name {o} used as a column or alias")
            for lit in re.findall(r"'(?:[^']|'')*'", s):                         # the renamer does not spare literals
                if names_rx.search(lit) and names_rx.search(lit).group(1) not in allow_alias:
                    out.append(f"{n}: step name in the literal {lit}")
            made.add(n)
        return out
