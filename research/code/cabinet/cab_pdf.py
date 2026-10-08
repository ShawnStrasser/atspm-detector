"""Scripted text-layer extraction from vector cabinet prints (PyMuPDF), plus renders for visual reading.

    python cab_pdf.py 01001 01005 ...      -> %DC_WORK%/cabinet/signals/<DN>.extract.json + render/<DN>/*.png

What is scripted (see research/labels/cabinet_print_guide.md):
  * cabinet type (title block);
  * the INPUT FILE: slot -> loop numbers (+ the printed phase marker), in three layouts:
      - 332 / 332S terminal-block drawing (TB2..TB7 rows `I1-D` ... with "Loop No." written beside),
      - 336 input-file front view (slot columns, phase + loops in the D/E and J/K boxes),
      - 332S loop table (loop -> distance ft -> phase -> slot, tied loops joined by a bracket);
  * camera / radar zone labels written on the diagram (`A 42`, `2A 50`, `D-8 23`, `6C/51`);
  * which pages are intersection diagrams, ranked by how many input-file loop numbers they carry
    (a print can hold two diagrams; the one with the loops wins);
  * renders: every candidate diagram page at 110 dpi plus 3x3 zoom tiles at 220 dpi, input-file crops,
    and a thumbnail of EVERY page (`thumb_p<N>.png`, 45 dpi, text upright) + one labelled contact sheet
    (`contact.png`) — the diagram ranking misses zone sheets, so readers look at all pages first.
Lanes, lane types, loop position (stop bar / advance / departure / bike) are read visually from the renders.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict

import pymupdf

from cab_common import RENDER_DIR, SIG_DIR, SLOT_TO_DET, copy_local, device_id

NUM = re.compile(r"^\d{1,3}(?:[,.&]\d{1,3})*[,.&]?$")  # "3," "4" as two words is common
TERM = re.compile(r"^([IJ])(\d{1,2})-([DEJK])$")


def _words(page):
    seen, out = set(), []
    for w in page.get_text("words"):
        k = (round(w[0]), round(w[1]), w[4])
        if k not in seen:  # some prints carry every word twice
            seen.add(k)
            out.append(w)
    return out


def _cy(w):
    return (w[1] + w[3]) / 2


def _loops(tokens):
    """['8','9'] or ['3,4'] or ['6.7'] -> '8,9' (sorted numerically, unique)."""
    nums = sorted({int(n) for t in tokens for n in re.split(r"[,.&]", t) if n})
    return ",".join(map(str, nums))


def cabinet_type(doc) -> tuple[str | None, list]:
    hits = Counter()
    for p in doc:
        t = p.get_text()
        for m in re.finditer(r"\b(332S|336S|336|332)\s*(?:ODOT\s*)?(?:CAB(?:INET)?\b|Cab Print)", t, re.I):
            hits[m.group(1).upper().replace("336S", "336")] += 1  # '336S' in a title block = 336 layout
    if not hits:
        return None, []
    return hits.most_common(1)[0][0], hits.most_common()


def parse_input_tb(page) -> list[dict]:
    """332/332S terminal-block drawing: rows `I1-D`.. with loop numbers written left of the block."""
    W = _words(page)
    labels = [w for w in W if TERM.match(w[4]) and TERM.match(w[4]).group(3) in "DJ"]
    rows = []
    for L in labels:
        rel = [(w[0] - L[0], _cy(w) - _cy(L), w) for w in W if w is not L]
        # a real terminal row has its terminal number just left of the label
        if not any(-36 < dx < -12 and abs(dy) < 7 and w[4].isdigit() for dx, dy, w in rel):
            continue
        # loop numbers may wrap onto a second line (dy ~ +17); rows are >= 22 apart
        cand = [(dx, dy, w[4]) for dx, dy, w in rel if -115 < dx < -37 and -4 < dy < 21 and NUM.match(w[4])]
        bike = any(-115 < dx < -37 and -4 < dy < 21 and w[4].upper().startswith("BIKE") for dx, dy, w in rel)
        rows.append((L, cand, bike))
    # per terminal-block column, the printed phase marker sits at a fixed dx (the modal dx)
    col = defaultdict(list)
    for L, cand, _ in rows:
        col[round(L[0] / 20)].extend(round(dx / 3) for dx, _, t in cand if len(t) == 1)
    out = []
    for L, cand, bike in rows:
        m = TERM.match(L[4])
        slot = f"{m.group(1)}{m.group(2)}{'U' if m.group(3) == 'D' else 'L'}"
        mode = Counter(col[round(L[0] / 20)]).most_common(1)
        pdx = mode[0][0] * 3 if mode else -999
        phase = [t for dx, _, t in cand if abs(dx - pdx) <= 5 and len(t) == 1]
        loops = [t for dx, _, t in cand if dx > pdx + 5]
        if loops:
            out.append(dict(slot=slot, loops=_loops(loops), phase_marker=int(phase[0]) if phase else None,
                            bike=bike, layout="tb", xy=[round(L[0]), round(L[1])]))
    return _dedupe(out)


def parse_input_336(page) -> list[dict]:
    """336 front-view input file: slot numbers 1..14 in a row; D/E box = upper input, J/K = lower."""
    W = _words(page)
    heads = [w for w in W if w[4] == "INPUT"]
    out = []
    for h in heads:
        row = [w for w in W if w[4].isdigit() and 0 < _cy(w) - _cy(h) < 25 and 1 <= int(w[4]) <= 14]
        if len(row) < 8:
            continue
        for s in row:
            x = s[0]
            letters = {w[4]: w for w in W if w[4] in "DEJK" and len(w[4]) == 1 and abs(w[0] - x) < 8
                       and 0 < w[1] - s[1] < 120}
            def at(letter):
                if letter not in letters:
                    return None
                y = _cy(letters[letter])
                t = [w[4] for w in W if abs(_cy(w) - y) < 4 and 8 < w[0] - x < 30 and NUM.match(w[4])]
                return t or None
            for up, (pl, ll) in (("U", ("D", "E")), ("L", ("J", "K"))):
                ph, lp = at(pl), at(ll)
                if lp:
                    out.append(dict(slot=f"I{s[4]}{up}", loops=_loops(lp),
                                    phase_marker=int(ph[0]) if ph and ph[0].isdigit() else None,
                                    layout="336", xy=[round(x), round(s[1])]))
    return _dedupe(out)


def parse_loop_table(page) -> list[dict]:
    """332S loop table: Loop Number | Distance Feet | Phase | Slot (tied loops share a bracket)."""
    W = _words(page)
    hd = {w[4]: w for w in W if w[4] in ("Loop", "Distance", "Phase", "Slot")}
    if len(hd) < 4 or abs(hd["Loop"][1] - hd["Slot"][1]) > 12:
        return []
    y0 = max(h[3] for h in hd.values()) + 6
    colx = {k: (hd[k][0] + hd[k][2]) / 2 for k in hd}
    def column(k, pat, tol=14):
        return [(_cy(w), w[4]) for w in W if w[1] > y0 and abs((w[0] + w[2]) / 2 - colx[k]) < tol and re.match(pat, w[4])]
    loops = column("Loop", r"^\d{1,3}$")
    loops = [(y, t) for y, t in loops if y < y0 + 12 * 60]
    ymax = max((y for y, _ in loops), default=y0) + 6
    dist = {round(y): t for y, t in column("Distance", r"^-?\d{1,4}$", 18) if y < ymax}
    phases = [(y, t) for y, t in column("Phase", r"^\d$") if y < ymax]
    slots = [(y, t) for y, t in column("Slot", r"^[IJ]\d{1,2}[UL]$") if y < ymax]
    # brackets: vertical segments between the distance and phase columns
    br = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if abs(a.x - b.x) < 1 and abs(a.y - b.y) > 5 and colx["Distance"] + 10 < a.x < colx["Phase"] - 5 \
                        and y0 - 5 < min(a.y, b.y) < ymax:
                    br.append((min(a.y, b.y), max(a.y, b.y)))
    out = []
    for ys, s in slots:
        ph = min(phases, key=lambda p: abs(p[0] - ys)) if phases else None
        grp = [b for b in br if b[0] - 2 <= ys <= b[1] + 2]
        if grp:
            lo, hi = min(g[0] for g in grp), max(g[1] for g in grp)
            mem = [(y, t) for y, t in loops if lo - 3 <= y <= hi + 3]
        else:
            mem = []
        if not mem:
            mem = [(y, t) for y, t in loops if abs(y - ys) < 3]
        dd = [dist.get(round(y)) or dist.get(round(y) + 1) or dist.get(round(y) - 1) for y, _ in mem]
        out.append(dict(slot=s, loops=_loops([t for _, t in mem]),
                        phase_marker=int(ph[1]) if ph and abs(ph[0] - ys) < 4 else None,
                        distance_ft=",".join(d for d in dd if d), layout="looptable", xy=[0, round(ys)]))
    return _dedupe(out)


def _dedupe(rows):
    seen, out = set(), []
    for r in rows:
        k = (r["slot"], r["loops"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


ZONE_PATTERNS = [
    re.compile(r"^(?P<dev>\d?[A-H])/(?P<det>\d{1,2})$"),          # 2A/49
    re.compile(r"^(?P<dev>[A-H])-(?P<ph>\d)/(?P<det>\d{1,2})$"),  # D-8/23
]


def zone_labels(page) -> list[dict]:
    """Camera / radar zone labels: `CAM A 42`, `A 42`, `2A 50`, `6C 51`, `D-8 23`, `2A/49`."""
    W = _words(page)
    out = []
    for w in W:
        for pat in ZONE_PATTERNS:
            m = pat.match(w[4])
            if m:
                out.append(dict(label=w[4], device=m.group("dev"), detector=int(m.group("det")), xy=[round(w[0]), round(w[1])]))
        if re.match(r"^(\d[A-H]|[A-H]|[A-H]-\d)$", w[4]):
            nxt = [v for v in W if v is not w and re.match(r"^\d{1,2}$", v[4]) and 0 < v[0] - w[0] < 22
                   and v[0] - w[2] < 12 and abs(_cy(v) - _cy(w)) < 6]
            # also vertical text (rotated pages): next word directly above/below
            nxt += [v for v in W if v is not w and re.match(r"^\d{1,2}$", v[4]) and abs(v[0] - w[0]) < 3 and 0 < w[1] - v[3] < 12]
            for v in nxt[:1]:
                # a bare device letter ('A 42') is only trusted for the video/radar channel range
                if 1 <= int(v[4]) <= 64 and (len(w[4]) > 1 or int(v[4]) >= 29):
                    out.append(dict(label=f"{w[4]} {v[4]}", device=w[4], detector=int(v[4]), xy=[round(w[0]), round(w[1])]))
    return out


def text_direction(page) -> int:
    """Rotation (deg) that makes most text horizontal when applied to the rendered page."""
    c = Counter()
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            dx, dy = l["dir"]
            n = sum(len(s["text"]) for s in l["spans"])
            c[(round(dx), round(dy))] += n
    d = c.most_common(1)[0][0] if c else (1, 0)
    return {(1, 0): 0, (0, -1): 90, (0, 1): -90, (-1, 0): 180}.get(d, 0)


def render(doc, pno: int, out, dpi=110, tiles=3, tile_dpi=220):
    p = doc[pno]
    rot = text_direction(p)
    files = []
    f = out / f"p{pno + 1}.png"
    p.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72).prerotate(rot)).save(f)
    files.append(f.name)
    r = p.rect
    if tiles:
        w, h = r.width / tiles, r.height / tiles
        for i in range(tiles):
            for j in range(tiles):
                clip = pymupdf.Rect(r.x0 + i * w - 0.1 * w, r.y0 + j * h - 0.1 * h,
                                    r.x0 + (i + 1) * w + 0.1 * w, r.y0 + (j + 1) * h + 0.1 * h) & r
                f = out / f"p{pno + 1}_t{j}{i}.png"
                p.get_pixmap(matrix=pymupdf.Matrix(tile_dpi / 72, tile_dpi / 72).prerotate(rot), clip=clip).save(f)
                files.append(f.name)
    return files, rot


def thumbs(doc, out, dpi=45, cols=4, cell=420):
    """Low-dpi thumbnail of every page + one contact sheet with the page numbers written on it."""
    from PIL import Image, ImageDraw
    files, ims = [], []
    for i, p in enumerate(doc):
        f = out / f"thumb_p{i + 1}.png"
        p.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72).prerotate(text_direction(p))).save(f)
        files.append(f.name)
        im = Image.open(f).convert("RGB")
        im.thumbnail((cell, cell))
        ims.append(im)
    if not ims:
        return files, None
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + 10), rows * (cell + 30)), "white")
    d = ImageDraw.Draw(sheet)
    for k, im in enumerate(ims):
        x, y = (k % cols) * (cell + 10), (k // cols) * (cell + 30)
        sheet.paste(im, (x, y + 22))
        d.rectangle([x, y + 22, x + im.width - 1, y + 22 + im.height - 1], outline="gray")
        d.text((x + 4, y + 4), f"page {k + 1}", fill="red")
    sheet.save(out / "contact.png")
    return files, "contact.png"


def crop(doc, pno, rect, name, out, dpi=220):
    p = doc[pno]
    f = out / name
    p.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72).prerotate(text_direction(p)), clip=pymupdf.Rect(*rect)).save(f)
    return f.name


def extract(dn: str, do_render=True) -> dict:
    t0 = time.time()
    dev = device_id(dn)
    files = copy_local(dn)
    rec = dict(DeviceName=dn, DeviceId=dev, pdf=files["pdf"], xlsm=files["xlsm"],
               pdf_versions=files["pdf_versions"], xlsm_versions=files["xlsm_versions"])
    if not files["pdf"]:
        rec["error"] = "no pdf"
        return rec
    doc = pymupdf.open(files["pdf"])
    rec["n_pages"] = len(doc)
    rec["cabinet_type"], rec["cabinet_type_hits"] = cabinet_type(doc)
    inputs, zones, pages = [], [], []
    for i, p in enumerate(doc):
        tb, c336, lt = parse_input_tb(p), parse_input_336(p), parse_loop_table(p)
        for r in tb + c336 + lt:
            r["page"] = i + 1
        inputs += tb + c336 + lt
        txt = p.get_text()
        hint = bool(re.search(r"ROTATION|PREEMPTION|Premption", txt))
        pages.append(dict(page=i + 1, words=len(txt.split()), rotation=text_direction(p),
                          input_rows=len(tb) + len(c336), looptable_rows=len(lt),
                          diagram_hint=hint))
    # detector numbers from the slot table
    cab = rec["cabinet_type"]
    for r in inputs:
        r["detector"] = SLOT_TO_DET.get(cab, {}).get(r["slot"]) if cab else None
    rec["input_file"] = inputs
    rec["zones"] = zones
    loopnums = {int(n) for r in inputs for n in r["loops"].split(",") if n}
    for pg in pages:
        nums = {int(w[4]) for w in _words(doc[pg["page"] - 1]) if w[4].isdigit() and len(w[4]) <= 2}
        pg["loops_on_page"] = len(loopnums & nums)
        # diagram candidates: no input-file rows, few words (not a wiring sheet), and a hint or loop numbers
        txt = doc[pg["page"] - 1].get_text()
        pg["wiring_sheet"] = bool(re.search(r"OUTPUT FILE|POWER DISTRIBUTION|INPUT FILE", txt))
        pg["diagram_candidate"] = (not pg["input_rows"] and pg["words"] < 900
                                   and (pg["diagram_hint"] or pg["loops_on_page"] >= 2))
        z = zone_labels(doc[pg["page"] - 1]) if pg["diagram_candidate"] and not pg["wiring_sheet"] else []
        for r in z:
            r["page"] = pg["page"]
        zones += z
        pg["zones_on_page"] = len(z)
    diag = [p for p in pages if p["diagram_candidate"]]
    diag.sort(key=lambda p: (p["wiring_sheet"], -(p["loops_on_page"] + 2 * p["zones_on_page"]), p["page"]))
    rec["pages"] = pages
    rec["diagram_pages"] = [p["page"] for p in diag]
    rec["diagram_page"] = diag[0]["page"] if diag else None
    rec["loops_in_input_file"] = sorted(loopnums)
    if do_render:
        out = RENDER_DIR / dn
        out.mkdir(parents=True, exist_ok=True)
        rec["renders"] = {}
        rec["thumbs"], rec["contact_sheet"] = thumbs(doc, out)
        for k, p in enumerate(rec["diagram_pages"][:3]):  # zoom tiles on the two best-ranked pages only
            rec["renders"][p], _ = render(doc, p - 1, out, tiles=3 if k < 2 else 0)
        if not any(r["layout"] != "looptable" for r in inputs):  # blank / unparsed input file: whole sheet
            for pg in pages:
                if pg["input_rows"] == 0 and re.search(r"\b[IJ]1-D\b|Channel\s+1\b", doc[pg["page"] - 1].get_text()):
                    f = f"input_p{pg['page']}.png"
                    doc[pg["page"] - 1].get_pixmap(dpi=130).save(out / f)
                    rec["renders"][f"input_p{pg['page']}"] = f
        for pg in {r["page"] for r in inputs}:
            xs = [r for r in inputs if r["page"] == pg and r["layout"] != "looptable"]
            if xs:
                x0 = min(r["xy"][0] for r in xs) - 160; y0 = min(r["xy"][1] for r in xs) - 60
                x1 = max(r["xy"][0] for r in xs) + 60; y1 = max(r["xy"][1] for r in xs) + 80
                rec["renders"][f"input_p{pg}"] = crop(doc, pg - 1, (x0, y0, x1, y1), f"input_p{pg}.png", out, dpi=160)
    rec["extract_seconds"] = round(time.time() - t0, 1)
    (SIG_DIR / f"{dn}.extract.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return rec


def _main():
    for dn in sys.argv[1:]:
        r = extract(dn)
        print(dn, r.get("cabinet_type"), "diagram", r.get("diagram_pages"), "inputs", len(r.get("input_file", [])),
              "zones", len(r.get("zones", [])), f"{r.get('extract_seconds')} s")


def zoom(dn: str, page: int, fx0: float, fy0: float, fx1: float, fy1: float, dpi=250, tag="z", out=None):
    """Crop a region given as fractions of the *rotated* page render (as seen in p<N>.png)."""
    ex = json.loads((SIG_DIR / f"{dn}.extract.json").read_text(encoding="utf-8"))
    doc = pymupdf.open(ex["pdf"])
    p = doc[page - 1]
    rot = text_direction(p)
    M = pymupdf.Matrix(1, 1).prerotate(rot)
    R = p.rect * M  # rendered bounding box in rotated space
    a = pymupdf.Point(R.x0 + fx0 * R.width, R.y0 + fy0 * R.height) * ~M
    b = pymupdf.Point(R.x0 + fx1 * R.width, R.y0 + fy1 * R.height) * ~M
    clip = pymupdf.Rect(min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y))
    out = out or RENDER_DIR / dn / f"p{page}_{tag}.png"
    p.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72).prerotate(rot), clip=clip).save(out)
    return str(out)


if __name__ == "__main__":
    if sys.argv[1] == "zoom":  # zoom DN PAGE fx0 fy0 fx1 fy1 [tag]
        a = sys.argv[2:]
        print(zoom(a[0], int(a[1]), *map(float, a[2:6]), tag=a[6] if len(a) > 6 else "z"))
    else:
        _main()
