# 82d — Health review sheet v1: second chart column "Day chart" (2026-10-05)
User request via orchestrator: per row, the detector's actuations per 15 min over the whole calendar day (00:00-24:00)
on which the problem was flagged; blue line = this detector, orange line = partner (same partner as the main chart),
flagged period shaded; title "Det N - actuations per 15 min, <day>". Code `research/code/health/h82_daychart.py`
(reuses h82_charts3 / h82_charts2 row loop); PNGs `review/health_review_v1_day_charts/` (70, same base names as the
main charts); titles `%DC_WORK%/health82/titles_daychart.csv`.
* Flagged period: stuck = the long ON; silent = the silent stretch; too many in 5 min = busiest 5 min; count drops =
  from the drop to the sample end; misses at night = the sample's nights; busier at night = 00:00-05:00; every other
  check = the sample itself (clipped to the day). Day = calendar day of the flagged period's start.
* Where the saved events end before midnight (some signals stop Mon ~15:35) the rest of the day is greyed and the
  subtitle says so.
* Workbook: backup `review/_backup/health_review_v1_20261005_155545.xlsx`; new column M "Day chart" (header + link per
  row); nothing else touched, existing Chart column and links unchanged (70 / 70 and 70 / 70 resolve). Cell-by-cell vs
  the backup: no existing cell differs. The user had meanwhile typed 11 answers (K4-K14, some with remarks); all 11
  identical after the save.
* Visual check: rows 1, 6, 41 day charts opened; lines, shading and titles as intended.
